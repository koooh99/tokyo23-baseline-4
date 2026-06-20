# 10_gnn_temporal.py
# ------------------------------------------------------------
# 時空間GNN・時間系列版（GCN + GRU）。09の静的スナップショットv0に「時間方向」を足す。
# 各町の直近LQの観測平米単価系列を、各時点GCN(重み共有)→GRUで集約し、直近性を学習する。
# アブレーションで「直近1Q ≫ 前年同期」だったので、時間重みを学べると上振れが期待できる。
#
# 同一の四半期フォールド上で XGB-full(Qboth) と比較。09(静的GCN)の同フォールド結果が
# outputs にあれば併記する。既定は直近8四半期、GNN_FULL_EVAL=1 で全64四半期。
# ------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb            # torchより先にimport（07/09と同様）
import torch
sys.path.append("src")
import config as C
from lib_eval import expanding_window_pooled, summarize, trim_by_iqr
from lib_features import StationTargetEncoder
from lib_spatial import load_town_polygons, build_neighbors
from lib_gnn import build_graph, SpatioTemporalGNN, Standardizer

DEVICE = os.environ.get("GNN_DEVICE", C.GNN_DEVICE)
torch.manual_seed(C.RANDOM_STATE)
np.random.seed(C.RANDOM_STATE)

FULL_EVAL = bool(os.environ.get("GNN_FULL_EVAL"))
RICH = C.GNN_RICH_FEATURES
# 時間集約器は config 既定（"gru"）。環境変数 TEMPORAL_AGG で attention/mean/last に切替可。
AGG = os.environ.get("TEMPORAL_AGG", C.TEMPORAL_AGG)


def _parse_window(env_val, default):
    """学習窓を環境変数 GNN_TRAIN_WINDOW から読む。未指定なら config 既定。
    "none"/"all"/"full"/"0"/"" は全期間(None=Expanding)を意味する。"""
    if env_val is None:
        return default
    v = env_val.strip().lower()
    if v in ("none", "all", "full", "0", ""):
        return None
    return int(v)


# 学習窓（直近何四半期で学習するか）。既定は config の 12。窓実験は環境変数で 12/24/None。
WIN = _parse_window(os.environ.get("GNN_TRAIN_WINDOW"), C.GNN_TRAIN_WINDOW)
WIN_TAG = "full" if WIN is None else str(WIN)
# エポックは config 既定。窓実験を軽く回したいとき環境変数 TEMPORAL_EPOCHS で上書き可
# （3窓で同一epoch＝窓以外の比較軸を増やさない。既定は既知値を再現する 30）。
EPOCHS = int(os.environ.get("TEMPORAL_EPOCHS", C.TEMPORAL_EPOCHS))
TEST_QUARTERS = C.EXPANDING_TEST_QUARTERS if FULL_EVAL else C.GNN_TEST_QUARTERS
_stem = "gnn_temporal_rich" if RICH else "gnn_temporal"
# 既定窓(=config値)のときは従来ファイル名を維持（11の棒/サブグループ図が読む）。
# 非既定窓のときだけ _win{tag} を付けて窓条件を区別する。
WIN_SUFFIX = "" if WIN == C.GNN_TRAIN_WINDOW else f"_win{WIN_TAG}"
OUT_NAME = _stem + ("_q_full" if FULL_EVAL else "_q") + WIN_SUFFIX + ".csv"
_agg_tag = {"gru": "(GRU)", "attention": "(attn)", "mean": "(mean)", "last": "(last)"}
GNN_LABEL = ("GNN-temporal+feat" if RICH else "GNN-temporal") + (
    "" if RICH else _agg_tag.get(AGG, f"({AGG})"))
L = C.TEMPORAL_SEQ_LEN

print(f"=== 10. 時空間GNN・時間系列版 (GCN+{AGG}, L={L}, 学習窓={WIN_TAG}, "
      f"epochs={EPOCHS}, device={DEVICE}, rich={RICH}, "
      f"{'全64四半期' if FULL_EVAL else '直近8四半期'}={len(TEST_QUARTERS)}fold) ===")
df = pd.read_csv(C.DATA_DIR / "tokyo23_model_table_q.csv")
df = df.dropna(subset=[C.TARGET, "Qidx"]).copy()
df["Qidx"] = df["Qidx"].astype(int)

subgroups = {"Central5": C.CENTRAL_5,
             "Others": [w for w in C.WARDS_23 if w not in C.CENTRAL_5]}

# --- グラフ構築 ----------------------------------------------------
node_keys = list(df.groupby(["Municipality", "DistrictName"]).groups.keys())
towns = load_town_polygons(C.SHAPEFILE_PATH, C.WARDS_23)
neighbors = build_neighbors(towns, method=C.SPATIAL_WEIGHT,
                            knn_k=C.KNN_K, projected_crs=C.PROJECTED_CRS)
node_index, edge_index = build_graph(node_keys, neighbors)
edge_index = edge_index.to(DEVICE)
df["node"] = list(zip(df["Municipality"], df["DistrictName"]))
df["node"] = df["node"].map(node_index)
N = len(node_keys)
print(f"  ノード(町)数: {N}  辺数(有向): {edge_index.shape[1]}")

# --- 町×四半期の観測平米単価系列を事前計算（前方補完＋観測フラグ） ---
cent = (df.groupby("node")[["centroid_x", "centroid_y"]].first().reindex(range(N)))
C_arr = cent.to_numpy(dtype=float)                       # [N,2]
price = (df.groupby(["node", "Qidx"])[C.TARGET].mean().reset_index())
qmin, qmax = int(price["Qidx"].min()), int(price["Qidx"].max())
all_q = list(range(qmin, qmax + 1))
qpos = {q: i for i, q in enumerate(all_q)}
P = np.full((len(all_q), N), np.nan)                     # [T, N] 観測平米単価
for r in price.itertuples(index=False):
    P[qpos[int(r.Qidx)], r.node] = getattr(r, C.TARGET)
OBS = (~np.isnan(P)).astype(float)                       # 観測フラグ [T,N]
# 前方補完（各町、過去の最後の観測を持ち越す）
Pf = P.copy()
for t in range(1, len(all_q)):
    m = np.isnan(Pf[t])
    Pf[t, m] = Pf[t - 1, m]


def raw_node_feat(q):
    """四半期qのノード特徴 [N,4] = [前方補完平米単価, 観測フラグ, cx, cy]（生値）。"""
    i = qpos[q]
    return np.column_stack([Pf[i], OBS[i], C_arr])


PROP = C.GNN_PROP_FEATURES


def seq_steps(t):
    """目的四半期tの系列ステップ [t-L, ..., t-1]（過去のみ＝リーク無し）。"""
    return [t - k for k in range(L, 0, -1)]


# --- 時間系列GNN: 四半期Expanding Window ---------------------------
def train_eval_temporal(train_t, test_q, train_df, test_df):
    # 標準化器（訓練の時点・物件のみでfit）
    need_ts = sorted({s for t in (train_t + [test_q]) for s in seq_steps(t)}
                     | set(train_t) | {test_q})
    need_ts = [t for t in need_ts if qmin <= t <= qmax]
    node_std = Standardizer().fit(np.vstack([raw_node_feat(t) for t in train_t]))
    y_mean = train_df[C.TARGET].mean(); y_std = train_df[C.TARGET].std() or 1.0

    # (c) 立地系特徴を物件ヘッドに追加（XGB-fullと同条件にする）。すべて訓練のみでfit。
    num_cols = list(PROP)
    cat_cols, ohe = [], None
    if RICH:
        if C.STATION_TE:                       # 駅名TE（フォールド内fit＝リーク防止）
            ste = StationTargetEncoder(C.TARGET, m=C.TE_SMOOTHING).fit(train_df)
            train_df = ste.transform(train_df); test_df = ste.transform(test_df)
            num_cols = num_cols + ["Station_TE"]
        cat_cols = ["Municipality"] + [c for c in C.CAT_FEATURES if c in train_df.columns]
        ohe = OneHotEncoder(handle_unknown="ignore").fit(train_df[cat_cols])
    prop_std = Standardizer().fit(train_df[num_cols].to_numpy(dtype=float))
    cat_dim = sum(len(c) for c in ohe.categories_) if ohe is not None else 0

    Xstd = {t: torch.tensor(node_std.transform(raw_node_feat(t)),
                            dtype=torch.float32, device=DEVICE) for t in need_ts}

    def prop_pack(rows):
        num = prop_std.transform(rows[num_cols].to_numpy(dtype=float))
        if ohe is not None:
            cat = ohe.transform(rows[cat_cols])
            cat = cat.toarray() if hasattr(cat, "toarray") else cat
            num = np.concatenate([num, cat], axis=1)
        return (torch.tensor(rows["node"].to_numpy(), dtype=torch.long, device=DEVICE),
                torch.tensor(num, dtype=torch.float32, device=DEVICE),
                torch.tensor((rows[C.TARGET].to_numpy() - y_mean) / y_std,
                             dtype=torch.float32, device=DEVICE))
    train_packs = {t: prop_pack(train_df[train_df["Qidx"] == t]) for t in train_t}
    test_pack = prop_pack(test_df)

    model = SpatioTemporalGNN(n_node_feat=4, n_prop_feat=len(num_cols) + cat_dim,
                              hidden=C.GNN_HIDDEN, aggregator=AGG).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=C.TEMPORAL_LR)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for ep in range(EPOCHS):
        # 四半期ごとにstep（GRUの収束には1エポック1stepでは更新が少なすぎるため）。
        # 各stepで対象四半期の系列分だけGCNを通す（L枚）。
        for t in train_t:
            opt.zero_grad()
            H = {s: model.encode(Xstd[s], edge_index) for s in seq_steps(t)}
            seq = torch.stack([H[s] for s in seq_steps(t)], dim=0)   # [L,N,hidden]
            h = model.temporal(seq)
            idx, pf, y = train_packs[t]
            loss = loss_fn(model.predict(h, idx, pf), y)
            loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        H = {t: model.encode(Xstd[t], edge_index) for t in seq_steps(test_q)}
        seq = torch.stack([H[s] for s in seq_steps(test_q)], dim=0)
        h = model.temporal(seq)
        idx, pf, _ = test_pack
        pred = model.predict(h, idx, pf).cpu().numpy() * y_std + y_mean
    return pred


records = []
for q in TEST_QUARTERS:
    win = WIN
    lo = q - win if win else qmin
    train_t = [t for t in all_q if lo <= t < q and (df["Qidx"] == t).any()
               and (t - L) >= qmin]
    test_df = df[df["Qidx"] == q]
    if len(train_t) < 2 or len(test_df) < 50:
        continue
    train_df = df[df["Qidx"].isin(train_t)]
    if C.IQR_TRIM:
        train_df, _ = trim_by_iqr(train_df, C.TARGET, C.IQR_K, C.IQR_BY_WARD)

    pred = train_eval_temporal(train_t, q, train_df, test_df)
    te = test_df.assign(pred=pred)
    r2_all = r2_score(te[C.TARGET], te["pred"])
    print(f"  Qidx={q} ({q // 4}Q{q % 4 + 1}) 訓練{len(train_df):,}件/"
          f"テスト{len(te):,}件  R²={r2_all:.3f}", flush=True)
    records.append({"test_year": q, "scope": "ALL", "name": "Tokyo23",
                    "n": len(te), "r2": r2_all, "r2_trim": np.nan})
    for label, wl in subgroups.items():
        g = te[te["Municipality"].isin(wl)]
        if len(g) >= 10:
            records.append({"test_year": q, "scope": "GROUP", "name": label,
                            "n": len(g), "r2": r2_score(g[C.TARGET], g["pred"]),
                            "r2_trim": np.nan})

temp_sum = summarize(pd.DataFrame(records))
temp_sum.insert(0, "model", GNN_LABEL)

# --- XGB-full(Qboth) 参照（同一フォールド） -----------------------
ONEHOT = ["Municipality"] + [c for c in C.CAT_FEATURES if c in df.columns]
pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), ONEHOT),
                         ("num", "passthrough",
                          C.BASE_FEATURES + C.DEFAULT_LAG_COLS + ["Station_TE"])])
mk = lambda: Pipeline([("pre", pre), ("reg", xgb.XGBRegressor(**C.XGB_PARAMS))])
print(f"\nXGB-full(Qboth) を同一フォールド・同一学習窓(={WIN_TAG})で評価中...")
rec_x, _ = expanding_window_pooled(
    df, mk, ONEHOT + C.BASE_FEATURES + C.DEFAULT_LAG_COLS + ["Station_TE"],
    C.TARGET, TEST_QUARTERS, iqr_trim=C.IQR_TRIM, iqr_k=C.IQR_K,
    iqr_by_ward=C.IQR_BY_WARD, subgroups=subgroups,
    fold_transform=StationTargetEncoder(C.TARGET, m=C.TE_SMOOTHING),
    time_col="Qidx", progress=False, train_window=WIN)
xgb_sum = summarize(rec_x); xgb_sum.insert(0, "model", "XGB-full(Qboth)")

# --- per-fold R²（GNN と XGB を同一窓・同一フォールドで）を保存 -----
# 11 の推移図が「同一窓どうし」で重ね描き＋差分曲線を引くための入力。
gnn_pf = pd.DataFrame([{"test_year": r["test_year"], "r2": r["r2"]}
                       for r in records if r["scope"] == "ALL"])
gnn_pf = gnn_pf.assign(model=GNN_LABEL, window=WIN_TAG)
xgb_pf = (rec_x[rec_x["scope"] == "ALL"][["test_year", "r2"]]
          .assign(model="XGB-full(Qboth)", window=WIN_TAG))
perfold = pd.concat([gnn_pf, xgb_pf], ignore_index=True)
perfold_path = C.OUT_DIR / f"perfold_win{WIN_TAG}.csv"
perfold.to_csv(perfold_path, index=False, encoding="utf-8-sig")
print(f"per-fold保存: {perfold_path.name} "
      f"(GNN {len(gnn_pf)}fold / XGB {len(xgb_pf)}fold, 窓={WIN_TAG})")

summary = pd.concat([temp_sum, xgb_sum], ignore_index=True)
out = C.OUT_DIR / OUT_NAME
summary.to_csv(out, index=False, encoding="utf-8-sig")

print(f"\n[全体R² 比較（同一フォールド: {len(TEST_QUARTERS)}四半期）]")
print(summary[summary["scope"] == "ALL"]
      [["model", "r2_mean", "r2_std", "r2_min", "r2_max", "n_folds"]]
      .round(3).to_string(index=False))
# 同一フォールドの既存結果を併記（静的GCN v0 / 最小特徴の時系列版）
def _ref(path, model_name, label):
    p = C.OUT_DIR / path
    if not p.exists():
        return
    d = pd.read_csv(p)
    row = d[(d["scope"] == "ALL") & (d["model"] == model_name)]
    if not row.empty:
        r = row.iloc[0]
        print(f"  [参考] {label}: {r['r2_mean']:.3f} ± {r['r2_std']:.3f}")

suf = "_q_full.csv" if FULL_EVAL else "_q.csv"
_ref("gnn_vs_xgb" + suf, "GNN(min v0)", "GNN 静的GCN v0 (09)")
_ref("gnn_temporal" + suf, "GNN-temporal(GRU)", "GNN-temporal 最小特徴 (10, rich=False)")
print("\n[サブグループ別 平均R²]")
print(summary[summary["scope"] == "GROUP"][["model", "name", "r2_mean", "r2_std"]]
      .round(3).to_string(index=False))
print(f"\n保存: {out}")
