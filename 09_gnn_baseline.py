# 09_gnn_baseline.py
# ------------------------------------------------------------
# 時空間GNN（最小版 v0）。町丁目queen隣接をグラフ化し、各四半期スナップショットに
# 2層GCNを適用 → 学習した町表現を物件特徴と結合して平米単価を回帰する。
# 「手作りS-lag(近隣平均)」を「グラフ伝播」に置き換えて勝てるかの最初の確認。
#
# 同一の四半期フォールド(GNN_TEST_QUARTERS)上で2つのXGB参照と比べる:
#   XGB-full   … Qboth + 駅TE + Municipality/Zoning one-hot（強いベースライン）
#   XGB-propT  … 物件特徴 + 自町T-lag のみ（S-lag・駅TE・one-hot無し＝GNNと同程度の情報）
# GNNが XGB-propT を上回れば「グラフ伝播が手作りS-lag的な近隣情報を学べている」と言える。
# ------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb            # torchより先にimport（07と同様、XGB→torchの順で安定）
import torch
sys.path.append("src")
import config as C
from lib_eval import expanding_window_pooled, summarize, trim_by_iqr
from lib_features import StationTargetEncoder
from lib_spatial import load_town_polygons, build_neighbors
from lib_gnn import build_graph, SpatioGCN, Standardizer, train_eval_fold

DEVICE = os.environ.get("GNN_DEVICE", C.GNN_DEVICE)
torch.manual_seed(C.RANDOM_STATE)
np.random.seed(C.RANDOM_STATE)

# 既定は直近8四半期(GNN_TEST_QUARTERS)。GNN_FULL_EVAL=1 で全64四半期(2010Q1〜2025Q4)を
# 評価し、頑健性を確認する。出力ファイルもモードで分ける。
FULL_EVAL = bool(os.environ.get("GNN_FULL_EVAL"))
TEST_QUARTERS = C.EXPANDING_TEST_QUARTERS if FULL_EVAL else C.GNN_TEST_QUARTERS
OUT_NAME = "gnn_vs_xgb_q_full.csv" if FULL_EVAL else "gnn_vs_xgb_q.csv"

print(f"=== 9. 時空間GNN (最小v0, device={DEVICE}, "
      f"{'全64四半期' if FULL_EVAL else '直近8四半期'}={len(TEST_QUARTERS)}fold) ===")
df = pd.read_csv(C.DATA_DIR / "tokyo23_model_table_q.csv")
df = df.dropna(subset=[C.TARGET, "Qidx"]).copy()
df["Qidx"] = df["Qidx"].astype(int)

subgroups = {"Central5": C.CENTRAL_5,
             "Others": [w for w in C.WARDS_23 if w not in C.CENTRAL_5]}

# --- グラフ構築（町丁目queen隣接） ---------------------------------
node_keys = list(df.groupby(["Municipality", "DistrictName"]).groups.keys())
print(f"ノード(町)数: {len(node_keys)}  / 隣接を構築中...")
towns = load_town_polygons(C.SHAPEFILE_PATH, C.WARDS_23)
neighbors = build_neighbors(towns, method=C.SPATIAL_WEIGHT,
                            knn_k=C.KNN_K, projected_crs=C.PROJECTED_CRS)
node_index, edge_index = build_graph(node_keys, neighbors)
deg = np.bincount(edge_index[0].numpy(), minlength=len(node_keys)) if edge_index.numel() else np.zeros(len(node_keys))
print(f"  辺数(有向): {edge_index.shape[1]}  / 隣接ありノード: {(deg > 0).sum()}/{len(node_keys)}")
df["node"] = list(zip(df["Municipality"], df["DistrictName"]))
df["node"] = df["node"].map(node_index)

# --- 町×四半期のノード特徴（自町ラグ＋重心）を事前計算 -------------
N = len(node_keys)
LAGF = ["Lag_q1_AvgPrice", "Lag_q4_AvgPrice"]
cent = (df.groupby("node")[["centroid_x", "centroid_y"]].first()
        .reindex(range(N)))
C_arr = cent.to_numpy(dtype=float)                       # [N,2] 重心（NaNは後で標準化0埋め）
tq = (df.groupby(["node", "Qidx"])[LAGF].first().reset_index())
lag_by_q = {}
for q, g in tq.groupby("Qidx"):
    m = np.full((N, 2), np.nan)
    m[g["node"].to_numpy()] = g[LAGF].to_numpy(dtype=float)
    lag_by_q[int(q)] = m

PROP = C.GNN_PROP_FEATURES


def node_matrix(q):
    """四半期qのノード特徴 [N,4] = [Lag_q1, Lag_q4, cx, cy]（生値・NaNあり）。"""
    return np.concatenate([lag_by_q[q], C_arr], axis=1)


# --- GNN: 四半期フォールドで Expanding Window ----------------------
gnn_records = []
for q in TEST_QUARTERS:
    if q not in lag_by_q:
        continue
    win = C.GNN_TRAIN_WINDOW
    lo = q - win if win else -10**9
    train_q = [t for t in sorted(lag_by_q) if lo <= t < q]
    train_q = [t for t in train_q if (df["Qidx"] == t).any()]
    test_df = df[df["Qidx"] == q]
    if len(train_q) < 2 or len(test_df) < 50:
        continue

    # 訓練物件：直近窓 → 区ごとIQRトリミング（ベースラインと同じ運用）
    train_df = df[df["Qidx"].isin(train_q)]
    if C.IQR_TRIM:
        train_df, _ = trim_by_iqr(train_df, C.TARGET, C.IQR_K, C.IQR_BY_WARD)

    # 標準化器（訓練のみでfit）
    node_std = Standardizer().fit(np.vstack([node_matrix(t) for t in train_q]))
    prop_std = Standardizer().fit(train_df[PROP].to_numpy(dtype=float))
    y_mean = train_df[C.TARGET].mean(); y_std = train_df[C.TARGET].std() or 1.0

    def make_snap(rows, qq):
        return dict(
            x=node_std.transform(node_matrix(qq)),
            node_idx=rows["node"].to_numpy(),
            prop=prop_std.transform(rows[PROP].to_numpy(dtype=float)),
            y=((rows[C.TARGET].to_numpy() - y_mean) / y_std),
        )
    snapshots = {t: make_snap(train_df[train_df["Qidx"] == t], t) for t in train_q}
    snapshots[q] = make_snap(test_df, q)

    model = SpatioGCN(n_node_feat=4, n_prop_feat=len(PROP), hidden=C.GNN_HIDDEN)
    pred_std = train_eval_fold(model, snapshots, edge_index, train_q, q,
                               epochs=C.GNN_EPOCHS, lr=C.GNN_LR, device=DEVICE)
    pred = pred_std * y_std + y_mean

    te = test_df.assign(pred=pred)
    r2_all = r2_score(te[C.TARGET], te["pred"])
    print(f"  Qidx={q} ({q // 4}Q{q % 4 + 1}) 訓練{len(train_df):,}件/"
          f"テスト{len(te):,}件  R²={r2_all:.3f}", flush=True)
    gnn_records.append({"test_year": q, "scope": "ALL", "name": "Tokyo23",
                        "n": len(te), "r2": r2_all, "r2_trim": np.nan})
    for label, wl in subgroups.items():
        g = te[te["Municipality"].isin(wl)]
        if len(g) >= 10:
            gnn_records.append({"test_year": q, "scope": "GROUP", "name": label,
                                "n": len(g), "r2": r2_score(g[C.TARGET], g["pred"]),
                                "r2_trim": np.nan})

gnn_sum = summarize(pd.DataFrame(gnn_records))
gnn_sum.insert(0, "model", "GNN(min v0)")

# --- XGB参照（同一フォールド GNN_TEST_QUARTERS 上で） --------------
ONEHOT = ["Municipality"] + [c for c in C.CAT_FEATURES if c in df.columns]


def run_xgb(num_feats, onehot, use_te, tag):
    pre = ColumnTransformer(
        ([("oh", OneHotEncoder(handle_unknown="ignore"), onehot)] if onehot else [])
        + [("num", "passthrough", num_feats)])
    mk = lambda: Pipeline([("pre", pre), ("reg", xgb.XGBRegressor(**C.XGB_PARAMS))])
    ft = StationTargetEncoder(C.TARGET, m=C.TE_SMOOTHING) if use_te else None
    rec, _ = expanding_window_pooled(
        df, mk, (onehot or []) + num_feats, C.TARGET, TEST_QUARTERS,
        iqr_trim=C.IQR_TRIM, iqr_k=C.IQR_K, iqr_by_ward=C.IQR_BY_WARD,
        subgroups=subgroups, fold_transform=ft, time_col="Qidx", progress=False)
    s = summarize(rec); s.insert(0, "model", tag)
    return s


print("\nXGB参照を同一フォールドで評価中...")
xgb_full = run_xgb(C.BASE_FEATURES + C.DEFAULT_LAG_COLS + ["Station_TE"],
                   ONEHOT, True, "XGB-full(Qboth)")
xgb_propT = run_xgb(PROP + C.DEFAULT_TLAG_COLS, [], False, "XGB-propT(自町T-lagのみ)")

summary = pd.concat([gnn_sum, xgb_full, xgb_propT], ignore_index=True)
out = C.OUT_DIR / OUT_NAME
summary.to_csv(out, index=False, encoding="utf-8-sig")

print(f"\n[全体R² 比較（同一フォールド: {len(TEST_QUARTERS)}四半期）]")
view = summary[summary["scope"] == "ALL"][
    ["model", "r2_mean", "r2_std", "r2_min", "r2_max", "n_folds"]].round(3)
print(view.to_string(index=False))
print("\n[サブグループ別 平均R²]")
print(summary[summary["scope"] == "GROUP"][["model", "name", "r2_mean", "r2_std"]]
      .round(3).to_string(index=False))
print(f"\n保存: {out}")
