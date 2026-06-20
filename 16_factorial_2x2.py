# 16_factorial_2x2.py
# ------------------------------------------------------------
# 空間×時間の 2×2 要因デザインを「因果推論的（ceteris paribus なトグル）」に評価する。
# 2因子＝「空間=近隣波及(graph)」×「時間=動態(temporal)」。4セル:
#   (graph=on , temporal=on ) = GCN + GRU          （現行 SpatioTemporalGNN）
#   (graph=on , temporal=off) = GCN + LastStep     （直前1ステップのみ＝時間集約なし）
#   (graph=off, temporal=on ) = per-node Linear + GRU   （近隣メッセージパッシングなし）
#   (graph=off, temporal=off) = per-node Linear + LastStep
# 各セルは1つのトグル以外を完全固定し、セル間の per-fold 差分をその因子の効果として帰属する。
#
# ceteris paribus の生命線:
#   - lib_gnn.assert_factorial_only_diff で「差分は graph/temporal の2箇所だけ」を実行前に検証。
#   - 同一フォールドでは4セル（×seed）が同じ訓練/テスト分割・同じ前処理(標準化/駅TE/one-hot/
#     系列)を共有する。前処理はフォールド内で1回だけ fit し、4セルで使い回す。
#   - 窓・系列・リーク規約は 10/lib_compare と同一: 学習窓 lo=q-WIN、系列 seq_steps=[q-L,…,q-1]
#     （過去のみ）、temporal=off は直前1ステップのみ、標準化/TE/one-hot は訓練フォールド内 fit、
#     テスト四半期の値は訓練にも系列にも入れない。
#
# ※重要な留保（README/冒頭明記）: これは「近隣波及(graph) vs 時間動態(temporal)」の分離で
#   あり、純粋な空間時間分離ではない。ノード特徴に自町の過去価格(前方補完平米単価)を含むため、
#   graph=off でも各町は自分の過去価格を見ている＝「自町の動態」は両 graph 水準に残る。
#   graph トグルが切り分けるのは「近隣（隣接町）からのメッセージパッシングの有無」だけである。
#   この事実を「修正」しようとしてノード特徴を変えないこと（軸名どおりに解釈する）。
#
# 環境変数:
#   GNN_FULL_EVAL=1     全64四半期（既定は直近8四半期で配線確認）
#   GNN_TRAIN_WINDOW    学習窓。既定 config(=12)。"none"/"all"/"full"/"0" で全期間
#   FACTORIAL_SEEDS     例 "42,43,44"（既定 "42"=1 seed）。複数で mean±std を出す
#   GNN_DEVICE          既定 config(=cpu)。MPS はハングし得るため CPU 前提
# 出力:
#   outputs/factorial_perfold[_full][_win{tag}].csv  per-fold（test_year,r2,cell,graph,temporal,seed,scope,n）
#   outputs/factorial_cells[_full][_win{tag}].csv     4セル × scope の R²(mean±std)
#   outputs/factorial_effects[_full][_win{tag}].csv   主効果・交互作用（mean/中央値/勝敗/Wilcoxon p）
# ------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
import torch
sys.path.append("src")
import config as C
from lib_eval import trim_by_iqr
from lib_features import StationTargetEncoder
from lib_compare import build_town_graph
from lib_gnn import SpatioTemporalGNN, Standardizer, assert_factorial_only_diff

DEVICE = os.environ.get("GNN_DEVICE", C.GNN_DEVICE)
FULL_EVAL = bool(os.environ.get("GNN_FULL_EVAL"))
TEST_QUARTERS = C.EXPANDING_TEST_QUARTERS if FULL_EVAL else C.GNN_TEST_QUARTERS
RICH = C.GNN_RICH_FEATURES               # 物件ヘッドに駅TE+区/用途one-hot（XGB-full同条件）
L = C.TEMPORAL_SEQ_LEN
EPOCHS = int(os.environ.get("TEMPORAL_EPOCHS", C.TEMPORAL_EPOCHS))
LR = C.TEMPORAL_LR
HIDDEN = C.GNN_HIDDEN
PROP = C.GNN_PROP_FEATURES


def _parse_window(env_val, default):
    """学習窓を環境変数 GNN_TRAIN_WINDOW から読む（10 と同一規約）。"""
    if env_val is None:
        return default
    v = env_val.strip().lower()
    if v in ("none", "all", "full", "0", ""):
        return None
    return int(v)


WIN = _parse_window(os.environ.get("GNN_TRAIN_WINDOW"), C.GNN_TRAIN_WINDOW)
WIN_TAG = "full" if WIN is None else str(WIN)
SEEDS = [int(s) for s in os.environ.get("FACTORIAL_SEEDS",
                                        str(C.RANDOM_STATE)).split(",") if s.strip()]

# 2×2 セル定義: (ラベル, graph, temporal, spatial, aggregator)
CELLS = [
    ("G+T+", "on",  "on",  "gcn",  "gru"),
    ("G+T-", "on",  "off", "gcn",  "last"),
    ("G-T+", "off", "on",  "none", "gru"),
    ("G-T-", "off", "off", "none", "last"),
]
SUBGROUPS = {"Central5": C.CENTRAL_5,
             "Others": [w for w in C.WARDS_23 if w not in C.CENTRAL_5]}

print(f"=== 16. 空間×時間 2×2 要因デザイン（因果推論的トグル, L={L}, 学習窓={WIN_TAG}, "
      f"epochs={EPOCHS}, device={DEVICE}, rich={RICH}, "
      f"{'全64四半期' if FULL_EVAL else '直近8四半期'}={len(TEST_QUARTERS)}fold, "
      f"seeds={SEEDS}） ===")
if len(SEEDS) == 1:
    print(f"  ※ seed={SEEDS[0]} の単一実行（CPU 軽量化のため）。複数 seed の mean±std は "
          f"FACTORIAL_SEEDS='42,43,44' で取得可。")

df = pd.read_csv(C.DATA_DIR / "tokyo23_model_table_q.csv")
df = df.dropna(subset=[C.TARGET, "Qidx"]).copy()
df["Qidx"] = df["Qidx"].astype(int)

# --- グラフ（町丁目 queen 隣接, lib_compare と同一手順。df に 'node' 列を付与） ---
print("グラフ（町丁目queen隣接）を構築中...")
node_index, edge_index, N = build_town_graph(df)
edge_index = edge_index.to(DEVICE)
print(f"  ノード(町)数: {N}  辺数(有向): {edge_index.shape[1]}")

# --- ceteris paribus の検証: 差分は graph/temporal の2トグルのみ ---
n_prop = len(PROP) + (1 if (RICH and C.STATION_TE) else 0)   # 連続側の概算（OHE はフォールド依存）
info = assert_factorial_only_diff(n_node_feat=4, n_prop_feat=n_prop, hidden=HIDDEN)
print("[2トグルのみ差分の確認] 4セルで head 形状・最終空間表現次元・g1/g2 の(in,out)次元が一致:")
print(f"  共有: hidden={info['shared']['hidden']}, 層数={info['shared']['layers']}, "
      f"g1{tuple(info['shared']['g1_io'])}→g2{tuple(info['shared']['g2_io'])}, "
      f"head_in={info['shared']['head_in']}")
print(f"  graph トグル: on={info['toggles']['graph']['on']} / off={info['toggles']['graph']['off']}")
print(f"  temporal トグル: on={info['toggles']['temporal']['on']} / "
      f"off={info['toggles']['temporal']['off']}")
print(f"  共有ハイパラ: optimizer=Adam, lr={LR}, epochs={EPOCHS}, L={L}, "
      f"窓={WIN_TAG}, seed={SEEDS}（変えるのは graph/temporal の2箇所のみ）")

# --- 町×四半期の観測平米単価系列（前方補完＋観測フラグ。lib_compare と同一） ---
cent = df.groupby("node")[["centroid_x", "centroid_y"]].first().reindex(range(N))
C_arr = cent.to_numpy(dtype=float)
price = df.groupby(["node", "Qidx"])[C.TARGET].mean().reset_index()
qmin, qmax = int(price["Qidx"].min()), int(price["Qidx"].max())
all_q = list(range(qmin, qmax + 1))
qpos = {q: i for i, q in enumerate(all_q)}
P = np.full((len(all_q), N), np.nan)
for r in price.itertuples(index=False):
    P[qpos[int(r.Qidx)], r.node] = getattr(r, C.TARGET)
OBS = (~np.isnan(P)).astype(float)
Pf = P.copy()
for t in range(1, len(all_q)):
    m = np.isnan(Pf[t]); Pf[t, m] = Pf[t - 1, m]


def raw_node_feat(q):
    i = qpos[q]
    return np.column_stack([Pf[i], OBS[i], C_arr])           # [N,4]


def seq_steps(t):
    return [t - k for k in range(L, 0, -1)]                  # [t-L,…,t-1]（過去のみ）


# --- 1セル分の学習＋評価（共有テンソルを受け取り、モデルだけ差し替える） ---
def run_cell(spatial, aggregator, n_prop_feat, Xstd, train_packs, test_pack,
             train_t, test_q, y_mean, y_std, seed):
    torch.manual_seed(seed); np.random.seed(seed)           # セル間で同一 seed から init
    model = SpatioTemporalGNN(n_node_feat=4, n_prop_feat=n_prop_feat, hidden=HIDDEN,
                              aggregator=aggregator, spatial=spatial).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(EPOCHS):
        for t in train_t:                                   # 四半期ごとに step（10 と同一）
            opt.zero_grad()
            H = {s: model.encode(Xstd[s], edge_index) for s in seq_steps(t)}
            seq = torch.stack([H[s] for s in seq_steps(t)], dim=0)   # [L,N,hidden]
            h = model.temporal(seq)                         # GRU/Last は meta を使わない
            idx, pf, y = train_packs[t]
            loss = loss_fn(model.predict(h, idx, pf), y)
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        H = {s: model.encode(Xstd[s], edge_index) for s in seq_steps(test_q)}
        seq = torch.stack([H[s] for s in seq_steps(test_q)], dim=0)
        h = model.temporal(seq)
        idx, pf, _ = test_pack
        pred = model.predict(h, idx, pf).cpu().numpy() * y_std + y_mean
    return pred


# --- フォールドループ（前処理はフォールド内で1回 fit → 4セル×seed で共有） ---
records = []
for q in TEST_QUARTERS:
    lo = q - WIN if WIN else qmin
    train_t = [t for t in all_q if lo <= t < q and (df["Qidx"] == t).any()
               and (t - L) >= qmin]
    test_df = df[df["Qidx"] == q]
    if len(train_t) < 2 or len(test_df) < 50:
        continue
    train_df = df[df["Qidx"].isin(train_t)]
    if C.IQR_TRIM:
        train_df, _ = trim_by_iqr(train_df, C.TARGET, C.IQR_K, C.IQR_BY_WARD)

    # ----- 共有前処理（全セル共通。訓練フォールド内でのみ fit＝リーク防止） -----
    need_ts = sorted({s for t in (train_t + [q]) for s in seq_steps(t)}
                     | set(train_t) | {q})
    need_ts = [t for t in need_ts if qmin <= t <= qmax]
    node_std = Standardizer().fit(np.vstack([raw_node_feat(t) for t in train_t]))
    y_mean = train_df[C.TARGET].mean(); y_std = train_df[C.TARGET].std() or 1.0
    num_cols = list(PROP); cat_cols, ohe = [], None
    _train_df, _test_df = train_df, test_df
    if RICH:
        if C.STATION_TE:                                    # 駅名TE（フォールド内 fit）
            ste = StationTargetEncoder(C.TARGET, m=C.TE_SMOOTHING).fit(train_df)
            _train_df = ste.transform(train_df); _test_df = ste.transform(test_df)
            num_cols = num_cols + ["Station_TE"]
        cat_cols = ["Municipality"] + [c for c in C.CAT_FEATURES if c in _train_df.columns]
        ohe = OneHotEncoder(handle_unknown="ignore").fit(_train_df[cat_cols])
    prop_std = Standardizer().fit(_train_df[num_cols].to_numpy(dtype=float))
    cat_dim = sum(len(c) for c in ohe.categories_) if ohe is not None else 0
    n_prop_feat = len(num_cols) + cat_dim
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
    train_packs = {t: prop_pack(_train_df[_train_df["Qidx"] == t]) for t in train_t}
    test_pack = prop_pack(_test_df)

    # ----- 4セル × seed（共有テンソルを使い回す＝ceteris paribus） -----
    for seed in SEEDS:
        for (label, g_on, t_on, spatial, agg) in CELLS:
            pred = run_cell(spatial, agg, n_prop_feat, Xstd, train_packs, test_pack,
                            train_t, q, y_mean, y_std, seed)
            te = test_df.assign(pred=pred)
            rows = [("ALL", te)]
            for name, wl in SUBGROUPS.items():
                g = te[te["Municipality"].isin(wl)]
                if len(g) >= 10:
                    rows.append((name, g))
            for scope, gg in rows:
                records.append({"test_year": q, "scope": scope, "cell": label,
                                "graph": g_on, "temporal": t_on, "seed": seed,
                                "n": len(gg), "r2": r2_score(gg[C.TARGET], gg["pred"])})
    done = sorted({r["test_year"] for r in records})
    print(f"  Qidx={q} ({q // 4}Q{q % 4 + 1}) 訓練{len(train_df):,}件/"
          f"テスト{len(test_df):,}件  4セル×{len(SEEDS)}seed 完了 "
          f"[{len(done)}/{len(TEST_QUARTERS)}fold]", flush=True)

pf = pd.DataFrame(records)
if pf.empty:
    raise SystemExit("有効なフォールドがありません。")

TAG = ("_full" if FULL_EVAL else "") + (f"_win{WIN_TAG}" if WIN != C.GNN_TRAIN_WINDOW else "")
pf_path = C.OUT_DIR / f"factorial_perfold{TAG}.csv"
pf.to_csv(pf_path, index=False, encoding="utf-8-sig")
print(f"\nper-fold 保存: {pf_path.name}  ({len(pf)}行)")

# --- 集計①: 4セル × scope の R²(mean±std)（seed は先に平均してフォールド単位に） ---
cell_fold = (pf.groupby(["scope", "cell", "graph", "temporal", "test_year"])["r2"]
             .mean().reset_index())                          # seed 平均
cell_sum = (cell_fold.groupby(["scope", "cell", "graph", "temporal"])["r2"]
            .agg(r2_mean="mean", r2_std="std", n_folds="count").reset_index())
SCOPE_ORDER = {"ALL": 0, "Central5": 1, "Others": 2}
CELL_ORDER = {"G+T+": 0, "G+T-": 1, "G-T+": 2, "G-T-": 3}
cell_sum = cell_sum.sort_values(
    ["scope", "cell"], key=lambda s: s.map(SCOPE_ORDER) if s.name == "scope"
    else s.map(CELL_ORDER)).reset_index(drop=True)
cells_path = C.OUT_DIR / f"factorial_cells{TAG}.csv"
cell_sum.to_csv(cells_path, index=False, encoding="utf-8-sig")

print(f"\n[4セルの R²（mean±std, {WIN_TAG} 窓 / seed 平均後フォールド集計）]")
for scope in ["ALL", "Central5", "Others"]:
    sub = cell_sum[cell_sum["scope"] == scope]
    if sub.empty:
        continue
    print(f"  --- {scope} ---")
    for _, r in sub.iterrows():
        print(f"    {r['cell']} (graph={r['graph']},temporal={r['temporal']}): "
              f"R²={r['r2_mean']:.3f} ± {r['r2_std']:.3f} ({int(r['n_folds'])}fold)")


# --- 集計②: 主効果・交互作用（per-fold のペア差で評価） ---
def eff_stats(d):
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    wins = int((d > 0).sum()); losses = int((d < 0).sum())
    if n == 0 or not np.any(d != 0):
        p = np.nan
    else:
        try:
            p = wilcoxon(d).pvalue
        except ValueError:
            p = np.nan
    return dict(n=n, mean=float(d.mean()) if n else np.nan,
                median=float(np.median(d)) if n else np.nan,
                wins=wins, losses=losses, p=p)


eff_rows = []
for scope in ["ALL", "Central5", "Others"]:
    sub = cell_fold[cell_fold["scope"] == scope]
    piv = sub.pivot_table(index="test_year", columns="cell", values="r2")
    if not set(["G+T+", "G+T-", "G-T+", "G-T-"]).issubset(piv.columns):
        continue
    piv = piv.dropna(subset=["G+T+", "G+T-", "G-T+", "G-T-"])
    oo, of = piv["G+T+"], piv["G+T-"]      # graph=on:  temporal on/off
    fo, ff = piv["G-T+"], piv["G-T-"]      # graph=off: temporal on/off
    effects = {
        # 時間動態の主効果 = temporal on−off（graph 2水準で平均）
        "temporal_main": 0.5 * ((oo - of) + (fo - ff)),
        # 近隣波及の主効果 = graph on−off（temporal 2水準で平均）
        "graph_main": 0.5 * ((oo - fo) + (of - ff)),
        # 交互作用 = (on,on)-(on,off) − [(off,on)-(off,off)]
        "interaction": (oo - of) - (fo - ff),
    }
    for name, d in effects.items():
        st = eff_stats(d.to_numpy())
        eff_rows.append({"scope": scope, "effect": name, **st})

eff = pd.DataFrame(eff_rows)
eff_path = C.OUT_DIR / f"factorial_effects{TAG}.csv"
eff.to_csv(eff_path, index=False, encoding="utf-8-sig")

EFF_JP = {"temporal_main": "時間動態の主効果", "graph_main": "近隣波及の主効果",
          "interaction": "交互作用(graph×temporal)"}
print(f"\n[要因分解（per-fold ペア差, Wilcoxon 符号順位 + 勝敗カウント, {WIN_TAG} 窓）]")
for scope in ["ALL", "Central5", "Others"]:
    sub = eff[eff["scope"] == scope]
    if sub.empty:
        continue
    print(f"  --- {scope} ---")
    for _, r in sub.iterrows():
        pstr = "n/a" if pd.isna(r["p"]) else f"{r['p']:.4f}"
        print(f"    {EFF_JP[r['effect']]:<20s} 平均差={r['mean']:+.4f} "
              f"中央値={r['median']:+.4f}  勝(＋){r['wins']}/負(−){r['losses']} "
              f"(n={r['n']})  Wilcoxon p={pstr}")

# Central5 の交互作用を明示（主要結論）
c5 = eff[(eff["scope"] == "Central5") & (eff["effect"] == "interaction")]
if not c5.empty:
    r = c5.iloc[0]
    sign = "正" if r["mean"] > 0 else ("負" if r["mean"] < 0 else "ゼロ")
    pstr = "n/a" if pd.isna(r["p"]) else f"{r['p']:.4f}"
    sig = "有意" if (not pd.isna(r["p"]) and r["p"] < 0.05) else "非有意"
    print(f"\n[主要結論] Central5 の交互作用(graph×temporal): 符号={sign} "
          f"(平均差 {r['mean']:+.4f}), Wilcoxon p={pstr} → {sig}")

print(f"\n保存: {pf_path.name} / {cells_path.name} / {eff_path.name}")
print("留保: ノード特徴に自町の過去価格を含むため、これは近隣波及(graph)と時間動態"
      "(temporal)の分離であり、純粋な空間/時間分離ではない（自町の動態は両 graph 水準に残る）。")
