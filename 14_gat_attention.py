# 14_gat_attention.py
# ------------------------------------------------------------
# 静的空間GNN の conv 層を GAT(GATv2) に差し替えた GNN-min-GAT を、GCN版と
# 完全同一プロトコル（leak-free Expanding Window, 同一 edge_index）で評価し、
#   (1) 学習した辺の注意 α が「等重み 1/deg」からどれだけ逸脱するか（§6d 時間
#       attention 分析の空間版）を gat_attention_summary.csv に出す。
#   (2) GNN-min-GAT の R²（全体/Central5/Others ほか）を results_long{,_full}.csv に
#       追記（GCN版・XGB-full の行はそのまま残す）。
#
# 注意 α は 1層目(多ヘッド GATv2)から return_attention_weights=True で取り出し、
# heads 平均 → 各「対象ノード(町) i」へ入る辺集合（=近隣 + self-loop）で集計する。
# PyG の GAT は対象ノード i の近傍について softmax するので、i へ入る辺の重みは和=1。
# self-loop は PyG 既定(add_self_loops=True)で付与され、GCNConv と同じく自己ループを
# 含めて近隣分布を正規化する（GCN版との整合）。等重み基準は 1/deg(i)（degは self-loop
# 込みの入次数）＝「GATがmean-pooling(手作りS-lag的)と等価になる重み」。
#
# 既定は直近8四半期で素早く確認。GNN_FULL_EVAL=1 で全64四半期（推奨・正本）。
# ------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("src")
import config as C
import lib_compare as LC
from lib_gnn import assert_conv_only_diff

try:
    import japanize_matplotlib  # noqa
except ImportError:
    pass

DEVICE = os.environ.get("GNN_DEVICE", C.GNN_DEVICE)
FULL_EVAL = bool(os.environ.get("GNN_FULL_EVAL"))
TEST_QUARTERS = C.EXPANDING_TEST_QUARTERS if FULL_EVAL else C.GNN_TEST_QUARTERS
CONV = os.environ.get("SPATIAL_CONV", "gatv2")     # 解釈分析は GAT系のみ意味を持つ
SUF = "_full" if FULL_EVAL else ""
SUB = {"Central5": C.CENTRAL_5,
       "Others": [w for w in C.WARDS_23 if w not in C.CENTRAL_5]}
RESULTS_NAME = "results_long_full.csv" if FULL_EVAL else "results_long.csv"

print(f"=== 14. 学習空間重み(GAT)の解釈分析 (conv={CONV}, device={DEVICE}, "
      f"{'全64四半期' if FULL_EVAL else '直近8四半期'}={len(TEST_QUARTERS)}fold) ===")

# フェア比較の生命線：GCN版と conv 層以外完全一致であることを実行前に明示。
_info = assert_conv_only_diff(n_node_feat=4, n_prop_feat=len(C.GNN_PROP_FEATURES),
                              hidden=C.GNN_HIDDEN, heads=C.GAT_HEADS, conv=CONV)
print(f"[conv層のみ差分の確認] GCN版と{CONV}版で head/最終空間表現次元が一致: "
      f"hidden={_info['hidden_out']}, head_in={_info['head_in']}"
      f"(=hidden+物件特徴{_info['n_prop_feat']})")
print(f"  共有ハイパラ: 層数=2, hidden={C.GNN_HIDDEN}, optimizer=Adam, lr={C.GNN_LR}, "
      f"epochs={C.GNN_EPOCHS}, seed={C.RANDOM_STATE}, GAT_HEADS={C.GAT_HEADS}"
      f"（変えるのは conv 層のみ）")

df = pd.read_csv(C.DATA_DIR / "tokyo23_model_table_q.csv")
df = df.dropna(subset=[C.TARGET, "Qidx"]).copy()
df["Qidx"] = df["Qidx"].astype(int)

print("グラフ（町丁目queen隣接）を構築中...")
node_index, edge_index, N = LC.build_town_graph(df)
print(f"  ノード(町)数: {N}  辺数(有向): {edge_index.shape[1]}")

# --- 重み収集フック：各テスト四半期の評価後に、町ごとの注意分布を集計 -------
# att_ei[2,E] / alpha[E,heads]。対象ノード i = att_ei[1]（i へ入る辺の重み和=1）。
rows = []   # (q, node, group, deg, l1, max_w, self_w, top_nb_w)


def sink(q, test_df, att_ei, alpha):
    a = alpha.mean(axis=1)                       # heads 平均 → [E]
    src, dst = att_ei[0], att_ei[1]
    # この四半期のテストに出現する町だけ集計（その町の予測に効いた重みのため）
    seen = {}
    for n, wd in zip(test_df["node"].to_numpy(), test_df["Municipality"].to_numpy()):
        seen.setdefault(int(n), "Central5" if wd in C.CENTRAL_5 else "Others")
    # 対象ノード i ごとに、i へ入る辺（= i の近傍 + self-loop）をまとめる
    for i, grp in seen.items():
        m = dst == i
        if not m.any():
            continue
        w = a[m]                                 # i へ入る辺の重み（和≈1, self-loop込み）
        s = src[m]
        deg = len(w)
        uni = 1.0 / deg
        self_w = float(w[s == i].sum()) if (s == i).any() else 0.0
        nb = w[s != i]
        top_nb = float(nb.max()) if nb.size else 0.0
        l1 = float(np.abs(w - uni).sum())
        rows.append((int(q), i, grp, deg, l1, float(w.max()), self_w, top_nb))


print(f"\n{CONV} 静的GNN(GNN-min-GAT)を同一フォールドで学習し注意重みを収集中...")
summ = LC.run_static_gnn(df, node_index, edge_index, N, TEST_QUARTERS, SUB,
                         device=DEVICE, conv=CONV, tag="GNN-min-GAT",
                         weight_sink=sink)
a_all = summ[summ["scope"] == "ALL"].iloc[0]
print(f"  GNN-min-GAT 全体R² = {a_all['r2_mean']:.3f} ± {a_all['r2_std']:.3f} "
      f"({int(a_all['n_folds'])}fold)")

W = pd.DataFrame(rows, columns=["Qidx", "node", "group", "deg", "l1",
                                "max_w", "self_w", "top_nb_w"])
print(f"  収集: {len(W):,} (町×四半期) 本の注意分布")

# --- 集計: 等重み 1/deg からの逸脱（全体 / Central5 / Others） -----------
def agg(d):
    return dict(
        n=len(d), deg_mean=d["deg"].mean(), uniform_mean=(1.0 / d["deg"]).mean(),
        l1_mean=d["l1"].mean(), l1_median=d["l1"].median(),
        max_w_mean=d["max_w"].mean(), self_w_mean=d["self_w"].mean(),
        top_nb_w_mean=d["top_nb_w"].mean())


summary_rows = [{"scope": "overall", **agg(W)}]
for g in ["Central5", "Others"]:
    summary_rows.append({"scope": g, **agg(W[W.group == g])})
summary = pd.DataFrame(summary_rows)
out_sum = C.OUT_DIR / f"gat_attention_summary{SUF}.csv"
summary.to_csv(out_sum, index=False, encoding="utf-8-sig")
print(f"\n  表: {out_sum.name}")
print(summary.round(4).to_string(index=False))

# --- results_long への追記（GNN-min-GAT 行を upsert。他モデルの行は残す） ----
COLS = ["model", "scope", "name", "r2_mean", "r2_std", "r2_min", "r2_max", "n_folds"]
new = summ.copy()
if "model" not in new.columns:
    new.insert(0, "model", "GNN-min-GAT")
new = new[[c for c in COLS if c in new.columns]]
res_path = C.OUT_DIR / RESULTS_NAME
if res_path.exists():
    old = pd.read_csv(res_path)
    old = old[old["model"] != "GNN-min-GAT"]
    merged = pd.concat([old, new], ignore_index=True)[COLS]
else:
    merged = new[COLS]
merged.to_csv(res_path, index=False, encoding="utf-8-sig")
print(f"  表: {RESULTS_NAME} に GNN-min-GAT 行を追記（GCN版・XGB-full 等は保持）")

# --- 図: 等重み(1/deg)からの逸脱を一望（§6d 空間版） --------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
# (a) 町ごと最大注意重み vs 平均等重み 1/deg
ax = axes[0]
ax.hist(W["max_w"], bins=40, color="steelblue", alpha=0.85,
        label="町ごと最大注意重み")
um = (1.0 / W["deg"]).mean()
ax.axvline(um, color="crimson", lw=1.6, ls="--",
           label=f"平均等重み 1/deg ≈ {um:.3f}")
ax.axvline(W["max_w"].mean(), color="black", lw=1.2,
           label=f"最大重み平均 = {W['max_w'].mean():.3f}")
ax.set_xlabel("最大注意重み（top-1 近隣/自己に乗る質量）")
ax.set_ylabel("町×四半期 件数")
ax.set_title("学習注意は等重みからどれだけ集中したか")
ax.legend(fontsize=8)
# (b) 町ごと L1 逸脱（0=完全一様, 2≈完全集中）
ax = axes[1]
for g, col in [("Central5", "darkorange"), ("Others", "seagreen")]:
    d = W[W.group == g]["l1"]
    ax.hist(d, bins=40, color=col, alpha=0.55, label=f"{g} (median={d.median():.3f})")
ax.axvline(0, color="black", lw=1.0)
ax.set_xlabel("等重み 1/deg からの L1 逸脱  Σ|α−1/deg|（0=一様, ~2=集中）")
ax.set_ylabel("町×四半期 件数")
ax.set_title("学習近隣重みの一様からの逸脱（Central5 vs Others）")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(C.OUT_DIR / f"fig_gat_attention{SUF}.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  図: fig_gat_attention{SUF}.png")

# --- 結論サマリ --------------------------------------------------------
ov = summary[summary.scope == "overall"].iloc[0]
verdict = ("ほぼ一様（時間attentionと同様に潰れた）"
           if ov["l1_mean"] < 0.30 else "等重みから有意に逸脱（特定近隣に集中）")
print(f"\n[観測（学習空間重みの解釈）]")
print(f"  平均入次数(self-loop込) deg = {ov['deg_mean']:.2f} → 等重み 1/deg ≈ "
      f"{ov['uniform_mean']:.3f}")
print(f"  等重みからの L1 逸脱: mean={ov['l1_mean']:.3f} / median={ov['l1_median']:.3f}"
      f"（0=完全一様, ~2=完全集中）")
print(f"  最大注意重み平均 = {ov['max_w_mean']:.3f}（自己ループ重み平均 "
      f"{ov['self_w_mean']:.3f} / top-1近隣 {ov['top_nb_w_mean']:.3f}）")
print(f"  → 学習空間重みは {verdict}")
print("\n完了")
