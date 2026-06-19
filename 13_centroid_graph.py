# 13_centroid_graph.py
# ------------------------------------------------------------
# GNNがノード特徴に使う「町丁目の重心座標(centroid_x/centroid_y, EPSG:6677 m系)」と
# 近隣グラフ(queen隣接 edge_index)を地理空間に可視化する。
#   図1: グラフ構造 — 重心を点、queen隣接を辺で結ぶ。都心5区 vs その他で色分け。
#   図2: 重心点を平均平米単価で着色（GNNが座標に紐付けて伝播させる「信号」）。
# 重心座標はノード特徴(config.GNN_NODE_FEATURES)の一部で、09/10/12 と同一のグラフを使う。
# ------------------------------------------------------------
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.path.append("src")
import config as C
import lib_compare as LC

try:
    import japanize_matplotlib  # noqa
except ImportError:
    pass

print("=== 13. 重心座標と近隣グラフの可視化 ===")
df = pd.read_csv(C.DATA_DIR / "tokyo23_model_table_q.csv")
df = df.dropna(subset=[C.TARGET, "Qidx"]).copy()
df["Qidx"] = df["Qidx"].astype(int)

print("グラフ（町丁目queen隣接）を構築中...")
node_index, edge_index, N = LC.build_town_graph(df)
print(f"  ノード(町)数: {N}  辺数(有向): {edge_index.shape[1]}")

# --- ノードごとの重心座標(m) と 都心5区フラグ・平均平米単価 ----------
cent = df.groupby("node")[["centroid_x", "centroid_y"]].first().reindex(range(N))
ward = df.groupby("node")["Municipality"].first().reindex(range(N))
price = df.groupby("node")[C.TARGET].mean().reindex(range(N))
X = cent["centroid_x"].to_numpy() / 1000.0   # km表示に
Y = cent["centroid_y"].to_numpy() / 1000.0
is_c5 = ward.isin(C.CENTRAL_5).to_numpy()

# --- 辺をセグメント配列に（有向重複は描画上問題ないが軽くするため片方向に） ----
ei = edge_index.numpy()
und = ei[:, ei[0] < ei[1]]      # i<j のみ＝無向辺1本ずつ
segs = np.stack([np.column_stack([X[und[0]], Y[und[0]]]),
                 np.column_stack([X[und[1]], Y[und[1]]])], axis=1)
print(f"  無向辺数: {segs.shape[0]}  / 平均次数: {2 * segs.shape[0] / N:.1f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))

# 図1: グラフ構造（重心点＋queen隣接辺、都心5区で色分け） ----------------
ax = axes[0]
ax.add_collection(LineCollection(segs, colors="lightgray", linewidths=0.4,
                                 alpha=0.6, zorder=1))
ax.scatter(X[~is_c5], Y[~is_c5], s=8, c="seagreen", label="その他18区",
           zorder=2, alpha=0.8)
ax.scatter(X[is_c5], Y[is_c5], s=10, c="darkorange", label="都心5区(Central5)",
           zorder=3, alpha=0.9)
ax.set_title(f"重心座標と近隣グラフ（queen隣接）\nノード{N}町・無向辺{segs.shape[0]}本")
ax.set_xlabel("centroid_x (km, EPSG:6677)")
ax.set_ylabel("centroid_y (km, EPSG:6677)")
ax.set_aspect("equal")
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(True, ls="--", alpha=0.3)

# 図2: 重心点を平均平米単価で着色 -------------------------------------
ax = axes[1]
ax.add_collection(LineCollection(segs, colors="lightgray", linewidths=0.3,
                                 alpha=0.4, zorder=1))
vmax = np.nanpercentile(price.to_numpy(), 97)   # 外れ値で潰れないよう97%点でクリップ
sc = ax.scatter(X, Y, s=14, c=price.to_numpy(), cmap="viridis",
                vmin=np.nanmin(price.to_numpy()), vmax=vmax, zorder=2)
cb = fig.colorbar(sc, ax=ax, shrink=0.85)
cb.set_label(f"平均平米単価 ({C.TARGET})")
ax.set_title("重心点を平均平米単価で着色\n（GNNが座標に紐付けて近隣伝播する信号）")
ax.set_xlabel("centroid_x (km, EPSG:6677)")
ax.set_ylabel("centroid_y (km, EPSG:6677)")
ax.set_aspect("equal")
ax.grid(True, ls="--", alpha=0.3)

plt.tight_layout()
out = C.OUT_DIR / "fig_centroid_graph.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  図: {out.name}")
print("完了")
