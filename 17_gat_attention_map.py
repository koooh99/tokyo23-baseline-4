# 17_gat_attention_map.py
# ------------------------------------------------------------
# 学習済みGAT(GATv2)の「近隣集約の注意重み」を地図上に載せ、空間的な異質性が
# どこに在るかを町丁目解像度で診断する（実験③=14 の地理版）。
#
# 何を見せるか:
#   Moran's I や SEM のλは「大域スカラー1個」なので局在を均す。本図はそれと違い、
#   GATが各町で学習した近隣重み w_i の「等重み 1/deg からの逸脱(非一様性)」を
#   町ごとに地図化し、"どこで" 重みが偏るかを一目で示す。主張は
#   「重みは大半の町でほぼ一様(=GCNの等重み平均と同等)だが、都心5区でのみ非一様
#     ＝局所的な空間異質性が存在する」の検証（実験③で言及した『都心5区は約2倍の
#     L1偏差』を地図と分布で裏取りする）。
#
# 重要な解釈上の注記（図にも明記する）:
#   本図は学習済みGATの近隣集約重みの非一様性を示す（attention is not explanation）。
#   価格予測における"価値"ではなく"構造の在り処"の診断であり、2×2要因デザイン(16)で
#   都心5区でも空間波及は冗長(交互作用は負=代替的)であった点と併せて解釈する。
#
# 抽出ロジックは再実装しない:
#   14_gat_attention.py と同一の LC.run_static_gnn(..., weight_sink=sink) を用い、
#   GCN版と conv 層以外は完全一致・leak-free Expanding Window(窓12)・同一 edge_index・
#   同一前処理(IQR/標準化はフォールド訓練のみ)で学習する。学習済みモデルへの事後抽出
#   なのでリークは増えない。違いは sink が「町スカラー」でなく「辺(dst,src)ごとの重み」を
#   保持する点だけ（地理に載せ替えるため。lib_gnn/lib_compare は一切変更しない）。
#
# self-loop の扱い:
#   GATv2 は PyG 既定(add_self_loops=True)で自己ループを付与し、対象町 i の入辺
#   （近隣 + 自己ループ）で softmax するので和=1。本スクリプトは
#     (A) self込み版（生のα、和=1）
#     (B) self除外＋近隣のみ行正規化版（近隣分布として再正規化）
#   の両方を計算し、図と非一様性スカラー(l1_dev)は既定で (B) を使う
#   （「近隣をどう不均等に混ぜたか」を見たいので自己ループは除く）。
#
# 平均する四半期:
#   既定は直近8四半期 (2024Q1–2025Q4, GNN_TEST_QUARTERS, 窓12) で各フォールドの
#   学習αを辺ごとに平均（10/run_compare と同一フォールド・前処理）。
#   GNN_FULL_EVAL=1 で全64四半期（14 と同じトグル。ただし長期平均は局在を均す傾向）。
#
# 出力（outputs/）:
#   - fig_gat_attention_map_overview.png : 地図①（全域 choropleth, 非一様性 l1_dev）
#   - fig_gat_attention_map_central5.png : 地図②（都心5区ズーム, 辺をGAT重みで描画）
#   - fig_gat_attention_l1_dist.png      : 分布図（都心5区 vs その他の l1_dev + MWU p）
#   - gat_attention_town_l1.csv          : 町ごとの非一様性スカラー
#       (town, ward, l1_dev, entropy, deg, scope, self_w, l1_dev_incl)
# ------------------------------------------------------------
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

sys.path.append("src")
import config as C
import lib_compare as LC
from lib_spatial import load_town_polygons
from lib_gnn import assert_conv_only_diff

try:
    import japanize_matplotlib  # noqa
except ImportError:
    pass

# --- 3色構成（ベース/サブ/アクセント, ゼミ作法） ----------------------
BASE = "#d9d9d9"        # ベース: 欠損・地味な辺（淡いグレー）
SUB = "#2c7fb7"         # サブ : その他18区
ACCENT = "#e6550d"      # アクセント: 都心5区/突出を強調

DEVICE = os.environ.get("GNN_DEVICE", C.GNN_DEVICE)
FULL_EVAL = bool(os.environ.get("GNN_FULL_EVAL"))
TEST_QUARTERS = C.EXPANDING_TEST_QUARTERS if FULL_EVAL else C.GNN_TEST_QUARTERS
CONV = os.environ.get("SPATIAL_CONV", "gatv2")      # 解釈分析は GAT系のみ意味を持つ
SUF = "_full" if FULL_EVAL else ""

print(f"=== 17. 学習GAT空間重みの地図化 (conv={CONV}, device={DEVICE}, "
      f"{'全64四半期' if FULL_EVAL else '直近8四半期'}={len(TEST_QUARTERS)}fold) ===")

# フェア比較の生命線（14と同じ）：GCN版と conv 層以外完全一致を実行前に明示。
_info = assert_conv_only_diff(n_node_feat=4, n_prop_feat=len(C.GNN_PROP_FEATURES),
                              hidden=C.GNN_HIDDEN, heads=C.GAT_HEADS, conv=CONV)
print(f"[conv層のみ差分の確認] hidden={_info['hidden_out']}, "
      f"head_in={_info['head_in']}（=hidden+物件特徴{_info['n_prop_feat']}）")

df = pd.read_csv(C.DATA_DIR / "tokyo23_model_table_q.csv")
df = df.dropna(subset=[C.TARGET, "Qidx"]).copy()
df["Qidx"] = df["Qidx"].astype(int)

print("グラフ（町丁目queen隣接）を構築中...")
node_index, edge_index, N = LC.build_town_graph(df)
print(f"  ノード(町)数: {N}  辺数(有向): {edge_index.shape[1]}")

# --- 重み収集フック：辺(dst,src)ごとに、各フォールドの学習αを足し込む ---------
# att_ei[2,E] は self-loop 付与後の辺、alpha[E,heads]。対象ノード i = att_ei[1]
# （i へ入る辺の重み和=1, self-loop込み）。14 と違い辺の src を保持して地理に載せる。
# alpha は評価forwardで全ノードについて出るので、テスト物件の有無に関わらず全辺を集める。
edge_sum = defaultdict(float)   # (dst, src) -> Σ_q  (heads平均後の重み)
edge_cnt = defaultdict(int)     # (dst, src) -> 平均した四半期数


def sink(q, test_df, att_ei, alpha):
    a = alpha.mean(axis=1)                          # heads 平均 → [E]
    src, dst = att_ei[0], att_ei[1]
    for e in range(a.shape[0]):
        k = (int(dst[e]), int(src[e]))
        edge_sum[k] += float(a[e])
        edge_cnt[k] += 1


print(f"\n{CONV} 静的GNN(GNN-min-GAT)を14と同一フォールドで学習し辺重みを収集中...")
summ = LC.run_static_gnn(
    df, node_index, edge_index, N, TEST_QUARTERS,
    {"Central5": C.CENTRAL_5,
     "Others": [w for w in C.WARDS_23 if w not in C.CENTRAL_5]},
    device=DEVICE, conv=CONV, tag="GNN-min-GAT", weight_sink=sink)
a_all = summ[summ["scope"] == "ALL"].iloc[0]
print(f"  GNN-min-GAT 全体R² = {a_all['r2_mean']:.3f} ± {a_all['r2_std']:.3f} "
      f"({int(a_all['n_folds'])}fold)  / 収集辺数(有向, self込) = {len(edge_sum):,}")

# --- 辺重みを「対象町 i の入辺分布」に束ね、非一様性スカラーを計算 -------------
# incoming[i] = [(src, mean_w), ...]（mean_w は四半期平均。各四半期で和=1 なので
# 平均も和=1）。self除外＋近隣のみ行正規化を既定とし、l1_dev/entropy を出す。
incoming = defaultdict(list)
for (dst, src), s in edge_sum.items():
    incoming[dst].append((src, s / edge_cnt[(dst, src)]))

# ノード→(区, 町)・重心。13/run_static_gnn と同一の groupby で対応づけ。
meta = df.groupby("node")[["Municipality", "DistrictName",
                           "centroid_x", "centroid_y"]].first()

town_rows = []
edge_draw = []   # 地図②用: (i, j, w_norm) ただし都心5区を含む辺のみ後で抽出
for i, lst in incoming.items():
    if i not in meta.index:
        continue
    srcs = np.array([s for s, _ in lst])
    w = np.array([v for _, v in lst], dtype=float)         # self込み（和≈1）
    self_mask = srcs == i
    self_w = float(w[self_mask].sum()) if self_mask.any() else 0.0
    # (A) self込み版の L1 逸脱（参考）
    deg_incl = len(w)
    l1_incl = float(np.abs(w - 1.0 / deg_incl).sum())
    # (B) self除外＋近隣のみ行正規化（既定）
    nb_srcs = srcs[~self_mask]
    nb_w = w[~self_mask]
    deg = len(nb_w)
    if deg == 0 or nb_w.sum() <= 0:
        continue
    nb_w = nb_w / nb_w.sum()                                # 近隣分布として再正規化
    uni = 1.0 / deg
    l1_dev = float(np.abs(nb_w - uni).sum())               # 0=完全一様, ~2=完全集中
    ent = float(-(nb_w * np.log(nb_w + 1e-12)).sum())
    ent_norm = ent / np.log(deg) if deg > 1 else 0.0        # 0=集中, 1=一様
    muni = meta.at[i, "Municipality"]
    scope = "Central5" if muni in C.CENTRAL_5 else "Others"
    town_rows.append(dict(node=i, town=meta.at[i, "DistrictName"], ward=muni,
                          l1_dev=l1_dev, entropy=ent_norm, deg=deg, scope=scope,
                          self_w=self_w, l1_dev_incl=l1_incl))
    for sj, wj in zip(nb_srcs, nb_w):
        edge_draw.append((i, int(sj), float(wj)))

T = pd.DataFrame(town_rows)
out_csv = C.OUT_DIR / f"gat_attention_town_l1{SUF}.csv"
T.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"  表: {out_csv.name}  ({len(T)} 町の非一様性スカラー)")

# --- 定量: 都心5区 vs その他の L1 逸脱（中央値 + Mann–Whitney U） --------------
c5 = T[T.scope == "Central5"]["l1_dev"].to_numpy()
ot = T[T.scope == "Others"]["l1_dev"].to_numpy()
u, p = mannwhitneyu(c5, ot, alternative="greater")    # 片側: 都心5区 > その他
ratio = np.median(c5) / np.median(ot) if np.median(ot) > 0 else np.nan
print(f"\n[非一様性 l1_dev（self除外・行正規化, 0=一様…2=集中）]")
print(f"  都心5区 : median={np.median(c5):.3f}  mean={c5.mean():.3f}  n={len(c5)}")
print(f"  その他  : median={np.median(ot):.3f}  mean={ot.mean():.3f}  n={len(ot)}")
print(f"  都心5区/その他 の中央値比 = {ratio:.2f}倍   "
      f"Mann–Whitney U p(片側 C5>Others) = {p:.2e}")

# --- 地図のための町ポリゴン（lib_spatial で町名整合済み, 6677へ投影） -----------
towns_poly = load_town_polygons(C.SHAPEFILE_PATH, C.WARDS_23).to_crs(C.PROJECTED_CRS)
gT = towns_poly.merge(T, left_on=["Municipality", "DistrictName"],
                      right_on=["ward", "town"], how="left")
n_join = gT["l1_dev"].notna().sum()
print(f"\n  ポリゴン結合: {n_join}/{len(towns_poly)} 町に l1_dev を付与 "
      f"(残りは取引なし/町名NaN → 欠損グレー表示)")
# 都心5区の区界（太線オーバーレイ用）
c5_boundary = (towns_poly[towns_poly["Municipality"].isin(C.CENTRAL_5)]
               .dissolve(by="Municipality").boundary)

NOTE = ("注: 学習済みGATの近隣集約重みの非一様性（attention is not explanation）。"
        "価格の“価値”でなく“構造の在り処”の診断。\n"
        "2×2要因(16)では都心5区でも空間波及は冗長（交互作用負）であった点と併せて解釈。")

# ============================ 地図①: 全域 choropleth =========================
fig, ax = plt.subplots(figsize=(9.5, 9.5))
vmax = float(np.nanpercentile(gT["l1_dev"], 97))   # 外れ値で潰れないよう97%点でクリップ
gT.plot(column="l1_dev", cmap="OrRd", vmin=0.0, vmax=vmax, ax=ax,
        linewidth=0.15, edgecolor="white",
        legend=True, legend_kwds={"shrink": 0.55,
                                  "label": "近隣重みの非一様性 L1 逸脱 Σ|w−1/deg|"},
        missing_kwds={"color": BASE, "edgecolor": "white", "linewidth": 0.1})
# 都心5区の区界（OrRdと被らない濃紺で。凡例で何の線かだけ示す）
C5LINE = "#08306b"
c5_boundary.plot(ax=ax, color=C5LINE, linewidth=2.2, zorder=5)
ax.legend(handles=[Line2D([0], [0], color=C5LINE, lw=2.2, label="都心5区 区界"),
                   Patch(facecolor=BASE, edgecolor="white",
                         label="欠損（取引なし/結合不可）")],
          loc="upper left", framealpha=0.9, fontsize=10)
ax.set_title("地図①　GAT近隣重みの非一様性（全23区・町丁目）\n"
             "濃=非一様（特定近隣に集中）／淡=一様（≒GCNの等重み平均）",
             fontsize=13)
ax.set_xlabel("centroid_x (m, EPSG:6677)")
ax.set_ylabel("centroid_y (m, EPSG:6677)")
ax.set_aspect("equal")
ax.text(0.01, -0.085, NOTE, transform=ax.transAxes, fontsize=8, color="#555",
        va="top")
plt.tight_layout()
fig.savefig(C.OUT_DIR / f"fig_gat_attention_map_overview{SUF}.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  図: fig_gat_attention_map_overview{SUF}.png")

# ============================ 地図②: 都心5区ズーム・辺 =======================
# 重心座標(centroid_x/y)を辺の端点に。重みの高い辺だけ強調し、低い辺は地味に。
cent_xy = {i: (meta.at[i, "centroid_x"], meta.at[i, "centroid_y"])
           for i in meta.index}
c5_nodes = set(T[T.scope == "Central5"]["node"])
# 端点の少なくとも一方が都心5区にある近隣辺のみ（区界・水際・高級隣接が見える）
segs, ws = [], []
for i, j, w in edge_draw:
    if i not in c5_nodes and j not in c5_nodes:
        continue
    if i in cent_xy and j in cent_xy:
        segs.append([cent_xy[i], cent_xy[j]])
        ws.append(w)
ws = np.array(ws)
# 重み→線幅/濃さ（高い辺を強調、低い辺は淡く）。等重み 1/deg 近辺は地味になる。
wn = (ws - ws.min()) / (ws.max() - ws.min() + 1e-12)
lw = 0.4 + 4.5 * wn ** 1.5
lc = LineCollection(segs, array=ws, cmap="OrRd",
                    linewidths=lw, alpha=0.9, zorder=3)

fig, ax = plt.subplots(figsize=(9.5, 9))
# 背景: 都心5区の町ポリゴンを淡く + 区界
gT[gT["Municipality"].isin(C.CENTRAL_5)].plot(
    ax=ax, color="#f7f7f7", edgecolor="#bdbdbd", linewidth=0.3, zorder=1)
c5_boundary.plot(ax=ax, color="#636363", linewidth=1.6, zorder=2)
# 重心点（都心5区）
c5_pts = np.array([cent_xy[i] for i in c5_nodes if i in cent_xy])
ax.scatter(c5_pts[:, 0], c5_pts[:, 1], s=10, c="#252525", zorder=4)
ax.add_collection(lc)
cb = fig.colorbar(lc, ax=ax, shrink=0.6)
cb.set_label("近隣辺のGAT重み（行正規化, self除外）")
# 突出辺（上位）を太字注記
ax.set_title("地図②　都心5区の近隣辺をGAT重みで描画\n"
             "太く濃い辺＝不均等に重い近隣関係（区界・水際・高級隣接）", fontsize=13)
ax.set_xlabel("centroid_x (m, EPSG:6677)")
ax.set_ylabel("centroid_y (m, EPSG:6677)")
ax.set_aspect("equal")
ax.legend(handles=[Line2D([0], [0], color=ACCENT, lw=4, label="重い近隣辺（強調）"),
                   Line2D([0], [0], color="#fdd0a2", lw=1, label="一様に近い辺（地味）")],
          loc="upper left", framealpha=0.9, fontsize=9)
ax.text(0.01, -0.085, NOTE, transform=ax.transAxes, fontsize=8, color="#555",
        va="top")
plt.tight_layout()
fig.savefig(C.OUT_DIR / f"fig_gat_attention_map_central5{SUF}.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  図: fig_gat_attention_map_central5{SUF}.png")

# ============================ 分布図: C5 vs Others ===========================
fig, ax = plt.subplots(figsize=(8.5, 5.2))
bins = np.linspace(0, max(T["l1_dev"].max(), 0.1), 40)
ax.hist(ot, bins=bins, color=SUB, alpha=0.55, density=True,
        label=f"その他18区 (median={np.median(ot):.3f}, n={len(ot)})")
ax.hist(c5, bins=bins, color=ACCENT, alpha=0.6, density=True,
        label=f"都心5区 (median={np.median(c5):.3f}, n={len(c5)})")
ax.axvline(np.median(ot), color=SUB, lw=1.6, ls="--")
ax.axvline(np.median(c5), color=ACCENT, lw=1.8, ls="--")
ax.set_xlabel("近隣重みの非一様性 L1 逸脱 Σ|w−1/deg|（0=一様, ~2=集中, self除外）")
ax.set_ylabel("密度")
ax.set_title(f"分布図　L1逸脱の都心5区 vs その他\n"
             f"中央値比 ≈ {ratio:.2f}倍 / Mann–Whitney U p={p:.1e}（片側 C5>Others）",
             fontsize=12)
ax.legend(fontsize=9)
ax.text(0.01, -0.16, NOTE, transform=ax.transAxes, fontsize=8, color="#555",
        va="top")
plt.tight_layout()
fig.savefig(C.OUT_DIR / f"fig_gat_attention_l1_dist{SUF}.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  図: fig_gat_attention_l1_dist{SUF}.png")

# --- 突出した町・辺の報告（区界/水際/高級隣接との対応の手掛かり） --------------
print("\n[非一様性が突出した都心5区の町 上位10]")
top_towns = T[T.scope == "Central5"].nlargest(10, "l1_dev")
for r in top_towns.itertuples(index=False):
    print(f"  {r.ward}{r.town}: l1_dev={r.l1_dev:.3f} (deg={r.deg}, "
          f"self_w={r.self_w:.3f})")

print("\n[重みが突出した近隣辺 上位10（都心5区を含む）]")
idx_to_key = {v: k for k, v in node_index.items()}
top_edges = sorted([(i, j, w) for i, j, w in edge_draw
                    if i in c5_nodes or j in c5_nodes],
                   key=lambda t: -t[2])[:10]
for i, j, w in top_edges:
    ki, kj = idx_to_key.get(i), idx_to_key.get(j)
    print(f"  {ki[0]}{ki[1]} ← {kj[0]}{kj[1]}: w={w:.3f}")

print("\n完了")
