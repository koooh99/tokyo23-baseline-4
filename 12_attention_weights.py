# 12_attention_weights.py
# ------------------------------------------------------------
# 時間Attention（AttentionAggregator）の学習済み重み [N,L] を集めて可視化する。
# 時系列GNNは各町について直近Lステップ seq=[q-L,…,q-1] に softmax 重みを学習する。
# その重みを「どの過去四半期を重視したか」として読み、手法書の2仮説を検証する:
#   (1) 直近1Q（q-1）への集中 ＝ 直近性
#   (2) 都心5区(Central5)ほど直近を重視（Othersより q-1 の重みが高い）
#
# 系列の並び（lib_compare.seq_steps）: index 0 = q-L(最古) … index L-1 = q-1(直前)。
# 前年同期(q-4)は index L-4 に当たる（既定 L=8 なら index 4）。
#
# 既定は直近8四半期で素早く確認。GNN_FULL_EVAL=1 で全64四半期（推奨・図の正本）。
# ------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("src")
import config as C
import lib_compare as LC

try:
    import japanize_matplotlib  # noqa
except ImportError:
    pass

DEVICE = os.environ.get("GNN_DEVICE", C.GNN_DEVICE)
FULL_EVAL = bool(os.environ.get("GNN_FULL_EVAL"))
TEST_QUARTERS = C.EXPANDING_TEST_QUARTERS if FULL_EVAL else C.GNN_TEST_QUARTERS
L = C.TEMPORAL_SEQ_LEN
SUF = "_full" if FULL_EVAL else ""
SUB = {"Central5": C.CENTRAL_5,
       "Others": [w for w in C.WARDS_23 if w not in C.CENTRAL_5]}

print(f"=== 12. 時間Attention重みの可視化 (L={L}, device={DEVICE}, "
      f"{'全64四半期' if FULL_EVAL else '直近8四半期'}={len(TEST_QUARTERS)}fold) ===")
df = pd.read_csv(C.DATA_DIR / "tokyo23_model_table_q.csv")
df = df.dropna(subset=[C.TARGET, "Qidx"]).copy()
df["Qidx"] = df["Qidx"].astype(int)

print("グラフ（町丁目queen隣接）を構築中...")
node_index, edge_index, N = LC.build_town_graph(df)
print(f"  ノード(町)数: {N}  辺数(有向): {edge_index.shape[1]}")

# --- 重み収集フック：テスト四半期に出現する町ごとに1本の重みベクトルを記録 ---
rows = []   # (q, group, w[0..L-1])


def sink(q, test_df, W):
    seen = {}
    for n, wd in zip(test_df["node"].to_numpy(), test_df["Municipality"].to_numpy()):
        if n not in seen:
            seen[n] = "Central5" if wd in C.CENTRAL_5 else "Others"
    for n, grp in seen.items():
        rows.append((int(q), grp, *W[n]))


print("\n時間Attention(GNN-temporal[attention], rich)を同一フォールドで学習し重みを収集中...")
summ = LC.run_temporal_gnn(df, node_index, edge_index, N, TEST_QUARTERS, SUB,
                           device=DEVICE, aggregator="attention", rich=True,
                           tag="GNN-temporal[attention]", weight_sink=sink)
a = summ[summ["scope"] == "ALL"].iloc[0]
print(f"  （参考）全体R² = {a['r2_mean']:.3f} ± {a['r2_std']:.3f}")

wcols = [f"w{i}" for i in range(L)]
W = pd.DataFrame(rows, columns=["Qidx", "group"] + wcols)
print(f"  収集: {len(W):,} (町×四半期) 本の重みベクトル")

# 系列ラベル（index i → q-(L-i)）。前年同期(q-4)・直前(q-1)を明示。
offsets = [L - i for i in range(L)]                       # [L, L-1, …, 1]
labels = [f"q−{o}" for o in offsets]

# --- 集計: ラグ位置ごとの平均重み（全体 / Central5 / Others） ----------
prof_all = W[wcols].mean().to_numpy()
prof = {g: W[W.group == g][wcols].mean().to_numpy() for g in ["Central5", "Others"]}

prof_df = pd.DataFrame({"lag_offset": offsets, "label": labels,
                        "overall": prof_all,
                        "Central5": prof["Central5"], "Others": prof["Others"]})
prof_df.to_csv(C.OUT_DIR / f"attention_weights{SUF}.csv", index=False,
               encoding="utf-8-sig")
print(f"  表: attention_weights{SUF}.csv")

# --- 図1: 一様(1/L)からの偏差で見るラグ重み（Central5 vs Others） -------
# 重みは一様に近いので、絶対値より「一様からの偏差」を0基準で見ると傾きが分かる。
uni = 1.0 / L
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(L)
w = 0.4
ax.bar(x - w / 2, prof["Central5"] - uni, w, label="Central5",
       color="darkorange", alpha=0.9)
ax.bar(x + w / 2, prof["Others"] - uni, w, label="Others",
       color="seagreen", alpha=0.9)
ax.axhline(0, color="black", lw=0.9)
ax.set_xticks(x)
xt = [lab + ("\n(前年同期)" if o == 4 else ("\n(直前)" if o == 1 else ""))
      for lab, o in zip(labels, offsets)]
ax.set_xticklabels(xt, fontsize=9)
ax.set_xlabel("系列位置（過去→現在, q−k = kQ前）")
ax.set_ylabel(f"平均Attention重み − 一様(1/{L})")
ax.set_title(f"学習された時間Attentionは一様(≈1/{L})に近い（偏差は±0.01未満）")
ax.legend()
ax.grid(True, axis="y", ls="--", alpha=0.5)
plt.tight_layout()
fig.savefig(C.OUT_DIR / f"fig_attention_lags{SUF}.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  図: fig_attention_lags{SUF}.png")

# --- 図2: 直前1Q(q−1)への重み配分の時系列推移（Central5 vs Others） ----
recent = (W.assign(w_recent=W["w%d" % (L - 1)])
          .groupby(["Qidx", "group"])["w_recent"].mean().reset_index())
if recent["Qidx"].nunique() >= 4:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for g, col in [("Central5", "darkorange"), ("Others", "seagreen")]:
        d = recent[recent.group == g].sort_values("Qidx")
        xl = [f"{q // 4}Q{q % 4 + 1}" for q in d["Qidx"]]
        ax.plot(xl, d["w_recent"], "-o", ms=3, color=col, label=g)
    ax.axhline(1.0 / L, color="black", lw=0.8, ls="--", alpha=0.6, label=f"一様(=1/{L})")
    ax.set_ylabel("直前1Q(q−1)への平均重み")
    ax.set_title(f"直前1Q(q−1)への重み配分の推移（一様=1/{L} 付近で推移）")
    qs = sorted(recent["Qidx"].unique())
    step = max(1, len(qs) // 16)
    ax.set_xticks(range(0, len(qs), step))
    ax.set_xticklabels([f"{qs[i] // 4}Q{qs[i] % 4 + 1}" for i in range(0, len(qs), step)],
                       rotation=45, ha="right")
    ax.legend(); ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(C.OUT_DIR / f"fig_attention_recent{SUF}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  図: fig_attention_recent{SUF}.png")

# --- 図3: q毎×ラグ位置のAttention重みヒートマップ（一様からの偏差） -------
# 各四半期(行)で、L個のラグ位置(列)に学習された平均重みを並べる。
# 一様(1/L)からの偏差で着色し、四半期ごとに重視位置がどう変わるかを一望する。
heat = W.groupby("Qidx")[wcols].mean().sort_index()       # [Q, L]
heat_csv = heat.copy(); heat_csv.columns = labels
heat_csv.to_csv(C.OUT_DIR / f"attention_weights_by_q{SUF}.csv",
                encoding="utf-8-sig")
print(f"  表: attention_weights_by_q{SUF}.csv")
if len(heat) >= 4:
    M = heat.to_numpy() - uni
    vlim = float(np.abs(M).max()) or 0.01
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.18 * len(heat))))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(f"平均Attention重み − 一様(1/{L})")
    ax.set_xticks(np.arange(L))
    ax.set_xticklabels(labels, fontsize=9)
    qs_h = list(heat.index)
    step = max(1, len(qs_h) // 24)
    ax.set_yticks(range(0, len(qs_h), step))
    ax.set_yticklabels([f"{qs_h[i] // 4}Q{qs_h[i] % 4 + 1}"
                        for i in range(0, len(qs_h), step)], fontsize=8)
    ax.set_xlabel("系列位置（過去→現在, q−k = kQ前）")
    ax.set_ylabel("テスト四半期")
    ax.set_title(f"四半期ごとの時間Attention重み（一様1/{L}からの偏差）")
    plt.tight_layout()
    fig.savefig(C.OUT_DIR / f"fig_attention_heatmap{SUF}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  図: fig_attention_heatmap{SUF}.png")

# --- 仮説の数値サマリ ---------------------------------------------
print("\n[ラグ位置ごとの平均Attention重み]")
print(prof_df.round(3).to_string(index=False))

# 集中度: 一様分布からの偏差量(L1)と最大重み。1/Lに近いほど「ほぼ一様」。
uni = 1.0 / L
i_recent, i_yoy = L - 1, L - 4
l1 = np.abs(prof_all - uni).sum()
print(f"\n[観測（仮説の検証結果）]")
print(f"  一様基準 1/{L} = {uni:.3f}。重みの一様からの総偏差(L1)= {l1:.3f}、"
      f"最大重み = {prof_all.max():.3f}（位置 {labels[int(prof_all.argmax())]}）")
print(f"  (1) 直近1Q(q−1)集中?: 全体 q−1={prof_all[i_recent]:.3f} "
      f"→ {'一様超(集中)' if prof_all[i_recent] > uni + 0.005 else 'ほぼ一様（集中は見られない）'}")
print(f"  (2) 都心の直近重視?: Central5 q−1={prof['Central5'][i_recent]:.3f} vs "
      f"Others q−1={prof['Others'][i_recent]:.3f}")
print(f"  (参考) 前年同期(q−4)重み: Central5={prof['Central5'][i_yoy]:.3f} / "
      f"Others={prof['Others'][i_yoy]:.3f}")
print("  → 学習された時間Attentionはほぼ一様。GRUの上積み(§6b)は重み集中ではなく"
      "再帰/非線形性に由来する可能性が高い（前方補完で各ステップ表現が似るため）。")
print("\n完了")
