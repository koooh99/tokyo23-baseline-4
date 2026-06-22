# 11_gnn_report.py
# ------------------------------------------------------------
# GNN系の進捗・成果を 06 と同じ作法（分布で正直に）で図表化する。
# 全64四半期(2010Q1〜2025Q4)の同一フォールドでのモデル比較:
#   XGB-propT → GNN静的v0 → XGB-full(Qboth) → GNN-temporal → GNN-temporal+feat
# 入力は 09/10 が保存した outputs/*_full.csv。推移図はGNN per-fold(10のログ)＋
# XGB-full per-fold(同条件で再計算)から作る。
# ------------------------------------------------------------
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("src")
import config as C

try:
    import japanize_matplotlib  # noqa
except ImportError:
    pass

print("=== 11. GNNレポート（進捗・成果のまとめ） ===")

# --- 結果CSVを集約 -------------------------------------------------
files = ["gnn_vs_xgb_q_full.csv", "gnn_temporal_q_full.csv",
         "gnn_temporal_rich_q_full.csv"]
frames = [pd.read_csv(C.OUT_DIR / f) for f in files if (C.OUT_DIR / f).exists()]
if not frames:
    raise SystemExit("結果CSVが見つかりません。先に 09/10 を GNN_FULL_EVAL=1 で実行してください。")
allres = pd.concat(frames, ignore_index=True).drop_duplicates(["model", "scope", "name"])

# モデルを能力順に並べる（物語の順）。存在するものだけ使う。
ORDER = ["XGB-propT(自町T-lagのみ)", "GNN(min v0)", "XGB-full(Qboth)",
         "GNN-temporal(GRU)", "GNN-temporal+feat"]
LABEL = {"XGB-propT(自町T-lagのみ)": "XGB\n物件+自町T-lag",
         "GNN(min v0)": "GNN 静的v0",
         "XGB-full(Qboth)": "XGB-full\n(Qboth)",
         "GNN-temporal(GRU)": "GNN+時系列\n(GRU)",
         "GNN-temporal+feat": "GNN+時系列\n+立地特徴"}
is_gnn = {m: m.startswith("GNN") for m in ORDER}

allrow = allres[allres["scope"] == "ALL"].set_index("model")
models = [m for m in ORDER if m in allrow.index]

# --- 図1: モデル進捗（全体R² 平均±標準偏差） ----------------------
fig, ax = plt.subplots(figsize=(9, 5))
xs = range(len(models))
means = [allrow.loc[m, "r2_mean"] for m in models]
stds = [allrow.loc[m, "r2_std"] for m in models]
colors = []
best = max(models, key=lambda m: allrow.loc[m, "r2_mean"])
for m in models:
    colors.append("crimson" if m == best else ("steelblue" if is_gnn[m] else "slategray"))
bars = ax.bar(xs, means, yerr=stds, capsize=5, color=colors, alpha=0.88)
for x, mu in zip(xs, means):
    ax.text(x, mu + 0.012, f"{mu:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(list(xs))
ax.set_xticklabels([LABEL.get(m, m) for m in models], fontsize=9)
ax.set_ylabel("R² (mean ± std, 64四半期フォールド)")
ax.set_title("時空間GNNの進捗：全体R²（赤=最良 / 青=GNN / 灰=XGB）")
ax.axhline(allrow.loc["XGB-full(Qboth)", "r2_mean"] if "XGB-full(Qboth)" in allrow.index
           else 0, color="black", lw=0.7, ls="--", alpha=0.6)
ax.grid(True, axis="y", ls="--", alpha=0.5)
plt.tight_layout()
fig.savefig(C.OUT_DIR / "fig_gnn_progress.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  図: fig_gnn_progress.png")

# --- 図2: サブグループ（都心5 vs その他） -------------------------
grp = allres[allres["scope"] == "GROUP"]
sub_models = [m for m in ["XGB-full(Qboth)", "GNN(min v0)", "GNN-temporal+feat"]
              if m in grp["model"].values]
if sub_models:
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.38
    x = np.arange(len(sub_models))
    for i, name in enumerate(["Central5", "Others"]):
        vals = [grp[(grp.model == m) & (grp.name == name)]["r2_mean"].iloc[0]
                for m in sub_models]
        errs = [grp[(grp.model == m) & (grp.name == name)]["r2_std"].iloc[0]
                for m in sub_models]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=4,
               label=name, color=["darkorange", "seagreen"][i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(m, m) for m in sub_models], fontsize=9)
    ax.set_ylabel("R² (mean ± std)")
    ax.set_title("サブグループ別R²：都心5区でGNNの優位が明確")
    ax.legend(); ax.grid(True, axis="y", ls="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(C.OUT_DIR / "fig_gnn_subgroups.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  図: fig_gnn_subgroups.png")

# --- 集約テーブル（論文用） ---------------------------------------
tbl = (allrow.reindex(models)[["r2_mean", "r2_std", "r2_min", "r2_max", "n_folds"]]
       .round(3).reset_index())
tbl.to_csv(C.OUT_DIR / "gnn_model_comparison.csv", index=False, encoding="utf-8-sig")
print("  表: gnn_model_comparison.csv")
print("\n[全体R² 比較（64四半期）]")
print(tbl.to_string(index=False))

# --- 図3: per-fold R² の時間推移（GNN vs XGB を「同一学習窓どうし」で比較） --
# 入力は 10_gnn_temporal.py が窓ごとに保存する outputs/perfold_win{tag}.csv
# （列: test_year, r2, model, window）。GNN(window=W) と XGB(window=W) を同一
# フォールドで並べることで、旧 fig（GNN=3年窓 vs XGB=全期間窓）にあった「モデル差と
# 学習窓差の交絡」を解消する。上段=重ね描き / 下段=差分曲線(GNN−XGB)。
GNN_PF_LABEL = "GNN+時系列+立地特徴"   # 図の凡例表示名（CSVのmodel名とは別でよい）
XGB_PF_NAME = "XGB-full(Qboth)"        # XGB行の識別名。GNN行は「XGB以外」で拾う
pf_files = sorted(C.OUT_DIR.glob("perfold_win*.csv"))
if not pf_files:
    print("\n[推移図] perfold_win*.csv が無いため skip。"
          "先に 10 を GNN_FULL_EVAL=1・各 GNN_TRAIN_WINDOW で実行してください。")
else:
    def _win_sort_key(p):                       # 12 < 24 < full の順に並べる
        tag = p.stem.replace("perfold_win", "")
        return (1, 0) if tag == "full" else (0, int(tag))
    print("\n[推移図] 学習窓ごとに GNN vs XGB を同一フォールドで比較:")
    for p in sorted(pf_files, key=_win_sort_key):
        win_tag = p.stem.replace("perfold_win", "")
        d = pd.read_csv(p)
        gnn_fold = (d[d["model"] != XGB_PF_NAME].set_index("test_year")["r2"].to_dict())
        xgb_fold = (d[d["model"] == XGB_PF_NAME].set_index("test_year")["r2"].to_dict())
        qs = sorted(set(gnn_fold) & set(xgb_fold))
        if not qs:
            print(f"  - 窓={win_tag}: 共通フォールド無し、skip")
            continue
        xlab = [f"{q // 4}Q{q % 4 + 1}" for q in qs]
        gv = np.array([gnn_fold[q] for q in qs])
        xv = np.array([xgb_fold[q] for q in qs])
        diff = gv - xv
        wins = int((diff > 0).sum())
        win_label = "全期間" if win_tag == "full" else f"直近{win_tag}四半期"

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 6.2), sharex=True,
            gridspec_kw={"height_ratios": [2.1, 1]})
        ax1.plot(xlab, gv, "-o", ms=3, color="crimson", label=GNN_PF_LABEL)
        ax1.plot(xlab, xv, "-s", ms=3, color="slategray", label=XGB_PF_NAME)
        ax1.set_ylabel("R² (各四半期テスト)")
        # タイトルは断定せず、実際の勝敗カウントだけを中立に述べる。
        ax1.set_title(
            f"四半期ごとのR²：学習窓を揃えた比較（窓={win_label}）"
            f"／GNN優位 {wins}/{len(qs)}四半期・平均差(GNN−XGB) {diff.mean():+.3f}")
        ax1.legend(); ax1.grid(True, ls="--", alpha=0.5)
        ax2.axhline(0, color="black", lw=0.8)
        ax2.bar(range(len(qs)), diff, color=np.where(diff > 0, "crimson", "slategray"),
                alpha=0.8)
        ax2.set_ylabel("差分 GNN−XGB")
        step = max(1, len(qs) // 16)
        ax2.set_xticks(range(0, len(qs), step))
        ax2.set_xticklabels([xlab[i] for i in range(0, len(qs), step)],
                            rotation=45, ha="right")
        ax2.grid(True, axis="y", ls="--", alpha=0.5)
        plt.tight_layout()
        out_png = C.OUT_DIR / f"fig_gnn_timeseries_win{win_tag}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  - 窓={win_tag}: {out_png.name}  "
              f"GNN平均R²={gv.mean():.3f} / XGB平均R²={xv.mean():.3f} / "
              f"GNN優位 {wins}/{len(qs)} 四半期 / 平均差 {diff.mean():+.3f}")

print("\n完了")
