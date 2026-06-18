# 11_gnn_report.py
# ------------------------------------------------------------
# GNN系の進捗・成果を 06 と同じ作法（分布で正直に）で図表化する。
# 全64四半期(2010Q1〜2025Q4)の同一フォールドでのモデル比較:
#   XGB-propT → GNN静的v0 → XGB-full(Qboth) → GNN-temporal → GNN-temporal+feat
# 入力は 09/10 が保存した outputs/*_full.csv。推移図はGNN per-fold(10のログ)＋
# XGB-full per-fold(同条件で再計算)から作る。
# ------------------------------------------------------------
import re
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

# --- 図3: per-fold R² の時間推移（GNN-temporal+feat vs XGB-full） --
def parse_log_r2(path):
    if not (C.OUT_DIR / path).exists():
        return {}
    txt = (C.OUT_DIR / path).read_text(encoding="utf-8", errors="ignore")
    out = {}
    for m in re.finditer(r"Qidx=(\d+).*?R²=(-?\d+\.\d+)", txt):
        out[int(m.group(1))] = float(m.group(2))
    return out


gnn_fold = parse_log_r2("10_rich_full_run.log")
if gnn_fold:
    try:
        import xgboost as xgb
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from lib_eval import expanding_window_pooled
        from lib_features import StationTargetEncoder
        df = pd.read_csv(C.DATA_DIR / "tokyo23_model_table_q.csv")
        df = df.dropna(subset=[C.TARGET, "Qidx"]).copy(); df["Qidx"] = df["Qidx"].astype(int)
        oh = ["Municipality"] + [c for c in C.CAT_FEATURES if c in df.columns]
        num = C.BASE_FEATURES + C.DEFAULT_LAG_COLS + ["Station_TE"]
        pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), oh),
                                 ("num", "passthrough", num)])
        mk = lambda: Pipeline([("pre", pre), ("reg", xgb.XGBRegressor(**C.XGB_PARAMS))])
        print("\n推移図用に XGB-full per-fold を同条件で再計算中...")
        rec, _ = expanding_window_pooled(
            df, mk, oh + num, C.TARGET, C.EXPANDING_TEST_QUARTERS,
            iqr_trim=C.IQR_TRIM, iqr_k=C.IQR_K, iqr_by_ward=C.IQR_BY_WARD,
            fold_transform=StationTargetEncoder(C.TARGET, m=C.TE_SMOOTHING),
            time_col="Qidx", progress=False)
        xgb_fold = (rec[(rec.scope == "ALL")].set_index("test_year")["r2"].to_dict())

        qs = sorted(set(gnn_fold) & set(xgb_fold))
        fig, ax = plt.subplots(figsize=(11, 4.5))
        xlab = [f"{q // 4}Q{q % 4 + 1}" for q in qs]
        ax.plot(xlab, [gnn_fold[q] for q in qs], "-o", ms=3, color="crimson",
                label="GNN+時系列+立地特徴")
        ax.plot(xlab, [xgb_fold[q] for q in qs], "-s", ms=3, color="slategray",
                label="XGB-full(Qboth)")
        ax.set_ylabel("R² (各四半期テスト)")
        ax.set_title("四半期ごとのR²推移：GNNは難局面(特に近年)で粘る")
        step = max(1, len(qs) // 16)
        ax.set_xticks(range(0, len(qs), step))
        ax.set_xticklabels([xlab[i] for i in range(0, len(qs), step)], rotation=45, ha="right")
        ax.legend(); ax.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(C.OUT_DIR / "fig_gnn_timeseries.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        win = sum(gnn_fold[q] > xgb_fold[q] for q in qs)
        print(f"  図: fig_gnn_timeseries.png  (GNN勝ち {win}/{len(qs)} 四半期)")
    except Exception as e:
        print(f"  [skip] 推移図: {e}")
else:
    print("  [skip] 推移図: 10_rich_full_run.log が無い")

print("\n完了")
