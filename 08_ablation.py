# 08_ablation.py
# ------------------------------------------------------------
# 特徴量アブレーション（教授報告用）。
# 問い: 立地系の特徴量（駅名・用途地域・駅まで時間）は、町丁目レベルの
#       価格ラグを入れた後でも追加的に効くのか？
# 方法: (1) 前進ネスト … 物件のみ → 1つずつ足す
#       (2) Leave-one-out … フルから1つ抜いて R² の落ち幅を見る
# 主指標 r2(テスト非トリム) と感度 r2_trim を併記。全16フォルドの平均±標準偏差。
# ------------------------------------------------------------
import sys
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
sys.path.append("src")
import config as C
from lib_eval import expanding_window_pooled, summarize
from lib_features import StationTargetEncoder

try:
    import japanize_matplotlib  # noqa
except ImportError:
    pass

# S-lag入りの完成テーブルがあればそれを、無ければT-lagまでのテーブルを使う
table = C.DATA_DIR / "tokyo23_model_table.csv"
if not table.exists():
    table = C.DATA_DIR / "tokyo23_with_tlag.csv"
df = pd.read_csv(table)
print(f"=== 8. 特徴量アブレーション (data={table.name}) ===")

subg = {"Central5": C.CENTRAL_5,
        "Others": [w for w in C.WARDS_23 if w not in C.CENTRAL_5]}
PROP = ["Age", "Area_num", "Is_Renovated", "Is_RC", "Rooms", "FAR_CAR_ratio"]
LAG = [c for c in ["Lag1_AvgPrice", "S_Lag1_AvgPrice"] if c in df.columns]


def evaluate(onehot, num_feats, use_te):
    pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), onehot),
                             ("num", "passthrough", num_feats)]) if onehot else \
          ColumnTransformer([("num", "passthrough", num_feats)])
    mk = lambda: Pipeline([("pre", pre), ("reg", xgb.XGBRegressor(**C.XGB_PARAMS))])
    ft = StationTargetEncoder(C.TARGET, m=C.TE_SMOOTHING) if use_te else None
    rec, _ = expanding_window_pooled(
        df, mk, (onehot or []) + num_feats, C.TARGET, C.EXPANDING_TEST_YEARS,
        iqr_trim=C.IQR_TRIM, iqr_k=C.IQR_K, iqr_by_ward=C.IQR_BY_WARD,
        subgroups=subg, fold_transform=ft)
    s = summarize(rec)
    g = lambda sc, nm: s[(s.scope == sc) & (s.name == nm)].iloc[0]
    a, c5, ot = g("ALL", "Tokyo23"), g("GROUP", "Central5"), g("GROUP", "Others")
    return dict(r2=a.r2_mean, r2_std=a.r2_std, r2_trim=a.r2_trim_mean,
                C5=c5.r2_mean, Others=ot.r2_mean)


# (1) 前進ネスト ----------------------------------------------------
forward = [
    ("物件のみ",          [],                          PROP, False),
    ("+Station_min",     [],                          PROP + ["Station_min"], False),
    ("+Municipality",    ["Municipality"],            PROP + ["Station_min"], False),
    ("+Zoning",          ["Municipality", "Zoning"],  PROP + ["Station_min"], False),
    ("+Station_TE(駅名)", ["Municipality", "Zoning"],  PROP + ["Station_min", "Station_TE"], True),
    ("+Lag(=フル)",       ["Municipality", "Zoning"],  PROP + ["Station_min", "Station_TE"] + LAG, True),
]
rows_f = []
print("\n[前進ネスト]")
for tag, oh, nf, te in forward:
    r = evaluate(oh, nf, te)
    r["model"] = tag
    rows_f.append(r)
    print(f"  {tag:20s} ALL {r['r2']:.3f}±{r['r2_std']:.3f}  "
          f"C5 {r['C5']:.3f}  Others {r['Others']:.3f}")
fwd = pd.DataFrame(rows_f)[["model", "r2", "r2_std", "r2_trim", "C5", "Others"]]

# (2) Leave-one-out -------------------------------------------------
FULL_OH = ["Municipality", "Zoning"]
FULL_NUM = PROP + ["Station_min", "Station_TE"] + LAG
full = evaluate(FULL_OH, FULL_NUM, True)
loo_specs = [
    ("フル",          FULL_OH,                 FULL_NUM, True),
    ("−Station_min", FULL_OH,                 [f for f in FULL_NUM if f != "Station_min"], True),
    ("−Station_TE",  FULL_OH,                 [f for f in FULL_NUM if f != "Station_TE"], False),
    ("−Zoning",      ["Municipality"],        FULL_NUM, True),
]
# 時間ラグ(自町の過去)と空間ラグ(近隣の過去)を個別に抜いて寄与を分離する。
# S_Lag が無い環境（shapefile未配置）では該当行を自動スキップ。
if "Lag1_AvgPrice" in df.columns:
    loo_specs.append(("−T_Lag(自町)", FULL_OH,
                      [f for f in FULL_NUM if f != "Lag1_AvgPrice"], True))
if "S_Lag1_AvgPrice" in df.columns:
    loo_specs.append(("−S_Lag(近隣)", FULL_OH,
                      [f for f in FULL_NUM if f != "S_Lag1_AvgPrice"], True))
loo_specs.append(("−Lag(両方)", FULL_OH,
                  [f for f in FULL_NUM if f not in LAG], True))
rows_l = []
print("\n[Leave-one-out (ΔはフルからのR²低下)]")
for tag, oh, nf, te in loo_specs:
    r = evaluate(oh, nf, te)
    r["model"] = tag
    r["delta"] = r["r2"] - full["r2"]
    rows_l.append(r)
    print(f"  {tag:14s} ALL {r['r2']:.3f}  Δ{r['delta']:+.3f}  "
          f"C5 {r['C5']:.3f}  Others {r['Others']:.3f}")
loo = pd.DataFrame(rows_l)[["model", "r2", "r2_trim", "delta", "C5", "Others"]]

fwd.to_csv(C.OUT_DIR / "ablation_forward.csv", index=False, encoding="utf-8-sig")
loo.to_csv(C.OUT_DIR / "ablation_loo.csv", index=False, encoding="utf-8-sig")

# 図: Leave-one-out の寄与（=抜いたときのR²低下幅）
contrib = loo[loo["model"] != "フル"].copy()
contrib["loss"] = -contrib["delta"]
contrib = contrib.sort_values("loss")
fig, ax = plt.subplots(figsize=(7, 4))
colors = ["seagreen" if l < 0.02 else "indianred" for l in contrib["loss"]]
ax.barh(contrib["model"], contrib["loss"], color=colors)
ax.set_xlabel("ΔR² when removed (larger = more important)")
ax.set_title("Marginal contribution of each feature (leave-one-out)")
ax.grid(True, axis="x", ls="--", alpha=0.5)
plt.tight_layout()
fig.savefig(C.OUT_DIR / "fig_ablation.png", dpi=150, bbox_inches="tight")
print(f"\n保存: outputs/ablation_forward.csv, ablation_loo.csv, fig_ablation.png")