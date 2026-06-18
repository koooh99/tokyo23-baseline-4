# 08q_ablation.py
# ------------------------------------------------------------
# 08_ablation.py の四半期版。S-lag を四半期で作り直したテーブル
# (tokyo23_model_table_q.csv) で、立地系特徴量とラグの寄与を再確認する。
# 既定ラグ構成は Qboth（直前1Q + 前年同期、それぞれ T-lag/S-lag）。
# 四半期化で S-lag の寄与が年次と変わったかを GNN 着手前に見ておくのが目的。
# 評価は四半期フォールド(EXPANDING_TEST_QUARTERS)。年次版 08 の出力は上書きしない。
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

table = C.DATA_DIR / "tokyo23_model_table_q.csv"
df = pd.read_csv(table)
print(f"=== 8Q. 特徴量アブレーション（四半期, 既定={C.DEFAULT_LAG_NAME}, "
      f"data={table.name}） ===")

subg = {"Central5": C.CENTRAL_5,
        "Others": [w for w in C.WARDS_23 if w not in C.CENTRAL_5]}
PROP = ["Age", "Area_num", "Is_Renovated", "Is_RC", "Rooms", "FAR_CAR_ratio"]
TLAG = [c for c in C.DEFAULT_TLAG_COLS if c in df.columns]   # 自町: Lag_q1/Lag_q4
SLAG = [c for c in C.DEFAULT_SLAG_COLS if c in df.columns]   # 近隣: S_Lag_q1/S_Lag_q4
LAG = TLAG + SLAG
# 季節(前年同期) vs 直近(直前1Q) を分離して抜くためのグループ
SEASONAL = [c for c in ["Lag_q4_AvgPrice", "S_Lag_q4_AvgPrice"] if c in df.columns]
RECENT = [c for c in ["Lag_q1_AvgPrice", "S_Lag_q1_AvgPrice"] if c in df.columns]


def evaluate(onehot, num_feats, use_te):
    pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), onehot),
                             ("num", "passthrough", num_feats)]) if onehot else \
          ColumnTransformer([("num", "passthrough", num_feats)])
    mk = lambda: Pipeline([("pre", pre), ("reg", xgb.XGBRegressor(**C.XGB_PARAMS))])
    ft = StationTargetEncoder(C.TARGET, m=C.TE_SMOOTHING) if use_te else None
    rec, _ = expanding_window_pooled(
        df, mk, (onehot or []) + num_feats, C.TARGET, C.EXPANDING_TEST_QUARTERS,
        iqr_trim=C.IQR_TRIM, iqr_k=C.IQR_K, iqr_by_ward=C.IQR_BY_WARD,
        subgroups=subg, fold_transform=ft, time_col="Qidx", progress=False)
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
    ("+Lag(=フル/Qboth)", ["Municipality", "Zoning"],  PROP + ["Station_min", "Station_TE"] + LAG, True),
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
    ("フル",          FULL_OH, FULL_NUM, True),
    ("−Station_min", FULL_OH, [f for f in FULL_NUM if f != "Station_min"], True),
    ("−Station_TE",  FULL_OH, [f for f in FULL_NUM if f != "Station_TE"], False),
    ("−Zoning",      ["Municipality"], FULL_NUM, True),
]
# 時間ラグ(自町) と 空間ラグ(近隣) を個別に抜いて寄与を分離（年次との比較が主目的）
if TLAG:
    loo_specs.append(("−T_Lag(自町)", FULL_OH,
                      [f for f in FULL_NUM if f not in TLAG], True))
if SLAG:
    loo_specs.append(("−S_Lag(近隣)", FULL_OH,
                      [f for f in FULL_NUM if f not in SLAG], True))
# 直近(直前1Q) と 季節(前年同期) を分離して抜く（四半期化で増えた切り口）
if RECENT:
    loo_specs.append(("−直前1Q(T+S)", FULL_OH,
                      [f for f in FULL_NUM if f not in RECENT], True))
if SEASONAL:
    loo_specs.append(("−前年同期(T+S)", FULL_OH,
                      [f for f in FULL_NUM if f not in SEASONAL], True))
loo_specs.append(("−Lag(全部)", FULL_OH,
                  [f for f in FULL_NUM if f not in LAG], True))
rows_l = []
print("\n[Leave-one-out (ΔはフルからのR²低下)]")
for tag, oh, nf, te in loo_specs:
    r = evaluate(oh, nf, te)
    r["model"] = tag
    r["delta"] = r["r2"] - full["r2"]
    rows_l.append(r)
    print(f"  {tag:16s} ALL {r['r2']:.3f}  Δ{r['delta']:+.3f}  "
          f"C5 {r['C5']:.3f}  Others {r['Others']:.3f}")
loo = pd.DataFrame(rows_l)[["model", "r2", "r2_trim", "delta", "C5", "Others"]]

fwd.to_csv(C.OUT_DIR / "ablation_forward_q.csv", index=False, encoding="utf-8-sig")
loo.to_csv(C.OUT_DIR / "ablation_loo_q.csv", index=False, encoding="utf-8-sig")

# 図: Leave-one-out の寄与（=抜いたときのR²低下幅）
contrib = loo[loo["model"] != "フル"].copy()
contrib["loss"] = -contrib["delta"]
contrib = contrib.sort_values("loss")
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ["seagreen" if l < 0.02 else "indianred" for l in contrib["loss"]]
ax.barh(contrib["model"], contrib["loss"], color=colors)
ax.set_xlabel("ΔR² when removed (larger = more important)")
ax.set_title(f"四半期アブレーション: 各特徴量の寄与 (leave-one-out, {C.DEFAULT_LAG_NAME})")
ax.grid(True, axis="x", ls="--", alpha=0.5)
plt.tight_layout()
fig.savefig(C.OUT_DIR / "fig_ablation_q.png", dpi=150, bbox_inches="tight")
print(f"\n保存: outputs/ablation_forward_q.csv, ablation_loo_q.csv, fig_ablation_q.png")
