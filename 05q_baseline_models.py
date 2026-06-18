# 05q_baseline_models.py
# 05_baseline_models.py の四半期版。Expanding Window を「四半期フォールド」で回す。
#   各 Qidx について < Qidx を訓練 / == Qidx をテスト（C.EXPANDING_TEST_QUARTERS）。
# T-lag/S-lag は四半期版（直前1Q・前年同期）を使う（C.LAG_CONFIGS_Q）。
# 年次版の outputs/baseline_summary.csv は上書きせず、_q 版を別ファイルに出すので
# 年次 vs 四半期を後から比較できる。
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
sys.path.append("src")
import config as C
from lib_eval import expanding_window_pooled, summarize
from lib_features import StationTargetEncoder

print("=== 5Q. ベースライン評価（四半期フォールド / プール OLS・XGBoost） ===")
df = pd.read_csv(C.DATA_DIR / "tokyo23_model_table_q.csv")

subgroups = {"Central5": C.CENTRAL_5,
             "Others": [w for w in C.WARDS_23 if w not in C.CENTRAL_5]}


def numeric_features(lag_cols):
    feats = C.BASE_FEATURES + lag_cols
    feats += [c for c in C.SPATIAL_FEATURES if df[c].notna().any()]
    if C.STATION_TE:
        feats += ["Station_TE"]   # フォールド内で生成される
    return feats


ONEHOT_COLS = ["Municipality"] + [c for c in C.CAT_FEATURES if c in df.columns]
print(f"one-hot: {ONEHOT_COLS} / 駅TE: {C.STATION_TE} / "
      f"四半期フォールド数: {len(C.EXPANDING_TEST_QUARTERS)}")


def make_ols(num_feats):
    pre = ColumnTransformer([
        ("oh", OneHotEncoder(handle_unknown="ignore"), ONEHOT_COLS),
        ("num", "passthrough", num_feats),
    ])
    return lambda: Pipeline([("pre", pre), ("reg", LinearRegression())])


def make_xgb(num_feats):
    pre = ColumnTransformer([
        ("oh", OneHotEncoder(handle_unknown="ignore"), ONEHOT_COLS),
        ("num", "passthrough", num_feats),
    ])
    return lambda: Pipeline([("pre", pre),
                             ("reg", xgb.XGBRegressor(**C.XGB_PARAMS))])


all_summaries = []
for lag_name, lag_cols in C.LAG_CONFIGS_Q.items():
    num_feats = numeric_features(lag_cols)
    feat_cols = ONEHOT_COLS + num_feats
    for model_name, factory in (("OLS", make_ols(num_feats)),
                                ("XGB", make_xgb(num_feats))):
        ft = (StationTargetEncoder(C.TARGET, m=C.TE_SMOOTHING)
              if C.STATION_TE else None)
        rec, _ = expanding_window_pooled(
            df, factory, feat_cols, C.TARGET, C.EXPANDING_TEST_QUARTERS,
            iqr_trim=C.IQR_TRIM, iqr_k=C.IQR_K, iqr_by_ward=C.IQR_BY_WARD,
            subgroups=subgroups, fold_transform=ft, time_col="Qidx")
        s = summarize(rec)
        s.insert(0, "lag", lag_name)
        s.insert(1, "model", model_name)
        all_summaries.append(s)
        overall = s[s["scope"] == "ALL"]
        if not overall.empty:
            r = overall.iloc[0]
            print(f"  {lag_name}/{model_name}: 全体 R² = "
                  f"{r['r2_mean']:.3f} ± {r['r2_std']:.3f} "
                  f"| trim併記 {r['r2_trim_mean']:.3f} ± {r['r2_trim_std']:.3f} "
                  f"({int(r['n_folds'])}四半期)")

summary = pd.concat(all_summaries, ignore_index=True)
out = C.OUT_DIR / "baseline_summary_q.csv"
summary.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n保存: {out}")
print("\n[サブグループ別 平均R²（四半期）]")
print(summary[summary["scope"] == "GROUP"]
      .sort_values(["lag", "model", "name"])
      [["lag", "model", "name", "r2_mean", "r2_std", "r2_trim_mean", "r2_trim_std", "n_folds"]]
      .round(3).to_string(index=False))
