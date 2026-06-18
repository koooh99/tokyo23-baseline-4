# 05_baseline_models.py
# ベースライン：23区プールで OLS と XGBoost を1モデルずつ学習し、
# Expanding Window で複数年検証して「分布」で評価する。
# 区別・サブグループ別(都心5/その他)のR²は予測の事後スライスで算出。
import sys
import numpy as np
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

print("=== 5. ベースライン評価（プール OLS / XGBoost） ===")
df = pd.read_csv(C.DATA_DIR / "tokyo23_model_table.csv")

subgroups = {"Central5": C.CENTRAL_5,
             "Others": [w for w in C.WARDS_23 if w not in C.CENTRAL_5]}

# プールするので区の情報を特徴量に入れる：Municipality を one-hot ＋ 重心座標。
# これで「1つのモデル」のまま区差を表現でき、Kobayashiさんの提案に沿う。
def numeric_features(lag_cols):
    feats = C.BASE_FEATURES + lag_cols
    feats += [c for c in C.SPATIAL_FEATURES if df[c].notna().any()]
    if C.STATION_TE:
        feats += ["Station_TE"]   # フォールド内で生成される
    return feats

ONEHOT_COLS = ["Municipality"] + [c for c in C.CAT_FEATURES if c in df.columns]
print(f"one-hot: {ONEHOT_COLS} / 駅TE: {C.STATION_TE}")

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
for lag_name, lag_cols in C.LAG_CONFIGS.items():
    num_feats = numeric_features(lag_cols)
    feat_cols = ONEHOT_COLS + num_feats
    for model_name, factory in (("OLS", make_ols(num_feats)),
                                ("XGB", make_xgb(num_feats))):
        ft = (StationTargetEncoder(C.TARGET, m=C.TE_SMOOTHING)
              if C.STATION_TE else None)
        rec, _ = expanding_window_pooled(
            df, factory, feat_cols, C.TARGET, C.EXPANDING_TEST_YEARS,
            iqr_trim=C.IQR_TRIM, iqr_k=C.IQR_K, iqr_by_ward=C.IQR_BY_WARD,
            subgroups=subgroups, fold_transform=ft)
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
                  f"({int(r['n_folds'])}年)")

summary = pd.concat(all_summaries, ignore_index=True)
out = C.OUT_DIR / "baseline_summary.csv"
summary.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n保存: {out}")
print("\n[サブグループ別 平均R²]")
print(summary[summary["scope"] == "GROUP"]
      .sort_values(["lag", "model", "name"])
      [["lag", "model", "name", "r2_mean", "r2_std", "r2_trim_mean", "r2_trim_std", "n_folds"]]
      .round(3).to_string(index=False))
