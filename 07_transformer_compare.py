# 07_transformer_compare.py
# Transformer を「比較対象の一つ」として、XGB と同一プロトコル
# （同じ特徴量・同じ Expanding Window・同じIQR運用）で比較する。
# 出力には r2(素) と r2_trim(訓練由来閾値でテストもトリム) を併記する。
#
# 計算時間の目安: Transformer は 16フォルド学習なのでXGBより重い。
# まず Lag1 のみ・TEST_YEARS を絞って様子を見るのも可（下の TEST_YEARS を編集）。
import sys
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
sys.path.append("src")
import config as C
from lib_eval import expanding_window_pooled, summarize
from lib_transformer import FTTransformerRegressor
from lib_features import StationTargetEncoder

LAG_NAME = "Lag1"                       # ベストだったラグ次数で比較
# CPUだと重いので既定は直近8年。全期間にするなら C.EXPANDING_TEST_YEARS に。
# MPS(Apple GPU)はTransformerEncoderでハングし得るため既定CPU。試すなら環境変数 TF_DEVICE=mps。
TEST_YEARS = list(range(2018, 2026))

print(f"=== 7. Transformer vs XGB 比較 ({LAG_NAME}) ===")
df = pd.read_csv(C.DATA_DIR / "tokyo23_model_table.csv")

lag_cols = C.LAG_CONFIGS[LAG_NAME]
num_feats = C.BASE_FEATURES + lag_cols + \
    [c for c in C.SPATIAL_FEATURES if df[c].notna().any()]
if C.STATION_TE:
    num_feats += ["Station_TE"]
ONEHOT_COLS = ["Municipality"] + [c for c in C.CAT_FEATURES if c in df.columns]
feat_cols = ONEHOT_COLS + num_feats
subgroups = {"Central5": C.CENTRAL_5,
             "Others": [w for w in C.WARDS_23 if w not in C.CENTRAL_5]}

def make_xgb():
    pre = ColumnTransformer([
        ("oh", OneHotEncoder(handle_unknown="ignore"), ONEHOT_COLS),
        ("num", "passthrough", num_feats)])
    return Pipeline([("pre", pre), ("reg", xgb.XGBRegressor(**C.XGB_PARAMS))])

def make_transformer():
    return FTTransformerRegressor(
        num_cols=num_feats, cat_cols=ONEHOT_COLS,
        d_model=64, n_heads=4, n_layers=3, dropout=0.1,
        lr=1e-3, batch_size=2048, max_epochs=20, patience=4,
        max_train=60000,          # CPUで現実的に終わる上限
        verbose=True,             # フォールド/エポックの進捗を表示
        random_state=C.RANDOM_STATE)

results = []
for model_name, factory in (("XGB", make_xgb), ("Transformer", make_transformer)):
    print(f"\n--- {model_name} ---")
    ft = (StationTargetEncoder(C.TARGET, m=C.TE_SMOOTHING)
          if C.STATION_TE else None)
    rec, _ = expanding_window_pooled(
        df, factory, feat_cols, C.TARGET, TEST_YEARS,
        iqr_trim=C.IQR_TRIM, iqr_k=C.IQR_K, iqr_by_ward=C.IQR_BY_WARD,
        subgroups=subgroups, fold_transform=ft)
    s = summarize(rec)
    s.insert(0, "lag", LAG_NAME)
    s.insert(1, "model", model_name)
    results.append(s)
    a = s[s["scope"] == "ALL"].iloc[0]
    print(f"  全体: R² {a['r2_mean']:.3f} ± {a['r2_std']:.3f} | "
          f"trim併記 {a['r2_trim_mean']:.3f} ± {a['r2_trim_std']:.3f} "
          f"({int(a['n_folds'])}年)")

out_df = pd.concat(results, ignore_index=True)
out = C.OUT_DIR / "transformer_vs_xgb.csv"
out_df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n保存: {out}")

print("\n[サブグループ別: r2 / r2_trim]")
view = out_df[out_df["scope"] == "GROUP"][
    ["model", "name", "r2_mean", "r2_std", "r2_trim_mean", "r2_trim_std", "n_folds"]]
print(view.round(3).to_string(index=False))
