# 03_temporal_lag.py
# 町丁目×年の平均平米単価から時間ラグ Lag1/2/3 を作成。
import pandas as pd
import config as C

print("=== 3. 時間ラグ(T-lag)の作成 ===")
df = pd.read_csv(C.DATA_DIR / "tokyo23_features.csv")

yearly = (df.groupby(["Municipality", "DistrictName", "Year"])[C.TARGET]
            .mean().reset_index().rename(columns={C.TARGET: "AvgPrice_Sqm"})
            .sort_values(["Municipality", "DistrictName", "Year"]))

g = yearly.groupby(["Municipality", "DistrictName"])["AvgPrice_Sqm"]
for k in (1, 2, 3):
    yearly[f"Lag{k}_AvgPrice"] = g.shift(k)

merged = df.merge(
    yearly[["Municipality", "DistrictName", "Year",
            "Lag1_AvgPrice", "Lag2_AvgPrice", "Lag3_AvgPrice"]],
    on=["Municipality", "DistrictName", "Year"], how="left")

out = C.DATA_DIR / "tokyo23_with_tlag.csv"
merged.to_csv(out, index=False, encoding="utf-8-sig")
print(f"保存: {out}  ({len(merged)}件)")
