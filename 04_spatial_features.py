# 04_spatial_features.py
# 重心座標・最寄り駅距離・空間ラグ(S-lag) を付与する。
# 旧 05_s_lag.py と違い、隣接は 23区を一括で計算する（区境の取りこぼしを解消）。
import sys
import pandas as pd
sys.path.append("src")
import config as C
from lib_spatial import (load_town_polygons, add_centroids, build_neighbors,
                         compute_spatial_lag)

print(f"=== 4. 空間特徴量の作成 (weight={C.SPATIAL_WEIGHT}) ===")
df = pd.read_csv(C.DATA_DIR / "tokyo23_with_tlag.csv")

print("町丁目ポリゴンを読み込み中...")
towns = load_town_polygons(C.SHAPEFILE_PATH, C.WARDS_23)
print(f"  町単位: {len(towns)}件")

# 重心座標
cent = add_centroids(towns, C.PROJECTED_CRS)
df = df.merge(cent, on=["Municipality", "DistrictName"], how="left")

# 駅距離は01でCSVの「最寄駅：距離（分）」から物件単位で付与済み（Station_min）。
# ここでは何もしない。

# 空間ラグ：23区一括の隣接（区をまたぐ）で近隣の時間ラグを平均
print("近隣ネットワーク構築 → 空間ラグ計算中...")
neighbors = build_neighbors(towns, method=C.SPATIAL_WEIGHT,
                            knn_k=C.KNN_K, projected_crs=C.PROJECTED_CRS)
town_year = (df[["Municipality", "DistrictName", "Year",
                 "Lag1_AvgPrice", "Lag2_AvgPrice", "Lag3_AvgPrice"]]
             .drop_duplicates())
s_lag = compute_spatial_lag(town_year, neighbors,
                            ["Lag1_AvgPrice", "Lag2_AvgPrice", "Lag3_AvgPrice"])
df = df.merge(s_lag, on=["Municipality", "DistrictName", "Year"], how="left")

out = C.DATA_DIR / "tokyo23_model_table.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"保存: {out}  ({len(df)}件)")
