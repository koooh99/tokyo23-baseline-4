# 04q_spatial_features.py
# 04_spatial_features.py の四半期版。重心座標は同じ。空間ラグ(S-lag)を四半期インデックス
# Qidx で計算する（近隣の同一四半期の T-lag を平均）。
# 年次版の tokyo23_model_table.csv は上書きせず、_q 版を別ファイルに出す。
import sys
import pandas as pd
sys.path.append("src")
import config as C
from lib_spatial import (load_town_polygons, add_centroids, build_neighbors,
                         compute_spatial_lag)

QLAG_COLS = ["Lag_q1_AvgPrice", "Lag_q4_AvgPrice"]

print(f"=== 4Q. 空間特徴量（四半期S-lag, weight={C.SPATIAL_WEIGHT}) ===")
df = pd.read_csv(C.DATA_DIR / "tokyo23_with_tlag_q.csv")

print("町丁目ポリゴンを読み込み中...")
towns = load_town_polygons(C.SHAPEFILE_PATH, C.WARDS_23)
print(f"  町単位: {len(towns)}件")

# 重心座標
cent = add_centroids(towns, C.PROJECTED_CRS)
df = df.merge(cent, on=["Municipality", "DistrictName"], how="left")

# 空間ラグ：23区一括の隣接（区をまたぐ）で近隣の四半期T-lagを平均
print("近隣ネットワーク構築 → 四半期空間ラグ計算中...")
neighbors = build_neighbors(towns, method=C.SPATIAL_WEIGHT,
                            knn_k=C.KNN_K, projected_crs=C.PROJECTED_CRS)
town_q = (df[["Municipality", "DistrictName", "Qidx", *QLAG_COLS]]
          .drop_duplicates())
s_lag = compute_spatial_lag(town_q, neighbors, QLAG_COLS, time_col="Qidx")
df = df.merge(s_lag, on=["Municipality", "DistrictName", "Qidx"], how="left")

print("[S-lag 取得率（物件行ベース）]")
for c in QLAG_COLS:
    print(f"  S_{c}: {df['S_' + c].notna().mean():.1%}")

out = C.DATA_DIR / "tokyo23_model_table_q.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"保存: {out}  ({len(df)}件)")
