# src/lib_spatial.py
# ------------------------------------------------------------
# 空間処理の共通ロジック。旧 05_s_lag.py の最大の問題
# （区ごとに閉じて隣接を計算 → 区境の町が隣区の近隣を取りこぼす）を解消する。
# ここでは 23区の町丁目ポリゴンを「一括で」扱い、区をまたいだ隣接を許す。
# ------------------------------------------------------------
import re
import geopandas as gpd
import numpy as np
import pandas as pd


def load_town_polygons(shapefile_path, wards, ward_col_candidates=("CITY_NAME", "GST_NAME")):
    """町丁目shapefileを読み込み、23区に絞り、丁目を落として町単位にdissolveする。
    返り値の GeoDataFrame は ['Municipality', 'DistrictName', 'geometry'] を持つ。
    旧コードと違い、区ごとに分けず 1枚の地図として返す点が肝。"""
    gdf = gpd.read_file(shapefile_path)
    ward_col = next((c for c in ward_col_candidates if c in gdf.columns), None)
    if ward_col is None:
        raise KeyError(f"区名の列が見つかりません。候補: {ward_col_candidates} / 実際: {list(gdf.columns)}")

    gdf = gdf[gdf[ward_col].isin(wards)].copy()
    # 「○丁目」を除去して町名に正規化
    gdf["DistrictName"] = gdf["S_NAME"].apply(
        lambda x: re.sub(r"[一二三四五六七八九十百０-９0-9]+丁目$", "", str(x))
    )
    gdf["Municipality"] = gdf[ward_col]
    # 区またぎで同名の町があり得るので Municipality+DistrictName でdissolve
    dissolved = gdf.dissolve(by=["Municipality", "DistrictName"]).reset_index()
    return dissolved[["Municipality", "DistrictName", "geometry"]]


def add_centroids(gdf_towns, projected_crs):
    """各町の重心座標(投影後 メートル系)を返す。特徴量 centroid_x / centroid_y に使う。"""
    proj = gdf_towns.to_crs(projected_crs)
    cent = proj.geometry.centroid
    out = gdf_towns[["Municipality", "DistrictName"]].copy()
    out["centroid_x"] = cent.x.values
    out["centroid_y"] = cent.y.values
    return out


def build_neighbors(gdf_towns, method="queen", knn_k=6, projected_crs="EPSG:6677"):
    """町ごとの近隣リストを {(区, 町): [(区, 町), ...]} で返す。
    method="queen": 行政区をまたいで隣接（境界を共有する町すべて）。
    method="knn"  : 重心間距離の k 近傍（行政界に一切依存しない）。
    どちらも 23区を一括で扱うので、区境の取りこぼしが起きない。"""
    keys = list(zip(gdf_towns["Municipality"], gdf_towns["DistrictName"]))

    if method == "knn":
        proj = gdf_towns.to_crs(projected_crs)
        cent = proj.geometry.centroid
        coords = np.column_stack([cent.x.values, cent.y.values])
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(knn_k + 1, len(coords))).fit(coords)
        _, idx = nn.kneighbors(coords)
        neighbors = {}
        for i, key in enumerate(keys):
            neighbors[key] = [keys[j] for j in idx[i] if j != i]
        return neighbors

    # queen（既定）: libpysal があれば高速・正確に。無ければ intersects でフォールバック。
    try:
        from libpysal.weights import Queen
        w = Queen.from_dataframe(gdf_towns, use_index=False, silence_warnings=True)
        return {keys[i]: [keys[j] for j in w.neighbors[i]] for i in range(len(keys))}
    except Exception:
        sindex = gdf_towns.sindex
        neighbors = {}
        geoms = gdf_towns.geometry.values
        for i, key in enumerate(keys):
            cand = list(sindex.query(geoms[i], predicate="intersects"))
            neighbors[key] = [keys[j] for j in cand if j != i and geoms[i].touches(geoms[j])]
        return neighbors


def compute_spatial_lag(df_town_year, neighbors, lag_cols, time_col="Year"):
    """近隣の(時間)ラグ価格を平均して空間ラグ S_Lag* を作る。
    df_town_year は ['Municipality','DistrictName', time_col, *lag_cols] を町×時点で持つ前提。
    time_col は年次なら "Year"、四半期なら "Qidx" を渡す（04q が四半期で呼ぶ）。"""
    idx = {(r.Municipality, r.DistrictName, getattr(r, time_col)): r
           for r in df_town_year.itertuples(index=False)}
    out_rows = []
    periods = sorted(df_town_year[time_col].unique())
    for (muni, dist), grp in df_town_year.groupby(["Municipality", "DistrictName"]):
        nb = neighbors.get((muni, dist), [])
        for period in periods:
            vals = {c: [] for c in lag_cols}
            for (n_muni, n_dist) in nb:
                rec = idx.get((n_muni, n_dist, period))
                if rec is None:
                    continue
                for c in lag_cols:
                    v = getattr(rec, c)
                    if pd.notna(v):
                        vals[c].append(v)
            row = {"Municipality": muni, "DistrictName": dist, time_col: period}
            for c in lag_cols:
                row["S_" + c] = np.mean(vals[c]) if vals[c] else np.nan
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def nearest_station_distance(gdf_towns, stations_path, projected_crs):
    """各町の重心から最寄り駅までの距離(m)。stations_path が無ければ None を返す（駅距離はスキップ）。"""
    from pathlib import Path
    if not Path(stations_path).exists():
        return None
    stations = gpd.read_file(stations_path)
    proj_towns = gdf_towns.to_crs(projected_crs).copy()
    proj_towns["geometry"] = proj_towns.geometry.centroid
    proj_st = stations.to_crs(projected_crs)
    joined = gpd.sjoin_nearest(proj_towns, proj_st[["geometry"]], distance_col="dist_to_station_m")
    joined = joined.drop_duplicates(subset=["Municipality", "DistrictName"])
    return joined[["Municipality", "DistrictName", "dist_to_station_m"]]
