# 01_load_data.py
# 不動産情報ライブラリのサイトから直接DLしたCSV(日本語列・Shift_JIS)を正本として読み込む。
# APIでは最寄駅情報が取れないため、こちらを使う。
# ここで「英語スキーマへの統一」「年の抽出」「最寄駅距離(分)の数値化」「23区・住宅への絞り込み」を行う。
import unicodedata
import pandas as pd
import config as C

print("=== 1. サイト版CSVの読み込み ===")
df = pd.read_csv(C.SOURCE_CSV, encoding=C.SOURCE_ENCODING, dtype=str)
print(f"読み込み: {len(df)}件, {df.shape[1]}列")

# 内部で使う英語列名に統一
df = df.rename(columns=C.COLMAP)

# 全角→半角などを正規化（間取り '２ＬＤＫ' や構造 'ＲＣ' が半角扱いになる）
for col in ["FloorPlan", "Structure", "Area", "StationMinutes_raw", "Zoning"]:
    df[col] = df[col].map(lambda x: unicodedata.normalize("NFKC", x) if isinstance(x, str) else x)

# 取引時期 '2021年第1四半期' → 年
df["Year"] = df["Period"].str.extract(r"(\d{4})年")[0].astype(float)

# 最寄駅：距離（分）の数値化。'30分～60分'等の範囲は代表値に置換、それ以外は数値化。
def parse_station_min(v):
    if not isinstance(v, str):
        return pd.NA
    v = v.strip()
    if v in C.STATION_MIN_RANGE:
        return C.STATION_MIN_RANGE[v]
    try:
        return float(v)
    except ValueError:
        return pd.NA
df["Station_min"] = df["StationMinutes_raw"].map(parse_station_min)

# 都市計画(用途地域): 欠損は「不明」カテゴリとして保持（除外しない）
df["Zoning"] = df["Zoning"].fillna("不明")
# 駅名: 欠損は「駅名不明」（ターゲットエンコーディングで全体平均に落ちる）
df["StationName"] = df["StationName"].fillna("駅名不明")

# 絞り込み：中古マンション(住宅) かつ 23区
before = len(df)
df = df[(df["Type"] == C.TARGET_TYPE) & (df["Use"] == C.TARGET_USE)]
df = df[df["Municipality"].isin(C.WARDS_23)].copy()
print(f"中古マンション(住宅)・23区に絞り込み: {before} → {len(df)}件")
print(f"年範囲: {int(df['Year'].min())}-{int(df['Year'].max())}")
print(df["Municipality"].value_counts())

out = C.RAW_DIR / "tokyo23_mansion_clean.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n保存: {out}")
