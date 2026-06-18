# 02_features.py
# 特徴量エンジニアリング ＋ 決定論的クレンジングのみ。
# 統計的な外れ値トリミング(IQR)はここでは行わない（リーク防止のため 05 の訓練フォールド内で実施）。
import pandas as pd
import config as C

print("=== 2. 特徴量エンジニアリング（決定論的クレンジングのみ） ===")
df = pd.read_csv(C.RAW_DIR / "tokyo23_mansion_clean.csv")
print(f"読み込み: {len(df)}件")

# 面積：'2,000㎡以上' のようなカンマ・単位を除去してから数値化
df["Area_num"] = (df["Area"].astype(str).str.replace(",", "", regex=False)
                  .str.extract(r"(\d+)").astype(float))

# 価格
df["TradePrice"] = pd.to_numeric(df["TradePrice"], errors="coerce")

# 築年数：'戦前' は1945年扱い、それ以外は西暦を抽出
byear = df["BuildingYear"].astype(str)
df["BuildYear_num"] = byear.str.extract(r"(\d{4})")[0].astype(float)
df.loc[byear.str.contains("戦前", na=False), "BuildYear_num"] = 1945
df["Age"] = df["Year"] - df["BuildYear_num"]

# 平米単価(万円/㎡) = 目的変数
df[C.TARGET] = (df["TradePrice"] / 10000) / df["Area_num"]

# フラグ・カテゴリ（FloorPlan/Structure は01でNFKC正規化済み → 半角で判定できる）
df["Is_Renovated"] = (df["Renovation"] == "改装済み").astype(int)
df["Is_RC"] = df["Structure"].astype(str).str.contains("RC", na=False).astype(int)
df["Rooms"] = df["FloorPlan"].astype(str).str.extract(r"(\d+)").astype(float)
df["CoverageRatio"] = pd.to_numeric(df["CoverageRatio"], errors="coerce")
df["FloorAreaRatio"] = pd.to_numeric(df["FloorAreaRatio"], errors="coerce")
df["FAR_CAR_ratio"] = df["FloorAreaRatio"] / df["CoverageRatio"]

# Station_min は01で数値化済み（そのまま特徴量として使う）
df["Station_min"] = pd.to_numeric(df["Station_min"], errors="coerce")

# 決定論的クレンジング：計算不能・あり得ない値のみ除去（恣意的な価格閾値は使わない）
before = len(df)
df = df[(df["Area_num"] > 0) & (df["TradePrice"] > 0)]
df = df.dropna(subset=[C.TARGET, "Age", "Area_num"])
df = df[df["Age"] >= 0]
print(f"クレンジング: {before} → {len(df)}件")
print(f"  Rooms 取得率: {df['Rooms'].notna().mean():.1%}, "
      f"Station_min 取得率: {df['Station_min'].notna().mean():.1%}")

out = C.DATA_DIR / "tokyo23_features.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"保存: {out}")
