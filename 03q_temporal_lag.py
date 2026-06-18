# 03q_temporal_lag.py
# 03_temporal_lag.py（年次ラグ）の四半期版。年次版と出力ファイルを分けているので、
# 既存の tokyo23_with_tlag.csv（年次）は上書きせず、比較用に残る。
#
# 取引時期 '2025年第1四半期' を四半期通し番号 Qidx = Year*4 + (Quarter-1) に変換し、
# 町丁目×四半期の平均平米単価から T-lag を2種類作る：
#   Lag_q1_AvgPrice … 直前1四半期 (Qidx-1)
#   Lag_q4_AvgPrice … 前年同期(4Q前) (Qidx-4)
# 欠四半期があり得るので groupby.shift は使わず、暦ベースの結合で正しい四半期を引く。
import unicodedata
import pandas as pd
import config as C

print("=== 3Q. 四半期ラグ(T-lag)の作成 ===")
df = pd.read_csv(C.DATA_DIR / "tokyo23_features.csv")

# 取引時期 '2025年第1四半期' → 年・四半期。全角混入に備えてNFKC正規化してから抽出。
period = df["Period"].astype(str).map(lambda s: unicodedata.normalize("NFKC", s))
df["Year"] = period.str.extract(r"(\d{4})年")[0].astype(float)
df["Quarter"] = period.str.extract(r"第(\d)四半期")[0].astype(float)
# 四半期通し番号。Qidx-1=直前1Q、Qidx-4=前年同期。欠四半期があっても暦どおりに引ける。
df["Qidx"] = (df["Year"] * 4 + (df["Quarter"] - 1)).astype("Int64")

# 町丁目×四半期の平均平米単価
q = (df.groupby(["Municipality", "DistrictName", "Qidx"], dropna=True)[C.TARGET]
       .mean().reset_index().rename(columns={C.TARGET: "AvgPrice_Sqm"}))

# 暦ベースの結合でラグを作成（back四半期前の値を現在のQidxに合わせて結合する）
base = q[["Municipality", "DistrictName", "Qidx", "AvgPrice_Sqm"]]
for col, back in (("Lag_q1_AvgPrice", 1), ("Lag_q4_AvgPrice", 4)):
    src = base.rename(columns={"AvgPrice_Sqm": col}).copy()
    src["Qidx"] = src["Qidx"] + back
    q = q.merge(src[["Municipality", "DistrictName", "Qidx", col]],
                on=["Municipality", "DistrictName", "Qidx"], how="left")

merged = df.merge(
    q[["Municipality", "DistrictName", "Qidx",
       "Lag_q1_AvgPrice", "Lag_q4_AvgPrice"]],
    on=["Municipality", "DistrictName", "Qidx"], how="left")

# --- 取得率（ラグが非欠損で取れた割合）---------------------------------------
lag_cols = ["Lag_q1_AvgPrice", "Lag_q4_AvgPrice"]
print("[T-lag 取得率]")
print(f"{'lag':<16}{'物件行ベース':>14}{'町丁目×四半期ベース':>22}")
for c in lag_cols:
    print(f"{c:<16}{merged[c].notna().mean():>13.1%}{q[c].notna().mean():>21.1%}")

# 年次版(03)の Lag1 取得率も読めれば併記して比較しやすくする
yearly_path = C.DATA_DIR / "tokyo23_with_tlag.csv"
if yearly_path.exists():
    yr = pd.read_csv(yearly_path, usecols=lambda c: c == "Lag1_AvgPrice")
    if "Lag1_AvgPrice" in yr.columns:
        print(f"  [参考] 年次 Lag1_AvgPrice 取得率(物件行): "
              f"{yr['Lag1_AvgPrice'].notna().mean():.1%}")

out = C.DATA_DIR / "tokyo23_with_tlag_q.csv"
merged.to_csv(out, index=False, encoding="utf-8-sig")
print(f"保存: {out}  ({len(merged)}件)")
