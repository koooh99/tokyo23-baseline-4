# src/lib_eval.py
# ------------------------------------------------------------
# 評価の共通ロジック。議事録の中心的な懸念
# 「R²=0.7 は特定年にたまたま当たっただけかもしれない」に正面から答える。
#   - 単年の一発R²ではなく、複数の検証年にわたる分布(平均±標準偏差)で報告する
#   - 23区プールで1つのモデルを学習し、区別・サブグループ別のR²は予測の事後スライスで出す
#   - IQRトリミングは訓練フォールドの分布からのみ閾値を決める（リーク防止）
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def trim_by_iqr(train_df, target, k=1.5, by_ward=True):
    """訓練データの分布だけから外れ値の閾値を決め、訓練データをトリミングして返す。
    返り値は (トリミング後train, 区ごとの[下限,上限])。テストには一切触れない。"""
    bounds = {}
    if by_ward:
        keep = []
        for ward, g in train_df.groupby("Municipality"):
            lo, hi = iqr_bounds(g[target], k)
            bounds[ward] = (lo, hi)
            keep.append(g[(g[target] >= lo) & (g[target] <= hi)])
        return pd.concat(keep), bounds
    lo, hi = iqr_bounds(train_df[target], k)
    bounds["_all"] = (lo, hi)
    return train_df[(train_df[target] >= lo) & (train_df[target] <= hi)], bounds


def _apply_bounds(df, target, bounds):
    """訓練フォールドで決めた閾値(bounds)をそのまま適用してトリミングする。
    テスト側に使ってもテストの分布は見ていないのでリークしない。"""
    if "_all" in bounds:
        lo, hi = bounds["_all"]
        return df[(df[target] >= lo) & (df[target] <= hi)]
    keep = []
    for ward, g in df.groupby("Municipality"):
        if ward in bounds:
            lo, hi = bounds[ward]
            keep.append(g[(g[target] >= lo) & (g[target] <= hi)])
        else:
            keep.append(g)  # 訓練に無かった区はそのまま
    return pd.concat(keep) if keep else df


def _r2_pair(g, target, bounds):
    """素のR²と、訓練由来閾値でテストもトリミングしたR²(r2_trim)を返す。"""
    r2 = r2_score(g[target], g["pred"])
    r2_trim = np.nan
    if bounds is not None:
        gt = _apply_bounds(g, target, bounds)
        if len(gt) >= 10:
            r2_trim = r2_score(gt[target], gt["pred"])
    return r2, r2_trim


def expanding_window_pooled(df, make_model, features, target, test_years,
                            iqr_trim=True, iqr_k=1.5, iqr_by_ward=True,
                            subgroups=None, fold_transform=None,
                            progress=True, time_col="Year"):
    """23区プールの Expanding Window 評価。
    各検証時点 y について「< y を訓練 / == y をテスト」で1モデルを学習し、
    全体・区別・サブグループ別のR²を記録する。
    time_col … 時間軸の列名。年次なら "Year"(既定)、四半期なら "Qidx"(05q が使う)。
    r2      … テストをトリミングしない正直な値（主指標）
    r2_trim … 訓練フォールド由来のIQR閾値をテストにも適用した感度分析値（併記用）
    fold_transform … (train, test)->(train, test)。訓練フォールドのみでfitする
                     特徴量変換（駅ターゲットエンコーディング等）をここに挿す。
                     IQRトリミング後に呼ぶので符号化はトリム済み分布で学習される。
    返り値: (時点×粒度のlong DataFrame, 予測明細DataFrame)"""
    subgroups = subgroups or {}
    records, preds = [], []
    need = [c for c in list(features) if c in df.columns] \
        + [target, time_col, "Municipality"]

    for i, y in enumerate(test_years, 1):
        d = df.dropna(subset=need)
        tr = d[d[time_col] < y].copy()
        te = d[d[time_col] == y].copy()
        if len(tr) < 200 or len(te) < 50:
            continue
        if progress:
            print(f"  [fold {i}/{len(test_years)}] テスト{time_col}={y} "
                  f"(訓練{len(tr):,}件 / テスト{len(te):,}件)", flush=True)
        bounds = None
        if iqr_trim:
            tr, bounds = trim_by_iqr(tr, target, iqr_k, iqr_by_ward)
        if fold_transform is not None:
            tr, te = fold_transform(tr, te)

        model = make_model()
        model.fit(tr[features], tr[target])
        te = te.assign(pred=model.predict(te[features]))
        preds.append(te.assign(test_year=y))

        def rec(scope, name, g):
            r2, r2_trim = _r2_pair(g, target, bounds)
            records.append({"test_year": y, "scope": scope, "name": name,
                            "n": len(g), "r2": r2, "r2_trim": r2_trim})

        rec("ALL", "Tokyo23", te)
        for ward, g in te.groupby("Municipality"):
            if len(g) >= 10:
                rec("WARD", ward, g)
        for label, ward_list in subgroups.items():
            g = te[te["Municipality"].isin(ward_list)]
            if len(g) >= 10:
                rec("GROUP", label, g)

    return pd.DataFrame(records), (pd.concat(preds) if preds else pd.DataFrame())


def summarize(records):
    """年をまたいだ平均±標準偏差にまとめる。r2(素) と r2_trim(テストもトリム) を併記。"""
    if records.empty:
        return records
    g = records.groupby(["scope", "name"])
    out = pd.DataFrame({
        "r2_mean": g["r2"].mean(),
        "r2_std": g["r2"].std(),
        "r2_min": g["r2"].min(),
        "r2_max": g["r2"].max(),
        "r2_trim_mean": g["r2_trim"].mean(),
        "r2_trim_std": g["r2_trim"].std(),
        "n_folds": g["r2"].count(),
    }).reset_index()
    return out.sort_values(["scope", "r2_mean"], ascending=[True, False])
