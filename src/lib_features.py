# src/lib_features.py
# ------------------------------------------------------------
# フォールド内で学習する特徴量変換。
# 駅名ターゲットエンコーディングは目的変数を使うため、
# 02(全データ)でやるとテストの価格情報が訓練に漏れる。
# そこで expanding_window_pooled の fold_transform フックから
# 「訓練フォールドのみで fit → 訓練/テスト両方に transform」する。
# ------------------------------------------------------------
import numpy as np


class StationTargetEncoder:
    """最寄駅名称 → 平滑化付き平均単価。
    enc(駅) = (駅の合計 + m × 全体平均) / (駅の件数 + m)
    少数物件の駅は全体平均に寄り、未知駅は全体平均になる。"""

    def __init__(self, target, col="StationName", out_col="Station_TE", m=50):
        self.target, self.col, self.out_col, self.m = target, col, out_col, m

    def fit(self, train_df):
        g = train_df.groupby(self.col)[self.target].agg(["sum", "count"])
        self.global_mean = train_df[self.target].mean()
        self.mapping = ((g["sum"] + self.m * self.global_mean)
                        / (g["count"] + self.m)).to_dict()
        return self

    def transform(self, df):
        df = df.copy()
        df[self.out_col] = (df[self.col].map(self.mapping)
                            .fillna(self.global_mean).astype(float))
        return df

    def __call__(self, tr, te):
        """fold_transform フック用: (train, test) を受けて変換後を返す。"""
        self.fit(tr)
        return self.transform(tr), self.transform(te)
