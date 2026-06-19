# src/lib_temporal.py
# ------------------------------------------------------------
# 時間集約（temporal aggregation）を差し替え式にするための共通部品。
# 時空間GNNは「各時点の空間表現 H_t [N,hidden] を直近Lステップ分積んだ系列
# h_seq [L,N,hidden]」を1つの町表現 h [N,hidden] に畳む。この畳み方を
# モデル本体から切り離し、共通インターフェースで差し替えられるようにする。
#
# 共通IF: forward(h_seq[L,N,hidden]) -> h[N,hidden]
#   - 系列の並びは過去→現在（h_seq[0]=最古 q-L, h_seq[-1]=直前 q-1）。
#   - AttentionAggregator は forward 後に self.last_weights [N,L] を保持し、
#     「どの過去四半期を重視したか」を後から可視化できる（次段の重み可視化用）。
#
# リーク防止の系列定義（seq_steps=[q-L,…,q-1]）は呼び出し側（10/run_compare）で
# 担保する。本モジュールは並びを所与として畳むだけで、時間方向の参照はしない。
# ------------------------------------------------------------
import torch
import torch.nn as nn


class TemporalAggregator(nn.Module):
    """時間集約器の基底。入力 h_seq [L,N,hidden] → 出力 h [N,hidden]。
    重みを持つ集約器（attention）は forward 後に last_weights [N,L] を埋める。"""

    #: True の集約器は forward 後に last_weights [N,L] を提供する
    returns_weights = False

    def __init__(self):
        super().__init__()
        self.last_weights = None   # [N,L]（重みを持たない集約器では None のまま）


class GRUAggregator(TemporalAggregator):
    """現行の時間集約を完全維持する版（GCN+GRU の GRU 部分）。
    系列をGRUに通し、最終隠れ状態 h_n[-1] を町表現とする。
    元の SpatioTemporalGNN.temporal() と同一の計算・同一のパラメータ構成。"""

    def __init__(self, hidden):
        super().__init__()
        self.gru = nn.GRU(hidden, hidden)        # 入力 [L,N,hidden]（Nをバッチ扱い）

    def forward(self, h_seq):
        _, h_n = self.gru(h_seq)                 # h_seq [L,N,hidden]
        return h_n[-1]                           # [N,hidden]


class AttentionAggregator(TemporalAggregator):
    """L個の過去ステップへの softmax 重みを学習する加法（Bahdanau風）注意。
    score_t = v^T tanh(W h_t) をステップ方向に softmax し、重み付き和で畳む。
    forward 後に self.last_weights [N,L]（町ごとの過去四半期重み）を保持する。"""

    returns_weights = True

    def __init__(self, hidden, attn_dim=None):
        super().__init__()
        attn_dim = attn_dim or hidden
        self.proj = nn.Linear(hidden, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h_seq):
        # h_seq [L,N,hidden]
        e = self.score(torch.tanh(self.proj(h_seq))).squeeze(-1)   # [L,N]
        w = torch.softmax(e, dim=0)                                 # ステップ方向に正規化
        self.last_weights = w.transpose(0, 1).detach()             # [N,L]（可視化用）
        return (w.unsqueeze(-1) * h_seq).sum(dim=0)                # [N,hidden]


class MeanAggregator(TemporalAggregator):
    """アブレーション用の単純基準：全ステップの単純平均（時間重みなし）。"""

    def __init__(self, hidden=None):
        super().__init__()

    def forward(self, h_seq):
        return h_seq.mean(dim=0)


class LastStepAggregator(TemporalAggregator):
    """アブレーション用の単純基準：直前ステップ(q-1)のみを使う（=時間集約なし）。"""

    def __init__(self, hidden=None):
        super().__init__()

    def forward(self, h_seq):
        return h_seq[-1]


# 名前 → クラス。run_compare 等が文字列で集約器を選べるようにする。
AGGREGATORS = {
    "gru": GRUAggregator,
    "attention": AttentionAggregator,
    "mean": MeanAggregator,
    "last": LastStepAggregator,
}


def build_aggregator(spec, hidden):
    """集約器を構築する。spec は次のいずれか:
      - 文字列（AGGREGATORS のキー: "gru"/"attention"/"mean"/"last"）
      - hidden を受け取り Module を返す factory（callable）
      - 既に構築済みの Module
    SpatioTemporalGNN.__init__ の内部から「g1,g2 の後・head の前」で呼ぶことで、
    GRU のパラメータ初期化順を元実装と一致させ、既知値を再現できるようにする。"""
    if isinstance(spec, str):
        try:
            return AGGREGATORS[spec](hidden)
        except KeyError:
            raise ValueError(f"未知の集約器: {spec!r}（候補: {list(AGGREGATORS)}）")
    if isinstance(spec, nn.Module):
        return spec
    if callable(spec):
        return spec(hidden)
    raise TypeError(f"集約器の指定が不正: {type(spec)}")
