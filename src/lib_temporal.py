# src/lib_temporal.py
# ------------------------------------------------------------
# 時間集約（temporal aggregation）を差し替え式にするための共通部品。
# 時空間GNNは「各時点の空間表現 H_t [N,hidden] を直近Lステップ分積んだ系列
# h_seq [L,N,hidden]」を1つの町表現 h [N,hidden] に畳む。この畳み方を
# モデル本体から切り離し、共通インターフェースで差し替えられるようにする。
#
# 共通IF: forward(h_seq[L,N,hidden], meta=None) -> h[N,hidden]
#   - 系列の並びは過去→現在（h_seq[0]=最古 q-L, h_seq[-1]=直前 q-1）。
#   - meta はステップ別メタ情報の dict（後方互換のため任意・既定 None）:
#       meta["gap"] : [L] 対象四半期 q からの時間ギャップ（=q−step四半期, 整数 L..1）
#       meta["obs"] : [L,N] 観測フラグ（1=実観測 / 0=前方補完で埋めたステップ）
#     既存集約器（GRU/Attention/Mean/Last）は meta を無視する＝完全後方互換。
#     time-aware 集約器（TimeAwareAttention）だけがこれを「直近を区別する材料」に使う。
#   - 重みを持つ集約器（attention 系）は forward 後に self.last_weights [N,L] を保持し、
#     「どの過去四半期を重視したか」を後から可視化できる（§6d の重み分析と同形式）。
#
# リーク防止の系列定義（seq_steps=[q-L,…,q-1]）は呼び出し側（10/run_compare）で
# 担保する。本モジュールは並びを所与として畳むだけで、時間方向の参照はしない。
# meta も呼び出し側が同じ過去限定の並びで作って渡す（本モジュールは所与とする）。
# ------------------------------------------------------------
import torch
import torch.nn as nn


class TemporalAggregator(nn.Module):
    """時間集約器の基底。入力 h_seq [L,N,hidden]（＋任意の meta dict）→ 出力 h [N,hidden]。
    forward の第2引数 meta はステップ別メタ情報（gap[L]/obs[L,N]）の dict で、
    既定 None。既存集約器は meta を受け取っても無視する（後方互換）。
    重みを持つ集約器（attention 系）は forward 後に last_weights [N,L] を埋める。"""

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

    def forward(self, h_seq, meta=None):
        _, h_n = self.gru(h_seq)                 # h_seq [L,N,hidden]（meta は無視）
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

    def forward(self, h_seq, meta=None):
        # h_seq [L,N,hidden]（meta は無視＝§6d と同一計算）
        e = self.score(torch.tanh(self.proj(h_seq))).squeeze(-1)   # [L,N]
        w = torch.softmax(e, dim=0)                                 # ステップ方向に正規化
        self.last_weights = w.transpose(0, 1).detach()             # [N,L]（可視化用）
        return (w.unsqueeze(-1) * h_seq).sum(dim=0)                # [N,hidden]


class TimeAwareAttention(TemporalAggregator):
    """§6d の AttentionAggregator を time-aware に拡張した加法注意。
    §6d では「前方補完で隣接四半期の表現が似すぎ softmax が discriminate できない」
    として重みが一様(≈1/L)に潰れた。本集約器は softmax に区別材料を与える:
        score_t = v^T tanh(W h_t + g(gap_t))            （variant 共通）
      - g(gap_t): 対象四半期 q からの時間ギャップ gap_t∈{1..L} の**学習埋め込み**を
        key に加える。前方補完で h_t がほぼ同一でも、ギャップ埋め込みが直近/遠方を
        区別する自由度になる（全 L ステップに注意を張る）。
    variant="obs" は上に加え、非観測（前方補完）ステップのスコアを obs_penalty だけ
    強く減衰（実質マスク）し、実観測ステップ中心に注意を張る＝系列を不等間隔として扱う:
        score_t ← score_t + (obs_t − 1)·obs_penalty      （obs_t=0 のとき −penalty）
    どちらも forward 後に last_weights[N,L]（§6d と同形式）を保持する。
    meta=None（旧シグネチャ呼び出し）では gap/obs の材料が無く §6d の素の加法注意に
    縮退する（ギャップ埋め込みは meta が無ければ加算されない）。

    フェア比較の不変条件: 空間 conv(GCN)・L・hidden・層数・head・optimizer・lr・epoch・
    seed は GRU 実行と同一。**変えるのは集約器のみ**（lib_gnn.assert_aggregator_only_diff
    で g1/g2/head の一致を実行前に検証する）。"""

    returns_weights = True

    def __init__(self, hidden, attn_dim=None, variant="gap",
                 max_gap=64, obs_penalty=8.0):
        super().__init__()
        attn_dim = attn_dim or hidden
        if variant not in ("gap", "obs"):
            raise ValueError(f"variant は 'gap'|'obs' のいずれか: {variant!r}")
        self.variant = variant
        self.obs_penalty = float(obs_penalty)
        self.proj = nn.Linear(hidden, attn_dim)
        # gap=1..max_gap を埋め込む（index 0 は未使用・ギャップ0のダミー）。
        self.gap_emb = nn.Embedding(max_gap + 1, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h_seq, meta=None):
        # h_seq [L,N,hidden]
        key = self.proj(h_seq)                                      # [L,N,attn_dim]
        if meta is not None and meta.get("gap") is not None:
            gap = meta["gap"].to(key.device).long().clamp(
                0, self.gap_emb.num_embeddings - 1)                 # [L]
            key = key + self.gap_emb(gap).unsqueeze(1)             # +[L,1,attn_dim]
        e = self.score(torch.tanh(key)).squeeze(-1)                # [L,N]
        if self.variant == "obs" and meta is not None and meta.get("obs") is not None:
            obs = meta["obs"].to(e.device)                         # [L,N]（1=観測,0=補完）
            e = e + (obs - 1.0) * self.obs_penalty                # 非観測に −penalty
        w = torch.softmax(e, dim=0)                                # ステップ方向に正規化
        self.last_weights = w.transpose(0, 1).detach()            # [N,L]（可視化用）
        return (w.unsqueeze(-1) * h_seq).sum(dim=0)               # [N,hidden]


class MeanAggregator(TemporalAggregator):
    """アブレーション用の単純基準：全ステップの単純平均（時間重みなし）。"""

    def __init__(self, hidden=None):
        super().__init__()

    def forward(self, h_seq, meta=None):
        return h_seq.mean(dim=0)


class LastStepAggregator(TemporalAggregator):
    """アブレーション用の単純基準：直前ステップ(q-1)のみを使う（=時間集約なし）。"""

    def __init__(self, hidden=None):
        super().__init__()

    def forward(self, h_seq, meta=None):
        return h_seq[-1]


# 名前 → クラス。run_compare 等が文字列で集約器を選べるようにする。
AGGREGATORS = {
    "gru": GRUAggregator,
    "attention": AttentionAggregator,
    "mean": MeanAggregator,
    "last": LastStepAggregator,
    # time-aware 注意（①）。factory が hidden を受けて variant を固定する。
    "time_attn_gap": lambda hidden: TimeAwareAttention(hidden, variant="gap"),
    "time_attn_obs": lambda hidden: TimeAwareAttention(hidden, variant="obs"),
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
