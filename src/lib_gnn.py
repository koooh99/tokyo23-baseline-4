# src/lib_gnn.py
# ------------------------------------------------------------
# 時空間GNN（最小版 v0）の部品。
#   - 町丁目 queen 隣接をグラフ(edge_index)にする
#   - 各四半期スナップショットの「自町ラグ＋重心」をノード特徴にする
#   - 2層GCNで近隣伝播 → 学習した町表現を物件特徴と結合してMLPで平米単価を回帰
# 手作りS-lag(近隣平均)を「グラフ伝播」に置き換えられるかを見るのが目的。
# 時間方向は当面 Expanding Window(四半期) で表現する（系列モデルは次段）。
# ------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv

from lib_temporal import build_aggregator


# 名前 → conv クラス。run_compare / 09 が文字列で空間conv層を選べるようにする
# （lib_temporal.AGGREGATORS の空間版）。
SPATIAL_CONVS = {"gcn": GCNConv, "gat": GATConv, "gatv2": GATv2Conv}


def build_spatial_conv(spec, in_dim, out_dim, heads=1, concat=True):
    """空間conv層を1つ構築する。spec は "gcn"|"gat"|"gatv2"。
      - "gcn": GCNConv(in,out)。対称正規化による近隣の等重み平均（heads/concatは無視）。
        既存挙動を一切変えないため、GCNConv(in_dim,out_dim) を素直に返す
        （パラメータ初期化のRNG順も従来と同一＝既知値を再現する）。
      - "gat"/"gatv2": 同じ edge_index 上で辺の注意重みのみを学習する。
        heads/concat で出力次元を制御（呼び出し側で hidden に揃える）。
        self-loop は PyG 既定(add_self_loops=True)＝GCNConvと同じく自己ループを含めて
        近隣分布を正規化する（GCN版との整合）。"""
    if spec == "gcn":
        return GCNConv(in_dim, out_dim)
    cls = SPATIAL_CONVS.get(spec)
    if cls is None:
        raise ValueError(f"未知の空間conv: {spec!r}（候補: {list(SPATIAL_CONVS)}）")
    return cls(in_dim, out_dim, heads=heads, concat=concat)


def build_graph(node_keys, neighbors):
    """node_keys: [(Municipality, DistrictName), ...]（データに出現する町の順序）。
    neighbors: {(区,町): [(区,町), ...]}。両端がnode_keysに在る辺だけ採用。
    返り値: (key->idx dict, edge_index[2,E] LongTensor)。孤立ノードはGCNの自己ループで残る。"""
    idx = {k: i for i, k in enumerate(node_keys)}
    src, dst = [], []
    for k, i in idx.items():
        for nb in neighbors.get(k, []):
            j = idx.get(nb)
            if j is not None:
                src.append(i); dst.append(j)   # 無向グラフ：両方向を入れる
    if not src:                                 # 念のため（隣接が空）
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    return idx, edge_index


class SpatioGCN(nn.Module):
    """2層の空間convで町表現を作り、物件特徴と結合して平米単価(標準化後)を回帰する。

    conv: 空間conv層の指定（"gcn"|"gat"|"gatv2"）。**変えるのはconv層だけ**で、
        hidden次元・層数・head(物件回帰MLP)・出力次元は全構成で完全一致させる
        （フェア比較の生命線）。既定 "gcn" は従来実装と同一計算・同一パラメータ初期化順
        （g1→g2→head）で既知値を再現する。
      - "gcn": 2層とも GCNConv。最終出力 hidden。
      - "gat"/"gatv2": 1層目 heads=GAT_HEADS concat=True → hidden*heads、
        2層目 heads=1 concat=False → hidden。最終出力は GCN版と同じ hidden。
        同じ edge_index 上で辺の注意重みのみを学習する。

    forward(..., return_attention=True) のとき、1層目(多ヘッド)の辺注意を
    self.last_attention [E,heads] / self.last_att_edge_index [2,E] に保持する
    （§6dの時間attentionの空間版＝学習近隣重みの解釈分析用）。"""
    def __init__(self, n_node_feat, n_prop_feat, hidden=32, conv="gcn", heads=4):
        super().__init__()
        self.conv_kind = conv
        if conv == "gcn":
            self.g1 = build_spatial_conv("gcn", n_node_feat, hidden)
            self.g2 = build_spatial_conv("gcn", hidden, hidden)
        else:
            self.g1 = build_spatial_conv(conv, n_node_feat, hidden,
                                         heads=heads, concat=True)   # → hidden*heads
            self.g2 = build_spatial_conv(conv, hidden * heads, hidden,
                                         heads=1, concat=False)      # → hidden
        self.head = nn.Sequential(
            nn.Linear(hidden + n_prop_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.last_attention = None          # [E,heads]（1層目, GAT系のみ）
        self.last_att_edge_index = None     # [2,E]（self-loop付与後の辺, GAT系のみ）

    def forward(self, x, edge_index, prop_node_idx, prop_feat, return_attention=False):
        if self.conv_kind == "gcn":
            h = torch.relu(self.g1(x, edge_index))
        elif return_attention:
            h, (ei, alpha) = self.g1(x, edge_index, return_attention_weights=True)
            self.last_attention = alpha.detach()
            self.last_att_edge_index = ei.detach()
            h = torch.relu(h)
        else:
            h = torch.relu(self.g1(x, edge_index))
        h = torch.relu(self.g2(h, edge_index))
        hp = h[prop_node_idx]                       # 各物件の所在町の表現
        z = torch.cat([hp, prop_feat], dim=1)
        return self.head(z).squeeze(-1)


class SpatioTemporalGNN(nn.Module):
    """各時点に2層GCN(重み共有)で空間表現を作り、直近Lステップの系列を
    時間集約器(lib_temporal)で1つの町表現に畳む。集約器を差し替えることで
    「どの過去四半期を重視するか」の学習方法（GRU / 注意 / 平均 / 直前）を比較できる。
    最終の町表現を物件特徴と結合してMLPで平米単価を回帰する。
    使い方: encode()で各時点のH[N,hidden]を作り（1エポックで使い回す）、
    temporal()で系列H[L,N,hidden]→h[N,hidden]、predict()で物件回帰。

    aggregator: 時間集約器の指定。文字列("gru"/"attention"/"mean"/"last")・
        hidden→Module の factory・構築済みModule のいずれか。既定 "gru" は
        旧実装(GRU直書き)と同一計算・同一パラメータ初期化順で、既知値を再現する。
        ※GRU初期化のRNG順を旧実装(g1,g2,gru,head)に合わせるため、集約器は
          __init__ 内の「g1,g2 の後・head の前」で build する。

    spatial: 空間エンコーダの指定。"gcn"(既定)=2層GCNConvで近隣メッセージパッシング、
        "none"=メッセージパッシングをしない per-node エンコーダ（g1/g2 を nn.Linear に
        差し替え、encode() が edge_index を使わない）。"none" は in→hidden・層数・hidden
        次元・head を "gcn" と完全に揃え、**近隣集約の有無だけ**を外す（2×2要因デザインの
        graph トグル, 12_factorial_2x2.py が使う）。既定 "gcn" は計算・パラメータ初期化順
        (g1,g2,aggregator,head) とも従来と一切変えない＝既知値を再現する。"""
    def __init__(self, n_node_feat, n_prop_feat, hidden=32, aggregator="gru",
                 spatial="gcn"):
        super().__init__()
        if spatial not in ("gcn", "none"):
            raise ValueError(f"spatial は 'gcn'|'none' のいずれか: {spatial!r}")
        self.spatial = spatial
        if spatial == "gcn":
            self.g1 = GCNConv(n_node_feat, hidden)
            self.g2 = GCNConv(hidden, hidden)
        else:
            # 近隣集約なしの per-node エンコーダ（edge_index を使わない線形2層）。
            # GCN版と in→hidden・層数・hidden 次元を揃え、メッセージパッシングのみ外す。
            self.g1 = nn.Linear(n_node_feat, hidden)
            self.g2 = nn.Linear(hidden, hidden)
        self.aggregator = build_aggregator(aggregator, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_prop_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, x, edge_index):
        if self.spatial == "gcn":
            h = torch.relu(self.g1(x, edge_index))
            return torch.relu(self.g2(h, edge_index))       # [N, hidden]
        # spatial=="none": edge_index は IF 互換のため受け取るが使わない（近隣集約なし）。
        h = torch.relu(self.g1(x))
        return torch.relu(self.g2(h))                       # [N, hidden]

    def temporal(self, h_seq, meta=None):
        # meta はステップ別メタ情報(gap[L]/obs[L,N])の任意 dict。既存集約器は無視する。
        return self.aggregator(h_seq, meta)                  # [L,N,hidden] → [N,hidden]

    def predict(self, h, prop_node_idx, prop_feat):
        z = torch.cat([h[prop_node_idx], prop_feat], dim=1)
        return self.head(z).squeeze(-1)


def assert_conv_only_diff(n_node_feat, n_prop_feat, hidden, heads, conv="gatv2"):
    """フェア比較の生命線：GCN版とconv版が「conv層(g1,g2)以外は完全一致」であることを
    検証し、ログ用dictを返す。head(物件回帰MLP)のパラメータ形状と最終空間表現の次元
    (=hidden)が一致することをassertする。差分が conv 層だけであることを明示するのに使う。"""
    gcn = SpatioGCN(n_node_feat, n_prop_feat, hidden=hidden, conv="gcn")
    alt = SpatioGCN(n_node_feat, n_prop_feat, hidden=hidden, conv=conv, heads=heads)

    def head_shapes(m):
        return {n: tuple(p.shape) for n, p in m.named_parameters()
                if n.startswith("head.")}
    hg, ha = head_shapes(gcn), head_shapes(alt)
    assert hg == ha, f"head が不一致（conv層以外が違う）: {hg} vs {ha}"
    assert gcn.head[0].in_features == alt.head[0].in_features, \
        "最終空間表現の次元が不一致（conv層以外が違う）"
    return {"conv": conv, "heads": heads, "hidden_out": hidden,
            "head_in": gcn.head[0].in_features, "head_shapes": hg,
            "n_node_feat": n_node_feat, "n_prop_feat": n_prop_feat}


def assert_aggregator_only_diff(n_node_feat, n_prop_feat, hidden, aggregator):
    """フェア比較の生命線（時間集約器版）：GRU 実行と aggregator 実行が
    「時間集約器(aggregator)以外は完全一致」であることを検証し、ログ用 dict を返す。
    空間conv g1/g2（GCN）と物件回帰 head のパラメータ形状が GRU 版と一致することを
    assert する。差分が aggregator サブモジュールだけであることを明示するのに使う。
    （集約器ごとにパラメータ数は当然違う＝それが比較したい唯一の差分。g1/g2/head が
    一致していれば「変えたのは集約器のみ」が担保される。）"""
    base = SpatioTemporalGNN(n_node_feat, n_prop_feat, hidden=hidden, aggregator="gru")
    alt = SpatioTemporalGNN(n_node_feat, n_prop_feat, hidden=hidden, aggregator=aggregator)

    def shapes(m, prefix):
        return {n: tuple(p.shape) for n, p in m.named_parameters()
                if n.startswith(prefix)}
    for pre in ("g1.", "g2.", "head."):
        sb, sa = shapes(base, pre), shapes(alt, pre)
        assert sb == sa, f"{pre} が不一致（集約器以外が違う）: {sb} vs {sa}"
    return {"aggregator": aggregator, "hidden": hidden,
            "n_node_feat": n_node_feat, "n_prop_feat": n_prop_feat,
            "g1": shapes(base, "g1."), "g2": shapes(base, "g2."),
            "head": shapes(base, "head."),
            "agg_gru": list(shapes(base, "aggregator.").items()),
            "agg_alt": list(shapes(alt, "aggregator.").items())}


def assert_factorial_only_diff(n_node_feat, n_prop_feat, hidden):
    """2×2要因デザイン（空間=近隣波及 graph × 時間=動態 temporal）のフェア比較の生命線。
    4セルを構築し、(graph, temporal) の2トグル以外が完全一致であることを実行前に検証する:
      - head(物件回帰MLP)のパラメータ形状が4セルで一致
      - 最終空間表現の次元(=hidden, =head入力−物件特徴)が4セルで一致
      - 空間エンコーダ g1/g2 の (in,out) 次元が4セルで一致（GCNConv でも nn.Linear でも）
      - n_node_feat / n_prop_feat が4セルで一致
    graph トグルは g1/g2 を GCNConv↔per-node nn.Linear に、temporal トグルは集約器を
    GRU↔LastStep に切り替える。差分がこの2箇所だけであることをログ用 dict で返す。
    （graph トグルでモジュール型が変わるため g1/g2 のパラメータ“名”は一致しない＝それが
      比較したい差分の一つ。代わりに in→out 次元の一致で「同じ形・同じ層数」を担保する。）"""
    cells = {
        ("on", "on"):   dict(spatial="gcn",  aggregator="gru"),
        ("on", "off"):  dict(spatial="gcn",  aggregator="last"),
        ("off", "on"):  dict(spatial="none", aggregator="gru"),
        ("off", "off"): dict(spatial="none", aggregator="last"),
    }
    models = {k: SpatioTemporalGNN(n_node_feat, n_prop_feat, hidden=hidden, **v)
              for k, v in cells.items()}

    def head_shapes(m):
        return {n: tuple(p.shape) for n, p in m.named_parameters()
                if n.startswith("head.")}

    def io_dims(layer):                          # GCNConv / nn.Linear 両対応
        if hasattr(layer, "in_features"):
            return (layer.in_features, layer.out_features)
        return (layer.in_channels, layer.out_channels)

    ref_key = ("on", "on")
    ref = models[ref_key]
    ref_head = head_shapes(ref)
    ref_g1, ref_g2 = io_dims(ref.g1), io_dims(ref.g2)
    ref_head_in = ref.head[0].in_features
    for k, m in models.items():
        assert head_shapes(m) == ref_head, \
            f"head が不一致（2トグル以外が違う）: {k} {head_shapes(m)} vs {ref_head}"
        assert m.head[0].in_features == ref_head_in, \
            f"最終空間表現の次元が不一致（2トグル以外が違う）: {k}"
        assert io_dims(m.g1) == ref_g1 and io_dims(m.g2) == ref_g2, \
            f"空間エンコーダ g1/g2 の(in,out)次元が不一致: {k} " \
            f"{(io_dims(m.g1), io_dims(m.g2))} vs {(ref_g1, ref_g2)}"
    assert ref_g1 == (n_node_feat, hidden) and ref_g2 == (hidden, hidden)
    return {
        "cells": {f"graph={g},temporal={t}": cells[(g, t)]
                  for (g, t) in cells},
        "shared": {"hidden": hidden, "head_in": ref_head_in,
                   "n_node_feat": n_node_feat, "n_prop_feat": n_prop_feat,
                   "g1_io": ref_g1, "g2_io": ref_g2, "head_shapes": ref_head,
                   "layers": 2},
        "toggles": {"graph": {"on": "GCNConv(近隣メッセージパッシング)",
                              "off": "nn.Linear(per-node, 近隣集約なし)"},
                    "temporal": {"on": "GRUAggregator", "off": "LastStepAggregator"}},
    }


class Standardizer:
    """列ごとの平均・標準偏差で標準化。NaNは平均(=標準化後0)で埋める。fitは訓練のみ。"""
    def __init__(self):
        self.mean = None; self.std = None

    def fit(self, X):
        self.mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        self.std = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X):
        Z = (X - self.mean) / self.std
        return np.nan_to_num(Z, nan=0.0)


def train_eval_fold(model, snapshots, edge_index, train_q, test_q,
                    epochs=40, lr=1e-3, device="cpu", verbose=False,
                    return_attention=False):
    """1フォールド学習＋評価。
    snapshots[q] = dict(x=[N,Fn] node特徴(標準化済), node_idx=[n] 物件→ノード,
                        prop=[n,Fp] 物件特徴(標準化済), y=[n] 目的(標準化済), y_raw=[n] 生値)。
    train_q: 訓練に使う四半期リスト / test_q: 評価する単一四半期。
    return_attention=True のとき、評価forwardで GAT系1層目の辺注意を
    model.last_attention / model.last_att_edge_index に格納する（GCN版では無視される）。
    返り値: pred[np]（標準化後の予測）。"""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ei = edge_index.to(device)
    loss_fn = nn.MSELoss()

    # テンソル化（標準化はフォールド前に済ませてある）
    def to_t(s):
        return dict(
            x=torch.tensor(s["x"], dtype=torch.float32, device=device),
            node_idx=torch.tensor(s["node_idx"], dtype=torch.long, device=device),
            prop=torch.tensor(s["prop"], dtype=torch.float32, device=device),
            y=torch.tensor(s["y"], dtype=torch.float32, device=device),
        )
    tr = {q: to_t(snapshots[q]) for q in train_q}
    te = to_t(snapshots[test_q])

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for q in train_q:                           # 四半期ごとのミニバッチSGD
            s = tr[q]
            opt.zero_grad()
            pred = model(s["x"], ei, s["node_idx"], s["prop"])
            loss = loss_fn(pred, s["y"])
            loss.backward(); opt.step()
            ep_loss += loss.item()
        if verbose and (ep + 1) % 10 == 0:
            print(f"      epoch {ep + 1}/{epochs} loss={ep_loss / len(train_q):.4f}",
                  flush=True)

    model.eval()
    with torch.no_grad():
        pred = model(te["x"], ei, te["node_idx"], te["prop"],
                     return_attention=return_attention).cpu().numpy()
    return pred
