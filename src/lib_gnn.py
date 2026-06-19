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
from torch_geometric.nn import GCNConv

from lib_temporal import build_aggregator


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
    """2層GCNで町表現を作り、物件特徴と結合して平米単価(標準化後)を回帰する。"""
    def __init__(self, n_node_feat, n_prop_feat, hidden=32):
        super().__init__()
        self.g1 = GCNConv(n_node_feat, hidden)
        self.g2 = GCNConv(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_prop_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, prop_node_idx, prop_feat):
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
          __init__ 内の「g1,g2 の後・head の前」で build する。"""
    def __init__(self, n_node_feat, n_prop_feat, hidden=32, aggregator="gru"):
        super().__init__()
        self.g1 = GCNConv(n_node_feat, hidden)
        self.g2 = GCNConv(hidden, hidden)
        self.aggregator = build_aggregator(aggregator, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_prop_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, x, edge_index):
        h = torch.relu(self.g1(x, edge_index))
        return torch.relu(self.g2(h, edge_index))           # [N, hidden]

    def temporal(self, h_seq):
        return self.aggregator(h_seq)                        # [L,N,hidden] → [N,hidden]

    def predict(self, h, prop_node_idx, prop_feat):
        z = torch.cat([h[prop_node_idx], prop_feat], dim=1)
        return self.head(z).squeeze(-1)


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
                    epochs=40, lr=1e-3, device="cpu", verbose=False):
    """1フォールド学習＋評価。
    snapshots[q] = dict(x=[N,Fn] node特徴(標準化済), node_idx=[n] 物件→ノード,
                        prop=[n,Fp] 物件特徴(標準化済), y=[n] 目的(標準化済), y_raw=[n] 生値)。
    train_q: 訓練に使う四半期リスト / test_q: 評価する単一四半期。
    返り値: (pred_raw[np], y_raw[np], muni[np])  ※muniは呼び出し側で持たせる。"""
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
        pred = model(te["x"], ei, te["node_idx"], te["prop"]).cpu().numpy()
    return pred
