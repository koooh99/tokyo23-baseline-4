# src/lib_transformer.py
# ------------------------------------------------------------
# 表形式データ向け Transformer 回帰器（FT-Transformer 簡易版）。
# 議事録の方針どおり「絶対的な目標」ではなく「比較対象の一つ」。
# そのため XGB と同じ expanding_window_pooled にそのまま載る
# sklearn 互換の fit(X_df, y) / predict(X_df) を実装する。
#
# 構成: 数値特徴 → 各特徴を d_model 次元にトークン化(線形)
#       Municipality → カテゴリ埋め込みトークン
#       [CLS] トークン + TransformerEncoder → CLS から回帰ヘッド
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


import os
def _device():
    # 既定はCPU。MPS(Apple GPU)はTransformerEncoderでハングし得るのでオプトイン。
    # 使いたい場合のみ環境変数 TF_DEVICE=mps / cuda を指定する。
    want = os.environ.get("TF_DEVICE", "cpu").lower()
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "mps" and getattr(torch.backends, "mps", None) \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _FTTransformer(nn.Module):
    def __init__(self, n_num, n_cat_classes, d_model=64, n_heads=4,
                 n_layers=3, dropout=0.1):
        super().__init__()
        # 数値特徴ごとに「重み×値＋バイアス」で d_model 次元トークン化
        self.num_w = nn.Parameter(torch.randn(n_num, d_model) * 0.02)
        self.num_b = nn.Parameter(torch.zeros(n_num, d_model))
        # カテゴリ列ごとに埋め込み（各 +1 は未知カテゴリ用）
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(n + 1, d_model) for n in n_cat_classes])
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, 1))

    def forward(self, x_num, x_cat):
        B = x_num.size(0)
        tok_num = x_num.unsqueeze(-1) * self.num_w + self.num_b   # (B, n_num, d)
        tok_cat = torch.stack(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)],
            dim=1)                                                # (B, n_cat, d)
        cls = self.cls.expand(B, -1, -1)                          # (B, 1, d)
        h = torch.cat([cls, tok_cat, tok_num], dim=1)
        h = self.encoder(h)
        return self.head(h[:, 0]).squeeze(-1)


class FTTransformerRegressor:
    """sklearn風インターフェース。X は DataFrame（cat_col + num_cols を含む）。
    数値は訓練統計で標準化、目的変数も標準化して学習し predict で戻す。
    未知の区は予約インデックスに落とすので学習時に無かった区でも落ちない。"""

    def __init__(self, num_cols, cat_cols=("Municipality",),
                 d_model=64, n_heads=4, n_layers=3, dropout=0.1,
                 lr=1e-3, weight_decay=1e-5, batch_size=1024,
                 max_epochs=30, patience=4, val_frac=0.1, random_state=42,
                 max_train=60000, verbose=True):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.hp = dict(d_model=d_model, n_heads=n_heads,
                       n_layers=n_layers, dropout=dropout)
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.max_epochs = batch_size, max_epochs
        self.patience, self.val_frac = patience, val_frac
        self.max_train = max_train
        self.random_state, self.verbose = random_state, verbose

    # --- 前処理 ---
    def _encode_cat(self, X):
        cols = []
        for c in self.cat_cols:
            m = self.cat_maps[c]
            idx = X[c].astype(str).map(m).fillna(len(m)).astype(int)
            cols.append(idx.values)
        return np.column_stack(cols)  # (N, n_cat) 未知→予約index

    def _prep_num(self, X):
        v = X[self.num_cols].astype(float).values
        return (v - self.num_mean) / self.num_std

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y, dtype=np.float64)

        self.cat_maps = {}
        for c in self.cat_cols:
            cats = sorted(X[c].astype(str).unique())
            self.cat_maps[c] = {v: i for i, v in enumerate(cats)}
        self.num_mean = X[self.num_cols].astype(float).values.mean(axis=0)
        self.num_std = X[self.num_cols].astype(float).values.std(axis=0)
        self.num_std[self.num_std == 0] = 1.0
        self.y_mean, self.y_std = y.mean(), max(y.std(), 1e-9)

        xn = self._prep_num(X).astype(np.float32)
        xc = self._encode_cat(X)
        yt = ((y - self.y_mean) / self.y_std).astype(np.float32)

        n = len(yt)
        # CPUで現実的に終わるよう訓練件数を上限でサブサンプル（汎化への影響は小さい）
        if self.max_train and n > self.max_train:
            sel = rng.choice(n, self.max_train, replace=False)
            xn, xc, yt = xn[sel], xc[sel], yt[sel]
            n = self.max_train
        idx = rng.permutation(n)
        n_val = max(int(n * self.val_frac), 1)
        vi, ti = idx[:n_val], idx[n_val:]

        dev = _device()
        if dev.type == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)
        self._dev = dev
        if self.verbose:
            print(f"  [TF] fold学習開始: {n}件 (device={dev.type}, "
                  f"max_epochs={self.max_epochs})", flush=True)
        n_classes = [len(self.cat_maps[c]) for c in self.cat_cols]
        self.model = _FTTransformer(len(self.num_cols), n_classes,
                                    **self.hp).to(dev)
        opt = torch.optim.AdamW(self.model.parameters(),
                                lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        tr_ds = TensorDataset(torch.tensor(xn[ti]), torch.tensor(xc[ti]),
                              torch.tensor(yt[ti]))
        tr_dl = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
        xv = (torch.tensor(xn[vi]).to(dev), torch.tensor(xc[vi]).to(dev))
        yv = torch.tensor(yt[vi]).to(dev)

        best, best_state, bad = np.inf, None, 0
        pbar = tqdm(total=self.max_epochs * len(tr_dl), unit="batch",
                    desc="    学習中", disable=not self.verbose, leave=False)
        for ep in range(self.max_epochs):
            self.model.train()
            for bx, bc, by in tr_dl:
                bx, bc, by = bx.to(dev), bc.to(dev), by.to(dev)
                opt.zero_grad()
                loss = loss_fn(self.model(bx, bc), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                pbar.update(1)
            self.model.eval()
            with torch.no_grad():
                vloss = loss_fn(self.model(*xv), yv).item()
            pbar.set_postfix(ep=f"{ep+1}/{self.max_epochs}",
                             val_mse=f"{vloss:.4f}", best=f"{min(best, vloss):.4f}")
            if vloss < best - 1e-5:
                best, bad = vloss, 0
                best_state = {k: v.detach().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        pbar.close()
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if self.verbose:
            print(f"  [TF] fold完了: {ep+1}エポックで停止 (best_val={best:.4f})",
                  flush=True)
        return self

    def predict(self, X):
        dev = getattr(self, "_dev", _device())
        xn = torch.tensor(self._prep_num(X).astype(np.float32))
        xc = torch.tensor(self._encode_cat(X))
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(xn), 8192):
                o = self.model(xn[i:i+8192].to(dev), xc[i:i+8192].to(dev))
                outs.append(o.cpu().numpy())
        z = np.concatenate(outs)
        return z * self.y_std + self.y_mean
