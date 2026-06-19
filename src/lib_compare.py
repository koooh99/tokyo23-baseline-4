# src/lib_compare.py
# ------------------------------------------------------------
# run_compare.py 用の共通ランナー。これまで 09/10 に重複していた
# 「XGB参照ロジック」「静的GCNの学習ループ」「時系列GNNの学習ループ」を
# ここに集約し、全モデルを“同一の四半期Expanding Windowフォールド”で評価する。
#
# リーク無し評価の不変条件は 09/10 と同一:
#   - IQRトリミングは各フォールドの訓練のみ（lib_eval.trim_by_iqr / expanding_window_pooled）
#   - 標準化器・駅TE・OneHot は訓練フォールドのみで fit
#   - 時系列の系列は seq_steps=[q-L,…,q-1]（過去限定）
# 各ランナーは入口で乱数を固定（09/10 と同じ「seed once → fold loop」構造）し、
# 既知値をシード誤差内で再現する。
#
# 返り値はすべて summarize() 後の DataFrame に先頭 model 列を付けたもの
# （列: model, scope, name, r2_mean, r2_std, r2_min, r2_max, r2_trim_*, n_folds）。
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import xgboost as xgb

import config as C
from lib_eval import expanding_window_pooled, summarize, trim_by_iqr
from lib_features import StationTargetEncoder
from lib_spatial import load_town_polygons, build_neighbors
from lib_gnn import build_graph, SpatioGCN, Standardizer, train_eval_fold, \
    SpatioTemporalGNN


# --- グラフ（町丁目 queen 隣接）。09/10 と同一手順 -----------------
def build_town_graph(df):
    """df に出現する町の順序でノードを作り、queen隣接から edge_index を構築。
    返り値: (node_index dict, edge_index[2,E], N)。df には 'node' 列を付与する。"""
    node_keys = list(df.groupby(["Municipality", "DistrictName"]).groups.keys())
    towns = load_town_polygons(C.SHAPEFILE_PATH, C.WARDS_23)
    neighbors = build_neighbors(towns, method=C.SPATIAL_WEIGHT,
                                knn_k=C.KNN_K, projected_crs=C.PROJECTED_CRS)
    node_index, edge_index = build_graph(node_keys, neighbors)
    df["node"] = list(zip(df["Municipality"], df["DistrictName"]))
    df["node"] = df["node"].map(node_index)
    return node_index, edge_index, len(node_keys)


# --- XGB / OLS 参照（共通ヘルパ。09/10 の重複を一本化） -----------
def run_pooled(df, factory, feat_cols, tag, test_quarters, subgroups,
               use_te=True):
    """sklearn互換モデルを expanding_window_pooled で評価し summarize して返す。
    use_te=True のとき駅TEをフォールド内fitで付与する（feat_colsに"Station_TE"を含めること）。"""
    ft = StationTargetEncoder(C.TARGET, m=C.TE_SMOOTHING) if use_te else None
    rec, _ = expanding_window_pooled(
        df, factory, feat_cols, C.TARGET, test_quarters,
        iqr_trim=C.IQR_TRIM, iqr_k=C.IQR_K, iqr_by_ward=C.IQR_BY_WARD,
        subgroups=subgroups, fold_transform=ft, time_col="Qidx",
        progress=False)
    s = summarize(rec)
    s.insert(0, "model", tag)
    return s


def _xgb_pipeline(num_feats, onehot):
    steps = ([("oh", OneHotEncoder(handle_unknown="ignore"), onehot)] if onehot
             else []) + [("num", "passthrough", num_feats)]
    pre = ColumnTransformer(steps)
    return lambda: Pipeline([("pre", pre), ("reg", xgb.XGBRegressor(**C.XGB_PARAMS))])


def run_ols(df, test_quarters, subgroups, tag="OLS"):
    """線形ベースライン（XGB-full と同じ特徴: Base+Qboth+駅TE+区/用途one-hot）。"""
    onehot = ["Municipality"] + [c for c in C.CAT_FEATURES if c in df.columns]
    num = C.BASE_FEATURES + C.DEFAULT_LAG_COLS + ["Station_TE"]
    pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), onehot),
                             ("num", "passthrough", num)])
    factory = lambda: Pipeline([("pre", pre), ("reg", LinearRegression())])
    return run_pooled(df, factory, onehot + num, tag, test_quarters, subgroups,
                      use_te=True)


def run_xgb_full(df, test_quarters, subgroups, tag="XGB-full(Qboth)"):
    onehot = ["Municipality"] + [c for c in C.CAT_FEATURES if c in df.columns]
    num = C.BASE_FEATURES + C.DEFAULT_LAG_COLS + ["Station_TE"]
    return run_pooled(df, _xgb_pipeline(num, onehot), onehot + num, tag,
                      test_quarters, subgroups, use_te=True)


def run_xgb_propT(df, test_quarters, subgroups, tag="XGB-propT(自町T-lagのみ)"):
    """物件特徴 + 自町T-lag のみ（S-lag・駅TE・one-hot無し＝GNNと同程度の情報）。"""
    num = C.GNN_PROP_FEATURES + C.DEFAULT_TLAG_COLS
    return run_pooled(df, _xgb_pipeline(num, []), num, tag, test_quarters,
                      subgroups, use_te=False)


# --- 静的GCN（09 の学習ループを抽出） -----------------------------
def run_static_gnn(df, node_index, edge_index, N, test_quarters, subgroups,
                   device="cpu", tag="GNN(min v0)"):
    """各四半期スナップショットに2層GCN → 物件特徴と結合して回帰（09と同一）。"""
    torch.manual_seed(C.RANDOM_STATE); np.random.seed(C.RANDOM_STATE)
    edge_index = edge_index.to(device)
    LAGF = ["Lag_q1_AvgPrice", "Lag_q4_AvgPrice"]
    PROP = C.GNN_PROP_FEATURES
    cent = df.groupby("node")[["centroid_x", "centroid_y"]].first().reindex(range(N))
    C_arr = cent.to_numpy(dtype=float)
    tq = df.groupby(["node", "Qidx"])[LAGF].first().reset_index()
    lag_by_q = {}
    for q, g in tq.groupby("Qidx"):
        m = np.full((N, 2), np.nan)
        m[g["node"].to_numpy()] = g[LAGF].to_numpy(dtype=float)
        lag_by_q[int(q)] = m

    def node_matrix(q):
        return np.concatenate([lag_by_q[q], C_arr], axis=1)

    records = []
    for q in test_quarters:
        if q not in lag_by_q:
            continue
        win = C.GNN_TRAIN_WINDOW
        lo = q - win if win else -10**9
        train_q = [t for t in sorted(lag_by_q) if lo <= t < q and (df["Qidx"] == t).any()]
        test_df = df[df["Qidx"] == q]
        if len(train_q) < 2 or len(test_df) < 50:
            continue
        train_df = df[df["Qidx"].isin(train_q)]
        if C.IQR_TRIM:
            train_df, _ = trim_by_iqr(train_df, C.TARGET, C.IQR_K, C.IQR_BY_WARD)
        node_std = Standardizer().fit(np.vstack([node_matrix(t) for t in train_q]))
        prop_std = Standardizer().fit(train_df[PROP].to_numpy(dtype=float))
        y_mean = train_df[C.TARGET].mean(); y_std = train_df[C.TARGET].std() or 1.0

        def make_snap(rows, qq):
            return dict(x=node_std.transform(node_matrix(qq)),
                        node_idx=rows["node"].to_numpy(),
                        prop=prop_std.transform(rows[PROP].to_numpy(dtype=float)),
                        y=((rows[C.TARGET].to_numpy() - y_mean) / y_std))
        snapshots = {t: make_snap(train_df[train_df["Qidx"] == t], t) for t in train_q}
        snapshots[q] = make_snap(test_df, q)
        model = SpatioGCN(n_node_feat=4, n_prop_feat=len(PROP), hidden=C.GNN_HIDDEN)
        pred = train_eval_fold(model, snapshots, edge_index, train_q, q,
                               epochs=C.GNN_EPOCHS, lr=C.GNN_LR, device=device)
        pred = pred * y_std + y_mean
        _record(records, test_df.assign(pred=pred), q, subgroups)
    s = summarize(pd.DataFrame(records)); s.insert(0, "model", tag)
    return s


# --- 時系列GNN（10 の学習ループを抽出。集約器を差し替え可能に） ----
def run_temporal_gnn(df, node_index, edge_index, N, test_quarters, subgroups,
                     device="cpu", aggregator="gru", rich=True, tag=None,
                     weight_sink=None):
    """各町の直近L四半期の観測平米単価系列を GCN→時間集約器 で畳み回帰（10と同一）。
    aggregator: lib_temporal の集約器指定（"gru"/"attention"/"mean"/"last"）。
    rich: 立地系特徴(駅TE+区/用途one-hot)を物件ヘッドに足すか（XGB-fullと同条件）。
    weight_sink: 任意。注意集約など重みを持つ集約器のとき、各テスト四半期の評価後に
        weight_sink(q, test_df, weights[N,L]) を呼ぶ（既定None＝挙動不変・数値不変）。"""
    torch.manual_seed(C.RANDOM_STATE); np.random.seed(C.RANDOM_STATE)
    edge_index = edge_index.to(device)
    L = C.TEMPORAL_SEQ_LEN
    PROP = C.GNN_PROP_FEATURES
    if tag is None:
        tag = "GNN-temporal+feat" if rich else f"GNN-temporal({aggregator})"

    cent = df.groupby("node")[["centroid_x", "centroid_y"]].first().reindex(range(N))
    C_arr = cent.to_numpy(dtype=float)
    price = df.groupby(["node", "Qidx"])[C.TARGET].mean().reset_index()
    qmin, qmax = int(price["Qidx"].min()), int(price["Qidx"].max())
    all_q = list(range(qmin, qmax + 1))
    qpos = {q: i for i, q in enumerate(all_q)}
    P = np.full((len(all_q), N), np.nan)
    for r in price.itertuples(index=False):
        P[qpos[int(r.Qidx)], r.node] = getattr(r, C.TARGET)
    OBS = (~np.isnan(P)).astype(float)
    Pf = P.copy()
    for t in range(1, len(all_q)):
        m = np.isnan(Pf[t]); Pf[t, m] = Pf[t - 1, m]

    def raw_node_feat(q):
        i = qpos[q]
        return np.column_stack([Pf[i], OBS[i], C_arr])

    def seq_steps(t):
        return [t - k for k in range(L, 0, -1)]              # [t-L,…,t-1]（過去のみ）

    def train_eval(train_t, test_q, train_df, test_df):
        need_ts = sorted({s for t in (train_t + [test_q]) for s in seq_steps(t)}
                         | set(train_t) | {test_q})
        need_ts = [t for t in need_ts if qmin <= t <= qmax]
        node_std = Standardizer().fit(np.vstack([raw_node_feat(t) for t in train_t]))
        y_mean = train_df[C.TARGET].mean(); y_std = train_df[C.TARGET].std() or 1.0
        num_cols = list(PROP); cat_cols, ohe = [], None
        if rich:
            if C.STATION_TE:
                ste = StationTargetEncoder(C.TARGET, m=C.TE_SMOOTHING).fit(train_df)
                train_df = ste.transform(train_df); test_df = ste.transform(test_df)
                num_cols = num_cols + ["Station_TE"]
            cat_cols = ["Municipality"] + [c for c in C.CAT_FEATURES if c in train_df.columns]
            ohe = OneHotEncoder(handle_unknown="ignore").fit(train_df[cat_cols])
        prop_std = Standardizer().fit(train_df[num_cols].to_numpy(dtype=float))
        cat_dim = sum(len(c) for c in ohe.categories_) if ohe is not None else 0
        Xstd = {t: torch.tensor(node_std.transform(raw_node_feat(t)),
                                dtype=torch.float32, device=device) for t in need_ts}

        def prop_pack(rows):
            num = prop_std.transform(rows[num_cols].to_numpy(dtype=float))
            if ohe is not None:
                cat = ohe.transform(rows[cat_cols])
                cat = cat.toarray() if hasattr(cat, "toarray") else cat
                num = np.concatenate([num, cat], axis=1)
            return (torch.tensor(rows["node"].to_numpy(), dtype=torch.long, device=device),
                    torch.tensor(num, dtype=torch.float32, device=device),
                    torch.tensor((rows[C.TARGET].to_numpy() - y_mean) / y_std,
                                 dtype=torch.float32, device=device))
        train_packs = {t: prop_pack(train_df[train_df["Qidx"] == t]) for t in train_t}
        test_pack = prop_pack(test_df)
        model = SpatioTemporalGNN(n_node_feat=4, n_prop_feat=len(num_cols) + cat_dim,
                                  hidden=C.GNN_HIDDEN, aggregator=aggregator).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=C.TEMPORAL_LR)
        loss_fn = torch.nn.MSELoss()
        model.train()
        for _ in range(C.TEMPORAL_EPOCHS):
            for t in train_t:
                opt.zero_grad()
                H = {s: model.encode(Xstd[s], edge_index) for s in seq_steps(t)}
                seq = torch.stack([H[s] for s in seq_steps(t)], dim=0)
                h = model.temporal(seq)
                idx, pf, y = train_packs[t]
                loss = loss_fn(model.predict(h, idx, pf), y)
                loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            H = {t: model.encode(Xstd[t], edge_index) for t in seq_steps(test_q)}
            seq = torch.stack([H[s] for s in seq_steps(test_q)], dim=0)
            h = model.temporal(seq)
            idx, pf, _ = test_pack
            pred = model.predict(h, idx, pf).cpu().numpy() * y_std + y_mean
        w = None
        if getattr(model.aggregator, "returns_weights", False):
            lw = model.aggregator.last_weights      # [N,L]（全ノード分）
            w = lw.cpu().numpy() if lw is not None else None
        return pred, w

    records = []
    for q in test_quarters:
        win = C.GNN_TRAIN_WINDOW
        lo = q - win if win else qmin
        train_t = [t for t in all_q if lo <= t < q and (df["Qidx"] == t).any()
                   and (t - L) >= qmin]
        test_df = df[df["Qidx"] == q]
        if len(train_t) < 2 or len(test_df) < 50:
            continue
        train_df = df[df["Qidx"].isin(train_t)]
        if C.IQR_TRIM:
            train_df, _ = trim_by_iqr(train_df, C.TARGET, C.IQR_K, C.IQR_BY_WARD)
        pred, weights = train_eval(train_t, q, train_df, test_df)
        if weight_sink is not None and weights is not None:
            weight_sink(q, test_df, weights)
        _record(records, test_df.assign(pred=pred), q, subgroups)
    s = summarize(pd.DataFrame(records)); s.insert(0, "model", tag)
    return s


def _record(records, te, q, subgroups):
    """全体・サブグループ別のR²を records に積む（09/10 共通の集計）。"""
    records.append({"test_year": q, "scope": "ALL", "name": "Tokyo23",
                    "n": len(te), "r2": r2_score(te[C.TARGET], te["pred"]),
                    "r2_trim": np.nan})
    for label, wl in subgroups.items():
        g = te[te["Municipality"].isin(wl)]
        if len(g) >= 10:
            records.append({"test_year": q, "scope": "GROUP", "name": label,
                            "n": len(g), "r2": r2_score(g[C.TARGET], g["pred"]),
                            "r2_trim": np.nan})
