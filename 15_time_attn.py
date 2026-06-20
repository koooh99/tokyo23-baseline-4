# 15_time_attn.py
# ------------------------------------------------------------
# ① 診断ドリブンな time-aware 時間 attention。
#
# §6d で時間 attention の重みはほぼ一様（≈1/8, L1=0.003, 最大0.126）に潰れ、③(GAT)で
# 空間 attention も等重みからほぼ逸脱しなかった。診断は「前方補完で隣接四半期の表現が
# 似すぎ softmax が discriminate できない」。本実験の問い:
#   一様化は前方補完の artifact か、データ内在の性質か。
# 前方補完を（実観測中心の注意で）直し、attention に区別材料（時間ギャップ・観測フラグ）を
# 与えたとき、重みが集中し R² が上がるなら artifact、依然一様なら内在的。
#
# 変えるのは時間集約器のみ:
#   GNN-temporal-TimeAttnGap = TimeAwareAttention(gap)：ギャップ埋め込みを key に加える加法注意
#   GNN-temporal-TimeAttnObs = TimeAwareAttention(obs)：上に加え前方補完ステップを強く減衰
# 比較基準 GRU / Attention(§6d) / 静的GCN / XGB-full を同一64フォールドで併記する。
# 空間conv(GCN)・L=8・hidden・層数・head・optimizer・lr・epoch・seed・rich物件特徴は
# GRU 実行と完全一致（assert_aggregator_only_diff で実行前に検証・明示）。
#
# 既定は直近8四半期で素早く確認（smoke）。GNN_FULL_EVAL=1 で全64四半期（正本・受け入れ基準）。
# 出力（FULL のとき canonical 名）:
#   outputs/results_long.csv         … 6モデル × scope の mean±std（§6dと同じ表に併記）
#   outputs/time_attn_summary.csv    … 重み分析（等重み1/Lからの逸脱・最大重み・q−1質量）を
#                                       overall / Central5 / Others で（§6dと同じ物差し）
# ------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
sys.path.append("src")
import config as C
import lib_compare as LC
from lib_gnn import assert_aggregator_only_diff

DEVICE = os.environ.get("GNN_DEVICE", C.GNN_DEVICE)
FULL_EVAL = bool(os.environ.get("GNN_FULL_EVAL"))
TEST_QUARTERS = C.EXPANDING_TEST_QUARTERS if FULL_EVAL else C.GNN_TEST_QUARTERS
L = C.TEMPORAL_SEQ_LEN
SUF = "" if FULL_EVAL else "_smoke"
SUB = {"Central5": C.CENTRAL_5,
       "Others": [w for w in C.WARDS_23 if w not in C.CENTRAL_5]}

print(f"=== 15. time-aware 時間Attention（①） (L={L}, device={DEVICE}, "
      f"{'全64四半期' if FULL_EVAL else '直近8四半期(smoke)'}={len(TEST_QUARTERS)}fold) ===")
df = pd.read_csv(C.DATA_DIR / "tokyo23_model_table_q.csv")
df = df.dropna(subset=[C.TARGET, "Qidx"]).copy()
df["Qidx"] = df["Qidx"].astype(int)

print("グラフ（町丁目queen隣接）を構築中...")
node_index, edge_index, N = LC.build_town_graph(df)
print(f"  ノード(町)数: {N}  辺数(有向): {edge_index.shape[1]}")

# --- フェア比較の生命線：集約器以外が GRU 実行と完全一致であることを実行前に検証・明示 ---
n_prop = len(C.GNN_PROP_FEATURES)
for agg in ("time_attn_gap", "time_attn_obs"):
    info = assert_aggregator_only_diff(n_node_feat=4, n_prop_feat=n_prop,
                                       hidden=C.GNN_HIDDEN, aggregator=agg)
    print(f"[集約器のみ差分の確認] GRU版 vs {agg}: g1/g2/head 形状が一致 "
          f"(head={info['head']})")
    print(f"   集約器パラメータ: GRU={info['agg_gru']}\n"
          f"                    {agg}={info['agg_alt']}")
print(f"  共有ハイパラ（全集約器で同一）: 空間conv=GCN, L={L}, hidden={C.GNN_HIDDEN}, "
      f"層数=2, optimizer=Adam, lr={C.TEMPORAL_LR}, epochs={C.TEMPORAL_EPOCHS}, "
      f"seed={C.RANDOM_STATE}, rich物件特徴={C.GNN_RICH_FEATURES}（変えるのは集約器のみ）")

# --- 重み収集フック：テスト四半期に出現する町ごとに1本の重みベクトル[N,L]を記録 -----
def make_sink(store):
    def sink(q, test_df, W):
        seen = {}
        for n, wd in zip(test_df["node"].to_numpy(), test_df["Municipality"].to_numpy()):
            if n not in seen:
                seen[n] = "Central5" if wd in C.CENTRAL_5 else "Others"
        for n, grp in seen.items():
            store.append((int(q), grp, *W[n]))
    return sink


weights = {"attention": [], "time_attn_gap": [], "time_attn_obs": []}

# --- 6モデルを同一フォールドで評価（attention系は重みも収集） ----------------------
summaries = []


def run_temporal(agg, tag, collect=False):
    sink = make_sink(weights[agg]) if collect else None
    s = LC.run_temporal_gnn(df, node_index, edge_index, N, TEST_QUARTERS, SUB,
                            device=DEVICE, aggregator=agg, rich=True, tag=tag,
                            weight_sink=sink)
    summaries.append(s)
    a = s[s["scope"] == "ALL"]
    if not a.empty:
        r = a.iloc[0]
        print(f"  全体 R² = {r['r2_mean']:.3f} ± {r['r2_std']:.3f} ({int(r['n_folds'])}fold)")
    return s


print("\n--- XGB-full(Qboth)（参照） ---", flush=True)
s = LC.run_xgb_full(df, TEST_QUARTERS, SUB, tag="XGB-full(Qboth)")
summaries.append(s)
print(f"  全体 R² = {s[s.scope=='ALL'].iloc[0]['r2_mean']:.3f}")

print("\n--- 静的GCN（参照） ---", flush=True)
s = LC.run_static_gnn(df, node_index, edge_index, N, TEST_QUARTERS, SUB,
                      device=DEVICE, conv="gcn", tag="GNN-static[GCN]")
summaries.append(s)
print(f"  全体 R² = {s[s.scope=='ALL'].iloc[0]['r2_mean']:.3f}")

print("\n--- GNN-temporal[GRU]（参照） ---", flush=True)
run_temporal("gru", "GNN-temporal[GRU]")
print("\n--- GNN-temporal[attention]（§6d・参照, 重み収集） ---", flush=True)
run_temporal("attention", "GNN-temporal[attention]", collect=True)
print("\n--- GNN-temporal-TimeAttnGap（①, 重み収集） ---", flush=True)
run_temporal("time_attn_gap", "GNN-temporal-TimeAttnGap", collect=True)
print("\n--- GNN-temporal-TimeAttnObs（①, 重み収集） ---", flush=True)
run_temporal("time_attn_obs", "GNN-temporal-TimeAttnObs", collect=True)

# --- results_long.csv（§6dと同じ表に新2変種＋参照4モデルを併記） ------------------
COLS = ["model", "scope", "name", "r2_mean", "r2_std", "r2_min", "r2_max", "n_folds"]
results = pd.concat(summaries, ignore_index=True)[COLS]
out_results = C.OUT_DIR / f"results_long{SUF}.csv"
results.to_csv(out_results, index=False, encoding="utf-8-sig")

# --- 重み分析（§6dと同じ物差し: 等重み1/Lからの逸脱・最大重み・q−1質量） -----------
wcols = [f"w{i}" for i in range(L)]
offsets = [L - i for i in range(L)]                  # index i → q−(L−i)。i=L-1 が q−1
labels = [f"q-{o}" for o in offsets]
uni = 1.0 / L
i_recent, i_yoy = L - 1, L - 4                        # q−1（直前） / q−4（前年同期）

agg_label = {"attention": "Attention(§6d)",
             "time_attn_gap": "TimeAttnGap(①)",
             "time_attn_obs": "TimeAttnObs(①)"}
rows = []
for agg in ("attention", "time_attn_gap", "time_attn_obs"):
    W = pd.DataFrame(weights[agg], columns=["Qidx", "group"] + wcols)
    if W.empty:
        continue
    scopes = {"overall": W, "Central5": W[W.group == "Central5"],
              "Others": W[W.group == "Others"]}
    for scope, Wg in scopes.items():
        if Wg.empty:
            continue
        p = Wg[wcols].mean().to_numpy()              # [L] 平均重みプロファイル
        rows.append({
            "aggregator": agg_label[agg], "scope": scope,
            "n_vectors": len(Wg),
            "l1_dev_from_uniform": float(np.abs(p - uni).sum()),
            "max_weight": float(p.max()),
            "max_at": labels[int(p.argmax())],
            "w_q1_recent": float(p[i_recent]),
            "w_q4_yoy": float(p[i_yoy]),
            # 一目で分かる判定（§6dの基準と整合）: 等重みからの逸脱が小さければ「一様」。
            "verdict": ("ほぼ一様" if np.abs(p - uni).sum() < 0.05
                        else ("直近(q−1)集中" if int(p.argmax()) == i_recent
                              else f"{labels[int(p.argmax())]}集中")),
        })
summary = pd.DataFrame(rows)
out_summary = C.OUT_DIR / f"time_attn_summary{SUF}.csv"
summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

# --- 表示 -----------------------------------------------------------------------
ORDER = ["XGB-full(Qboth)", "GNN-static[GCN]", "GNN-temporal[GRU]",
         "GNN-temporal[attention]", "GNN-temporal-TimeAttnGap",
         "GNN-temporal-TimeAttnObs"]
oi = {m: i for i, m in enumerate(ORDER)}
print(f"\n[全体R²（同一{len(TEST_QUARTERS)}四半期フォールド）]")
view = results[results.scope == "ALL"].copy()
view = view.sort_values("model", key=lambda s: s.map(oi))
print(view[["model", "r2_mean", "r2_std", "n_folds"]].round(3).to_string(index=False))

print("\n[サブグループ別 平均R²（Central5 / Others）]")
grp = results[results.scope == "GROUP"].copy()
grp["mo"] = grp["model"].map(oi)
grp = grp.sort_values(["name", "mo"])
print(grp[["model", "name", "r2_mean", "r2_std"]].round(3).to_string(index=False))

print(f"\n[時間Attention 重み分析（等重み 1/{L}={uni:.3f} が基準）]")
print(summary.round(4).to_string(index=False))

print(f"\n保存: {out_results}")
print(f"保存: {out_summary}")
print("\n完了")
