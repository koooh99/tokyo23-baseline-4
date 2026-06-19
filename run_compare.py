# run_compare.py
# ------------------------------------------------------------
# 全モデルを「同一の四半期Expanding Windowフォールド」で評価し、tidy な1表
# outputs/results_long.csv（列: model, scope, name, r2_mean, r2_std,
# r2_min, r2_max, n_folds）に集約する統合ハーネス。
#
# これまで 09/10 はモデルごとに別スクリプト・別CSVで結果が分裂していた。
# ここで OLS / XGB-propT / XGB-full / GNN-min / GNN-temporal[GRU] /
# GNN-temporal[attention] を同じフォールド・同じ前処理規約で1表にまとめる。
# 学習・評価ロジックは src/lib_compare.py に集約（XGB参照の重複も一本化）。
#
# 既定は直近8四半期(C.GNN_TEST_QUARTERS)で素早く確認。GNN_FULL_EVAL=1 で
# 全64四半期(2010Q1〜2025Q4)を評価し、既知値の再現を確認する。
# FT-Transformer は別プロトコル(07, 年次2018-2025)のため本表には含めない。
# ------------------------------------------------------------
import os
import sys
import pandas as pd
sys.path.append("src")
import config as C
import lib_compare as LC

DEVICE = os.environ.get("GNN_DEVICE", C.GNN_DEVICE)
FULL_EVAL = bool(os.environ.get("GNN_FULL_EVAL"))
TEST_QUARTERS = C.EXPANDING_TEST_QUARTERS if FULL_EVAL else C.GNN_TEST_QUARTERS
OUT_NAME = "results_long_full.csv" if FULL_EVAL else "results_long.csv"

# 評価対象モデル（物語の順）。各値は lib_compare のランナー呼び出し。
# 比較用に attention 集約版も同フォールドに載せる（次段の重み可視化の足場）。
MODELS = [
    ("OLS",                  lambda df, g: LC.run_ols(df, TEST_QUARTERS, SUB)),
    ("XGB-propT",            lambda df, g: LC.run_xgb_propT(df, TEST_QUARTERS, SUB)),
    ("XGB-full",             lambda df, g: LC.run_xgb_full(df, TEST_QUARTERS, SUB)),
    ("GNN-min",              lambda df, g: LC.run_static_gnn(
        df, *g, TEST_QUARTERS, SUB, device=DEVICE)),
    ("GNN-temporal[GRU]",    lambda df, g: LC.run_temporal_gnn(
        df, *g, TEST_QUARTERS, SUB, device=DEVICE, aggregator="gru", rich=True,
        tag="GNN-temporal[GRU]")),
    ("GNN-temporal[attn]",   lambda df, g: LC.run_temporal_gnn(
        df, *g, TEST_QUARTERS, SUB, device=DEVICE, aggregator="attention",
        rich=True, tag="GNN-temporal[attention]")),
]

SUB = {"Central5": C.CENTRAL_5,
       "Others": [w for w in C.WARDS_23 if w not in C.CENTRAL_5]}

print(f"=== 比較ハーネス（同一{len(TEST_QUARTERS)}四半期フォールド, device={DEVICE}, "
      f"{'全64四半期' if FULL_EVAL else '直近8四半期'}） ===")
df = pd.read_csv(C.DATA_DIR / "tokyo23_model_table_q.csv")
df = df.dropna(subset=[C.TARGET, "Qidx"]).copy()
df["Qidx"] = df["Qidx"].astype(int)

# グラフは一度だけ構築して全GNNで共有（df に 'node' 列が付く）。
print("グラフ（町丁目queen隣接）を構築中...")
node_index, edge_index, N = LC.build_town_graph(df)
graph = (node_index, edge_index, N)
print(f"  ノード(町)数: {N}  辺数(有向): {edge_index.shape[1]}")

summaries = []
for name, run in MODELS:
    print(f"\n--- {name} を {len(TEST_QUARTERS)}四半期フォールドで評価中 ---", flush=True)
    s = run(df, graph)
    summaries.append(s)
    a = s[s["scope"] == "ALL"]
    if not a.empty:
        r = a.iloc[0]
        print(f"  全体 R² = {r['r2_mean']:.3f} ± {r['r2_std']:.3f} "
              f"({int(r['n_folds'])}fold)", flush=True)

COLS = ["model", "scope", "name", "r2_mean", "r2_std", "r2_min", "r2_max", "n_folds"]
results = pd.concat(summaries, ignore_index=True)[COLS]
out = C.OUT_DIR / OUT_NAME
results.to_csv(out, index=False, encoding="utf-8-sig")

# モデルの並びを物語順に固定して表示。
ORDER = [s[s["scope"] == "ALL"]["model"].iloc[0] for s in summaries
         if not s[s["scope"] == "ALL"].empty]
order_idx = {m: i for i, m in enumerate(ORDER)}
print(f"\n[全体R²（同一フォールド: {len(TEST_QUARTERS)}四半期）]")
view = results[results["scope"] == "ALL"].copy()
view = view.sort_values("model", key=lambda s: s.map(order_idx))
print(view.round(3).to_string(index=False))
print("\n[サブグループ別 平均R²]")
grp = results[results["scope"] == "GROUP"].copy()
grp = grp.sort_values(["name", "model"], key=lambda s: s.map(order_idx)
                      if s.name == "model" else s)
print(grp[["model", "name", "r2_mean", "r2_std"]].round(3).to_string(index=False))
print(f"\n保存: {out}")
