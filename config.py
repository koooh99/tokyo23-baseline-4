# config.py
# ------------------------------------------------------------
# プロジェクト全体の設定を一元管理する。
# パスや区リスト・特徴量定義をスクリプトに散らさないことで、
# 「どのモードで・どの特徴量で・どのデータで」回したかを論文に明示しやすくする。
# ------------------------------------------------------------
from pathlib import Path

# --- ディレクトリ ---------------------------------------------------
# 環境依存の絶対パス（旧 05_s_lag.py の /Users/koyasato/... のような）は
# ここ以外に書かない。各自の環境ではこの3つだけ直せば動くようにする。
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # 生成・中間CSVの置き場
RAW_DIR = DATA_DIR / "raw"        # APIから取得した素データ
SHAPE_DIR = DATA_DIR / "shape"    # 国勢調査町丁目shapefile等
OUT_DIR = ROOT / "outputs"        # 図表・結果CSV
for d in (DATA_DIR, RAW_DIR, SHAPE_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# データ取得元：不動産情報ライブラリのサイトから直接DLしたCSV（日本語列）。
# APIでは最寄駅情報が取れないため、サイト版CSVを正本とする。
# 各自の環境ではこのCSVをここに置く。
SOURCE_CSV = RAW_DIR / "Tokyo_20053_20254.csv"
SOURCE_ENCODING = "cp932"   # サイト版CSVはShift_JIS(cp932)

# S-lag・重心の計算には引き続き町丁目shapefileが必要。
SHAPEFILE_PATH = SHAPE_DIR / "r2ka13.shp"        # 令和2年 国勢調査 東京都 町丁目
# 駅距離はサイト版CSVの「最寄駅：距離（分）」を使うので、駅shapefileは不要になった。

# --- 区リスト -------------------------------------------------------
WARDS_23 = [
    "千代田区", "中央区", "港区", "新宿区", "文京区", "台東区", "墨田区",
    "江東区", "品川区", "目黒区", "大田区", "世田谷区", "渋谷区", "中野区",
    "杉並区", "豊島区", "北区", "荒川区", "板橋区", "練馬区", "足立区",
    "葛飾区", "江戸川区",
]

# 都心5区（議事録：サブグループ分析）。標準的な定義を採用。
CENTRAL_5 = ["千代田区", "中央区", "港区", "新宿区", "渋谷区"]

WARD_EN = {
    "千代田区": "Chiyoda", "中央区": "Chuo", "港区": "Minato", "新宿区": "Shinjuku",
    "文京区": "Bunkyo", "台東区": "Taito", "墨田区": "Sumida", "江東区": "Koto",
    "品川区": "Shinagawa", "目黒区": "Meguro", "大田区": "Ota", "世田谷区": "Setagaya",
    "渋谷区": "Shibuya", "中野区": "Nakano", "杉並区": "Suginami", "豊島区": "Toshima",
    "北区": "Kita", "荒川区": "Arakawa", "板橋区": "Itabashi", "練馬区": "Nerima",
    "足立区": "Adachi", "葛飾区": "Katsushika", "江戸川区": "Edogawa",
}

# --- データ取得 -----------------------------------------------------
# サイト版CSVは「取引時期(2021年第1四半期)」から年を抽出する。データは2005-2025年。
TARGET_TYPE = "中古マンション等"
TARGET_USE = "住宅"

# サイト版CSV(日本語列) → パイプライン内部で使う英語名 への対応
COLMAP = {
    "種類": "Type",
    "市区町村名": "Municipality",
    "地区名": "DistrictName",
    "最寄駅：名称": "StationName",
    "最寄駅：距離（分）": "StationMinutes_raw",
    "取引価格（総額）": "TradePrice",
    "間取り": "FloorPlan",
    "面積（㎡）": "Area",
    "建築年": "BuildingYear",
    "建物の構造": "Structure",
    "用途": "Use",
    "建ぺい率（％）": "CoverageRatio",
    "容積率（％）": "FloorAreaRatio",
    "取引時期": "Period",
    "都市計画": "Zoning",
    "改装": "Renovation",
}
# 最寄駅距離の範囲表記 → 代表値(分)
STATION_MIN_RANGE = {"30分～60分": 45, "1H～1H30": 75, "1H30～2H": 105, "2H～": 120}

# --- 目的変数・特徴量 -----------------------------------------------
TARGET = "UnitPrice_Sqm_ManYen"   # 平米単価（万円/㎡）

# 物件固有の特徴量（駅距離=徒歩分はサイト版CSVから物件単位で得られる）
BASE_FEATURES = ["Age", "Area_num", "Is_Renovated", "Is_RC", "Rooms",
                 "FAR_CAR_ratio", "Station_min"]

# one-hot するカテゴリ特徴量。Zoning(都市計画/用途地域)を追加。
# Municipality は従来どおり別枠で one-hot。
CAT_FEATURES = ["Zoning"]

# 最寄駅名称のターゲットエンコーディング（リーク防止のため訓練フォールド内で符号化）
STATION_TE = True
TE_SMOOTHING = 50     # 平滑化: (sum + m*全体平均) / (count + m)。少数駅を全体平均に寄せる

# 空間特徴量（町単位、04で付与）。駅距離は物件側に移したので重心座標のみ。
SPATIAL_FEATURES = ["centroid_x", "centroid_y"]

# 時間ラグ・空間ラグ（ラグ次数ごとにモデルを比較する）
LAG_CONFIGS = {
    "Lag1": ["Lag1_AvgPrice", "S_Lag1_AvgPrice"],
    "Lag2": ["Lag2_AvgPrice", "S_Lag2_AvgPrice"],
    "Lag3": ["Lag3_AvgPrice", "S_Lag3_AvgPrice"],
}

# --- 四半期ラグ（年次との比較用。年次の設定 LAG_CONFIGS は上に残す） --------
# 取引時期 '2025年第1四半期' から四半期通し番号 Qidx = Year*4 + (Quarter-1) を作り、
# T-lag を2種類で取る：直前1四半期(Qidx-1) と 前年同期(=4Q前, Qidx-4)。
# S-lag も同じ四半期インデックスで近隣平均を取る（接尾辞 S_）。
# 03q/04q/05q が使う。年次パイプライン(03/04/05)とは出力ファイルを分けてあるので
# 既存の結果(tokyo23_with_tlag.csv / _model_table.csv / baseline_summary.csv)は消えない。
LAG_CONFIGS_Q = {
    "Qprev": ["Lag_q1_AvgPrice", "S_Lag_q1_AvgPrice"],            # 直前1Qのみ
    "Qyoy":  ["Lag_q4_AvgPrice", "S_Lag_q4_AvgPrice"],            # 前年同期(4Q前)のみ
    "Qboth": ["Lag_q1_AvgPrice", "Lag_q4_AvgPrice",
              "S_Lag_q1_AvgPrice", "S_Lag_q4_AvgPrice"],          # 両方
}
# Expanding Window を四半期フォールドで回す検証対象（2010Q1〜2025Q4）。
# 各 Qidx について「< Qidx を訓練 / == Qidx をテスト」で1モデルずつ学習する。
EXPANDING_TEST_QUARTERS = [y * 4 + q for y in range(2010, 2026) for q in range(4)]

# 既定のラグ構成を「四半期 Qboth（直前1Q + 前年同期、それぞれ T-lag/S-lag）」に確定。
# 05q の比較で Qboth が最良だったため、08q アブレーションや時空間GNN等の下流はこれを参照する。
# 内訳: 直前1Q=Lag_q1 / 前年同期(4Q前)=Lag_q4 と、その空間ラグ S_。
DEFAULT_LAG_NAME = "Qboth"
DEFAULT_LAG_COLS = LAG_CONFIGS_Q[DEFAULT_LAG_NAME]
DEFAULT_TLAG_COLS = ["Lag_q1_AvgPrice", "Lag_q4_AvgPrice"]      # 自町の過去（時間ラグ）
DEFAULT_SLAG_COLS = ["S_Lag_q1_AvgPrice", "S_Lag_q4_AvgPrice"]  # 近隣の過去（空間ラグ）

# --- 時空間GNN（09, 最小スタート） ---------------------------------
# 方針: 04qの町丁目queen隣接(940町)をグラフ化し、各四半期スナップショットに
#       空間GCNを適用 → 学習した空間表現を物件特徴と結合して平米単価を回帰する。
#       手作りS-lag（近隣平均）を「グラフ伝播」に置き換えて勝てるかを見る v0。
# 時間方向は当面 Expanding Window(四半期) で表現し、直近性重視の系列モデルは次段で足す。
# ノード特徴は町丁目×四半期で持つ自町ラグ（直前1Q/前年同期）＋重心座標。
GNN_NODE_FEATURES = ["Lag_q1_AvgPrice", "Lag_q4_AvgPrice", "centroid_x", "centroid_y"]
# 物件側の数値特徴（駅TEは最小版では使わず、グラフの寄与を切り分ける）。
GNN_PROP_FEATURES = ["Age", "Area_num", "Is_Renovated", "Is_RC", "Rooms",
                     "FAR_CAR_ratio", "Station_min"]
# CPUで現実的に回すため、まず直近8四半期(2024Q1〜2025Q4)だけで検証。
GNN_TEST_QUARTERS = [y * 4 + q for y in range(2024, 2026) for q in range(4)]
GNN_TRAIN_WINDOW = 12         # 各フォールドで使う直近の訓練四半期数（None=全期間）
GNN_HIDDEN = 32               # GCN隠れ次元
GNN_EPOCHS = 40
GNN_LR = 1e-3
GNN_DEVICE = "cpu"            # MPSはハングし得るためCPU既定（環境変数 GNN_DEVICE で上書き）

# 空間conv層のセレクタ（TEMPORAL_AGG と対になる空間版, lib_gnn.SPATIAL_CONVS のキー）。
# "gcn"=対称正規化の等重み平均（既定・既存挙動を一切変えない）、
# "gat"=GATConv(static attention)、"gatv2"=GATv2Conv(dynamic attention, 推奨)。
# 09/run_compare は環境変数 SPATIAL_CONV で上書き可。GATは同じedge_index上で辺重みのみ学習する。
SPATIAL_CONV = "gcn"
# GAT系1層目のヘッド数。1層目 heads=GAT_HEADS concat=True → hidden*heads、
# 2層目 heads=1 concat=False → hidden（出力次元をGCN版の hidden に一致させる）。
GAT_HEADS = 4

# --- 時空間GNN・時間系列版（10, GCN+GRU） --------------------------
# 各町の「直近LQの観測平米単価系列」を GCN→GRU で集約し、直近性を学習させる。
# ノード特徴は観測平米単価(前方補完)＋観測フラグ＋重心。系列長は直近性/季節性を
# 両方カバーするよう既定8四半期(=2年, 前年同期も含む)。
TEMPORAL_SEQ_LEN = 8          # 時間集約に入れる直近四半期数
TEMPORAL_EPOCHS = 30          # 四半期ごとにstepする（=エポックあたり訓練四半期数の更新）
TEMPORAL_LR = 3e-3
# 時間集約器（lib_temporal.AGGREGATORS のキー）。"gru" が既定挙動（既知値再現）。
# "attention"=過去Lステップへのsoftmax重み学習（重み[N,L]可視化可）、
# "mean"/"last"=アブレーション用の単純基準。10は環境変数 TEMPORAL_AGG で上書き可。
# --- time-aware 注意（①, 15が使う。§6dの一様化が前方補完の artifact か内在かを切り分ける）---
# "time_attn_gap"=学習した時間ギャップ埋め込みを key に加える加法注意（全Lに注意・
#   直近を区別する自由度を付与）。"time_attn_obs"=上に加え前方補完(非観測)ステップを
#   強く減衰し実観測中心に注意を張る（不等間隔系列扱い）。どちらも last_weights[N,L] を
#   §6d と同形式で出す。既定 TEMPORAL_AGG は変更しない（"gru" のまま）。
TEMPORAL_AGG = "gru"
# (c) 公平比較: XGB-full(Qboth) と同じ立地系特徴を物件側ヘッドにも入れる。
# Station_TE（駅名ターゲットエンコーディング, フォールド内fitでリーク防止）＋
# Municipality / Zoning の one-hot を結合する。Falseで最小版（グラフ＋物件数値のみ）。
GNN_RICH_FEATURES = True

# --- 外れ値処理（議事録：慎重に・論文で明示） ----------------------
# 2段階に分ける：
#  (A) 決定論的クレンジング … 02で実施。価格0や面積欠損など「あり得ない/計算不能」を除去。
#  (B) 統計的トリミング(IQR) … 05のモデル内で、訓練フォールドの分布からのみ閾値を決める。
#      テストデータの情報を使わない＝リークしない。論文には (B) を適用した旨と係数を明記する。
IQR_TRIM = True       # 統計的トリミングを使うか
IQR_K = 1.5           # IQR の何倍を閾値にするか
IQR_BY_WARD = True    # 区ごとに閾値を決めるか（都心と周辺で価格帯が違うため推奨）

# --- 空間ラグの作り方（議事録：行政区隣接より精緻に） --------------
# "queen"     : 全23区の町丁目に対して一括でクイーン隣接（区をまたぐ）
# "knn"        : 重心からの k 近傍（行政界に依存しない）
SPATIAL_WEIGHT = "queen"
KNN_K = 6

# 距離計算用の投影座標系（東京＝平面直角座標系 第IX系 / JGD2011）
PROJECTED_CRS = "EPSG:6677"

# --- モデル ---------------------------------------------------------
XGB_PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)
EXPANDING_TEST_YEARS = list(range(2010, 2026))  # 2010-2025 を順に検証年に
RANDOM_STATE = 42
