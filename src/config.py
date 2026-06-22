import os

# =========================
# Paths
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")

ITEM_FILE = os.path.join(DATA_DIR, "items.csv")
REVIEW_FILE = os.path.join(DATA_DIR, "reviews.csv")

PROCESSED_ITEM_FILE = os.path.join(DATA_DIR, "processed_items.csv")
PROCESSED_INTERACTION_FILE = os.path.join(DATA_DIR, "processed_interactions.csv")

CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
METRIC_DIR = os.path.join(RESULT_DIR, "metrics")
PLOT_DIR = os.path.join(RESULT_DIR, "plots")

# =========================
# Column names
# =========================
ITEM_ID_COL = "product_id"
USER_ID_COL = "user_id"

PRICE_COL = "price_num"

# 질문에 사용할 상품 속성
ATTRIBUTES = [
    "category2",
    "color",
    "fit",
    "material_mapped",
    "style_mapped",
    "price_bin",
]

# 추천 점수 계산용 popularity columns
POPULARITY_COLS = ["like_num", "review_num", "rating_num"]

# =========================
# Environment
# =========================
MAX_TURN = 10
TOP_K = 3

MIN_USER_INTERACTIONS = 2
MAX_ITEMS_FOR_TRAINING = None  # 빠른 실험용으로 5000 등으로 제한 가능. 전체 사용은 None.

# =========================
# Reward
# =========================
REWARD_RECOMMEND_SUCCESS = 1.0
REWARD_RECOMMEND_FAIL = -0.2
REWARD_ASK_SUCCESS = 0.01
REWARD_INVALID_ASK = -0.1
REWARD_CANDIDATE_EMPTY = -0.5
REWARD_MAX_TURN = -1.0

# =========================
# DQN
# =========================
SEED = 42

NUM_EPISODES = 1000
EVAL_EPISODES = 200

GAMMA = 0.95
LR = 1e-3
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000

TARGET_UPDATE_FREQ = 200
TRAIN_FREQ = 1

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 2000

HIDDEN_DIM = 128