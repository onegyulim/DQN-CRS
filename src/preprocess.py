import os
import ast
import random
import numpy as np
import pandas as pd

from config import (
    ITEM_FILE,
    REVIEW_FILE,
    PROCESSED_ITEM_FILE,
    PROCESSED_INTERACTION_FILE,
    ITEM_ID_COL,
    USER_ID_COL,
    PRICE_COL,
    ATTRIBUTES,
    POPULARITY_COLS,
    MIN_USER_INTERACTIONS,
    MAX_ITEMS_FOR_TRAINING,
    DATA_DIR,
    RESULT_DIR,
    CHECKPOINT_DIR,
    METRIC_DIR,
    PLOT_DIR,
    SEED,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs():
    for path in [DATA_DIR, RESULT_DIR, CHECKPOINT_DIR, METRIC_DIR, PLOT_DIR]:
        os.makedirs(path, exist_ok=True)


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def clean_text_value(x):
    if pd.isna(x):
        return "unknown"
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return "unknown"
    return x


def parse_multi_label_first(x):
    """
    style_mapped, material_mapped처럼 "minimal, sporty" 형태면 첫 번째 값을 대표값으로 사용.
    DQN 실험 초기 버전에서는 state/action space를 줄이기 위해 대표값만 사용.
    """
    if pd.isna(x):
        return "unknown"
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return "unknown"
    parts = [p.strip() for p in x.split(",") if p.strip()]
    return parts[0] if parts else "unknown"


def make_price_bin(items: pd.DataFrame) -> pd.DataFrame:
    if PRICE_COL not in items.columns:
        raise ValueError(f"'{PRICE_COL}' column is required in items.csv")

    items[PRICE_COL] = pd.to_numeric(items[PRICE_COL], errors="coerce")
    items = items.dropna(subset=[PRICE_COL]).copy()

    # 분위수 기반 5구간. 중복 edge 문제를 피하기 위해 duplicates='drop'
    items["price_bin"] = pd.qcut(
        items[PRICE_COL],
        q=5,
        labels=False,
        duplicates="drop",
    )

    items["price_bin"] = items["price_bin"].fillna(0).astype(int).astype(str)
    items["price_bin"] = "price_bin_" + items["price_bin"]
    return items


def normalize_popularity(items: pd.DataFrame) -> pd.DataFrame:
    for col in POPULARITY_COLS:
        if col not in items.columns:
            items[col] = 0.0
        items[col] = pd.to_numeric(items[col], errors="coerce").fillna(0.0)

    for col in POPULARITY_COLS:
        max_val = items[col].max()
        min_val = items[col].min()
        if max_val > min_val:
            items[f"{col}_norm"] = (items[col] - min_val) / (max_val - min_val)
        else:
            items[f"{col}_norm"] = 0.0

    items["popularity_score"] = (
        0.5 * items["like_num_norm"]
        + 0.3 * items["review_num_norm"]
        + 0.2 * items["rating_num_norm"]
    )
    return items


def preprocess_items(items: pd.DataFrame) -> pd.DataFrame:
    if ITEM_ID_COL not in items.columns:
        raise ValueError(f"'{ITEM_ID_COL}' column is required in items.csv")

    items = items.copy()
    items[ITEM_ID_COL] = items[ITEM_ID_COL].astype(str)

    items = make_price_bin(items)

    # 사용할 속성 정리
    for col in ATTRIBUTES:
        if col not in items.columns:
            raise ValueError(f"Attribute column '{col}' is missing in items.csv")

    for col in ATTRIBUTES:
        if col in ["style_mapped", "material_mapped"]:
            items[col] = items[col].apply(parse_multi_label_first)
        else:
            items[col] = items[col].apply(clean_text_value)

    items = normalize_popularity(items)

    # 학습용으로 속성이 모두 존재하는 상품만 사용
    for col in ATTRIBUTES:
        items = items[items[col] != "unknown"]

    items = items.drop_duplicates(subset=[ITEM_ID_COL]).reset_index(drop=True)

    if MAX_ITEMS_FOR_TRAINING is not None:
        items = (
            items.sort_values("popularity_score", ascending=False)
            .head(MAX_ITEMS_FOR_TRAINING)
            .reset_index(drop=True)
        )

    return items


def preprocess_interactions(reviews: pd.DataFrame, valid_item_ids: set) -> pd.DataFrame:
    required_cols = [ITEM_ID_COL, USER_ID_COL]
    for col in required_cols:
        if col not in reviews.columns:
            raise ValueError(f"'{col}' column is required in reviews.csv")

    interactions = reviews[[USER_ID_COL, ITEM_ID_COL]].copy()
    interactions[USER_ID_COL] = interactions[USER_ID_COL].astype(str)
    interactions[ITEM_ID_COL] = interactions[ITEM_ID_COL].astype(str)

    interactions = interactions[interactions[ITEM_ID_COL].isin(valid_item_ids)]
    interactions = interactions.drop_duplicates()

    user_counts = interactions.groupby(USER_ID_COL)[ITEM_ID_COL].nunique()
    valid_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
    interactions = interactions[interactions[USER_ID_COL].isin(valid_users)]

    interactions = interactions.reset_index(drop=True)
    return interactions


def main():
    set_seed(SEED)
    ensure_dirs()

    print("[INFO] Loading raw files...")
    items = safe_read_csv(ITEM_FILE)
    reviews = safe_read_csv(REVIEW_FILE)

    print("[INFO] Preprocessing items...")
    items = preprocess_items(items)
    valid_item_ids = set(items[ITEM_ID_COL].astype(str).tolist())

    print("[INFO] Preprocessing interactions...")
    interactions = preprocess_interactions(reviews, valid_item_ids)

    print("[INFO] Saving processed files...")
    items.to_csv(PROCESSED_ITEM_FILE, index=False)
    interactions.to_csv(PROCESSED_INTERACTION_FILE, index=False)

    print(f"[DONE] processed_items: {len(items):,}")
    print(f"[DONE] processed_interactions: {len(interactions):,}")
    print(f"[DONE] users: {interactions[USER_ID_COL].nunique():,}")
    print(f"[DONE] items in interactions: {interactions[ITEM_ID_COL].nunique():,}")
    print(f"[SAVE] {PROCESSED_ITEM_FILE}")
    print(f"[SAVE] {PROCESSED_INTERACTION_FILE}")


if __name__ == "__main__":
    main()