import random
import numpy as np
import pandas as pd

from config import (
    ITEM_ID_COL,
    USER_ID_COL,
    ATTRIBUTES,
    MAX_TURN,
    TOP_K,
    REWARD_RECOMMEND_SUCCESS,
    REWARD_RECOMMEND_FAIL,
    REWARD_ASK_SUCCESS,
    REWARD_INVALID_ASK,
    REWARD_CANDIDATE_EMPTY,
    REWARD_MAX_TURN,
)


class CRSEnv:
    """
    DQN용 CRS-MDP Environment.

    State:
        s_t = (t, K_t, B_t, C_t, H_t^rec)

    코드 내부:
        - K_t: known_preferences dict
        - B_t: asked_attributes set
        - C_t: candidate item ids set
        - H_t^rec: rejected item ids set
    """

    def __init__(
        self,
        items: pd.DataFrame,
        interactions: pd.DataFrame,
        max_turn: int = MAX_TURN,
        top_k: int = TOP_K,
        seed: int = 42,
    ):
        self.items = items.copy()
        self.interactions = interactions.copy()
        self.max_turn = max_turn
        self.top_k = top_k
        self.rng = random.Random(seed)

        # --------------------------------------------------
        # IMPORTANT:
        # CSV를 다시 읽으면 product_id가 int로 읽힐 수 있음.
        # reviews/interactions 쪽 product_id는 str로 처리하고 있었기 때문에
        # 여기서 양쪽 모두 문자열로 통일해야 함.
        # --------------------------------------------------
        self.items[ITEM_ID_COL] = self.items[ITEM_ID_COL].astype(str)
        self.interactions[ITEM_ID_COL] = self.interactions[ITEM_ID_COL].astype(str)
        self.interactions[USER_ID_COL] = self.interactions[USER_ID_COL].astype(str)

        # 필요한 속성 컬럼도 문자열로 통일
        for attr in ATTRIBUTES:
            if attr not in self.items.columns:
                raise ValueError(f"Missing attribute column in items: {attr}")
            self.items[attr] = self.items[attr].astype(str)

        if "popularity_score" not in self.items.columns:
            raise ValueError(
                "Missing 'popularity_score' column. "
                "Run preprocess.py before train.py."
            )

        self.item_ids = self.items[ITEM_ID_COL].tolist()
        self.item_table = self.items.set_index(ITEM_ID_COL, drop=False)

        valid_item_set = set(self.item_ids)

        # interaction에 있는 상품 중 items에 실제 존재하는 것만 유지
        self.interactions = self.interactions[
            self.interactions[ITEM_ID_COL].isin(valid_item_set)
        ].copy()

        self.user_to_items = (
            self.interactions.groupby(USER_ID_COL)[ITEM_ID_COL]
            .apply(lambda x: list(set(x.astype(str).tolist())))
            .to_dict()
        )

        # 실제 items 테이블에 존재하는 상품을 가진 user만 유지
        self.user_to_items = {
            user: [item for item in item_list if item in valid_item_set]
            for user, item_list in self.user_to_items.items()
        }
        self.user_to_items = {
            user: item_list
            for user, item_list in self.user_to_items.items()
            if len(item_list) > 0
        }

        self.users = list(self.user_to_items.keys())

        if len(self.users) == 0:
            raise ValueError(
                "No valid users found. "
                "Check whether product_id values in reviews.csv match items.csv."
            )

        self.attr_to_action = {attr: idx for idx, attr in enumerate(ATTRIBUTES)}
        self.action_to_attr = {idx: attr for attr, idx in self.attr_to_action.items()}
        self.recommend_action = len(ATTRIBUTES)
        self.num_actions = len(ATTRIBUTES) + 1

        # category/value vocabulary for compact known-value encoding
        self.attr_value_to_idx = {}
        self.attr_num_values = {}
        for attr in ATTRIBUTES:
            values = sorted(self.items[attr].dropna().astype(str).unique().tolist())
            self.attr_value_to_idx[attr] = {v: i for i, v in enumerate(values)}
            self.attr_num_values[attr] = max(len(values), 1)

        self.reset()

    @property
    def state_dim(self):
        # turn_ratio 1
        # asked binary len(attrs)
        # known binary len(attrs)
        # known value normalized len(attrs)
        # candidate bucket 1
        # fail count normalized 1
        return 1 + len(ATTRIBUTES) + len(ATTRIBUTES) + len(ATTRIBUTES) + 1 + 1

    def reset(self):
        self.turn = 1

        # 재귀 호출 대신 valid user/item에서 직접 sampling
        self.user_id = self.rng.choice(self.users)
        user_items = self.user_to_items[self.user_id]

        if len(user_items) == 0:
            raise RuntimeError(f"User {self.user_id} has no valid target items.")

        self.target_item_id = self.rng.choice(user_items)

        if self.target_item_id not in self.item_table.index:
            raise RuntimeError(
                f"Target item {self.target_item_id} not found in item_table. "
                f"Check product_id type consistency."
            )

        self.target_item = self.item_table.loc[self.target_item_id]

        self.known_preferences = {}
        self.asked_attributes = set()
        self.candidate_items = set(self.item_ids)
        self.recommend_fail_history = set()

        self.done = False
        self.success = False
        self.failure_reason = None

        return self.get_state_vector()

    def get_valid_actions(self):
        valid = []

        for attr, action_idx in self.attr_to_action.items():
            if attr not in self.asked_attributes:
                valid.append(action_idx)

        if len(self.candidate_items) > 0:
            valid.append(self.recommend_action)

        return valid

    def get_action_mask(self):
        mask = np.zeros(self.num_actions, dtype=np.float32)
        for a in self.get_valid_actions():
            mask[a] = 1.0
        return mask

    def candidate_bucket(self):
        n = len(self.candidate_items)
        if n == 0:
            return 0.0
        if n <= 10:
            return 0.25
        if n <= 50:
            return 0.50
        if n <= 200:
            return 0.75
        return 1.0

    def get_state_vector(self):
        turn_ratio = self.turn / self.max_turn

        asked_vec = []
        known_vec = []
        value_vec = []

        for attr in ATTRIBUTES:
            asked_vec.append(1.0 if attr in self.asked_attributes else 0.0)
            known_vec.append(1.0 if attr in self.known_preferences else 0.0)

            if attr in self.known_preferences:
                value = str(self.known_preferences[attr])
                idx = self.attr_value_to_idx[attr].get(value, 0)
                denom = max(self.attr_num_values[attr] - 1, 1)
                value_vec.append(idx / denom)
            else:
                value_vec.append(0.0)

        cand = self.candidate_bucket()
        fail_count = min(len(self.recommend_fail_history) / max(self.top_k * 3, 1), 1.0)

        state = np.array(
            [turn_ratio]
            + asked_vec
            + known_vec
            + value_vec
            + [cand, fail_count],
            dtype=np.float32,
        )
        return state

    def _filter_candidates(self, attr: str, value: str):
        if len(self.candidate_items) == 0:
            return set()

        candidate_df = self.items[self.items[ITEM_ID_COL].isin(self.candidate_items)]
        filtered = candidate_df[candidate_df[attr].astype(str) == str(value)]
        return set(filtered[ITEM_ID_COL].astype(str).tolist())

    def _recommend_top_k(self):
        if len(self.candidate_items) == 0:
            return []

        candidate_df = self.items[self.items[ITEM_ID_COL].isin(self.candidate_items)].copy()

        candidate_df = candidate_df[
            ~candidate_df[ITEM_ID_COL].astype(str).isin(self.recommend_fail_history)
        ]

        if len(candidate_df) == 0:
            return []

        candidate_df = candidate_df.sort_values("popularity_score", ascending=False)
        return candidate_df[ITEM_ID_COL].astype(str).head(self.top_k).tolist()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is already done. Call reset().")

        valid_actions = self.get_valid_actions()

        if action not in valid_actions:
            reward = REWARD_INVALID_ASK
            self.turn += 1

            if self.turn > self.max_turn:
                self.done = True
                self.failure_reason = "max_turn"
                reward += REWARD_MAX_TURN

            return self.get_state_vector(), reward, self.done, self._info()

        # Ask action
        if action != self.recommend_action:
            attr = self.action_to_attr[action]

            if attr in self.asked_attributes:
                reward = REWARD_INVALID_ASK
            else:
                value = str(self.target_item[attr])

                self.known_preferences[attr] = value
                self.asked_attributes.add(attr)
                self.candidate_items = self._filter_candidates(attr, value)

                reward = REWARD_ASK_SUCCESS

                if len(self.candidate_items) == 0:
                    self.done = True
                    self.failure_reason = "candidate_empty"
                    reward += REWARD_CANDIDATE_EMPTY

            self.turn += 1

        # Recommend action
        else:
            recommended = self._recommend_top_k()

            if len(recommended) == 0:
                reward = REWARD_CANDIDATE_EMPTY
                self.done = True
                self.failure_reason = "candidate_empty"
                return self.get_state_vector(), reward, self.done, self._info()

            if self.target_item_id in recommended:
                reward = REWARD_RECOMMEND_SUCCESS
                self.done = True
                self.success = True
                self.failure_reason = None
                return self.get_state_vector(), reward, self.done, self._info()

            # Recommendation rejected, continue dialogue
            self.recommend_fail_history.update(recommended)
            self.candidate_items = self.candidate_items.difference(set(recommended))

            reward = REWARD_RECOMMEND_FAIL
            self.turn += 1

            if len(self.candidate_items) == 0:
                self.done = True
                self.failure_reason = "candidate_empty"
                reward += REWARD_CANDIDATE_EMPTY

        if not self.done and self.turn > self.max_turn:
            self.done = True
            self.failure_reason = "max_turn"
            reward += REWARD_MAX_TURN

        return self.get_state_vector(), reward, self.done, self._info()

    def _info(self):
        return {
            "user_id": self.user_id,
            "target_item_id": self.target_item_id,
            "turn": self.turn,
            "candidate_count": len(self.candidate_items),
            "success": self.success,
            "failure_reason": self.failure_reason,
            "asked_attributes": list(self.asked_attributes),
            "recommend_fail_count": len(self.recommend_fail_history),
        }