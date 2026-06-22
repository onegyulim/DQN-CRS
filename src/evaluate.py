import os
import random
import numpy as np
import pandas as pd
import torch

from config import (
    PROCESSED_ITEM_FILE,
    PROCESSED_INTERACTION_FILE,
    CHECKPOINT_DIR,
    METRIC_DIR,
    EVAL_EPISODES,
    HIDDEN_DIM,
    SEED,
)
from env import CRSEnv
from dqn_model import DQN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_dqn_action(model, state, valid_mask, device):
    # [실험 비교] epsilon-greedy 방식도 시도함 → 순수 greedy보다 성공률이 낮고 불안정하여 채택하지 않음
    # valid_actions = np.where(valid_mask > 0)[0].tolist()
    # if random.random() < 0.05:
    #     return random.choice(valid_actions)
    #
    # 최종 채택: 순수 greedy (argmax Q-value)
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model(state_t).squeeze(0).cpu().numpy()
    q_values[valid_mask <= 0] = -1e9
    return int(np.argmax(q_values))


def run_policy(env, policy_name, model=None, device=None, num_episodes=500):
    results = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        ask_count = 0
        rec_count = 0
        last_info = None

        while not done:
            valid_mask = env.get_action_mask()
            valid_actions = np.where(valid_mask > 0)[0].tolist()

            if policy_name == "random":
                action = random.choice(valid_actions)

            elif policy_name == "heuristic":
                # heuristic: 정해진 순서대로 질문 가능한 속성을 먼저 묻고, 후보가 충분히 줄면 추천
                recommend_action = env.recommend_action

                if len(env.candidate_items) <= 50 and recommend_action in valid_actions:
                    action = recommend_action
                else:
                    ask_actions = [a for a in valid_actions if a != recommend_action]
                    if len(ask_actions) > 0:
                        action = ask_actions[0]
                    else:
                        action = recommend_action

            elif policy_name == "dqn":
                if model is None:
                    raise ValueError("DQN model is required for dqn policy.")
                action = select_dqn_action(model, state, valid_mask, device)

            else:
                raise ValueError(f"Unknown policy: {policy_name}")

            if action == env.recommend_action:
                rec_count += 1
            else:
                ask_count += 1

            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            last_info = info

        results.append(
            {
                "policy": policy_name,
                "success": 1 if last_info and last_info["success"] else 0,
                "total_reward": total_reward,
                "turn": last_info["turn"] if last_info else env.max_turn,
                "ask_count": ask_count,
                "recommend_count": rec_count,
                "failure_reason": last_info["failure_reason"] if last_info else "unknown",
            }
        )

    return pd.DataFrame(results)


def summarize(df):
    summary = (
        df.groupby("policy")
        .agg(
            success_rate=("success", "mean"),
            avg_reward=("total_reward", "mean"),
            avg_turn=("turn", "mean"),
            avg_ask_count=("ask_count", "mean"),
            avg_recommend_count=("recommend_count", "mean"),
        )
        .reset_index()
    )

    failure = (
        df[df["success"] == 0]
        .groupby(["policy", "failure_reason"])
        .size()
        .reset_index(name="count")
    )

    return summary, failure


def main():
    set_seed(SEED)

    os.makedirs(METRIC_DIR, exist_ok=True)

    print("[INFO] Loading processed data...")
    items = pd.read_csv(PROCESSED_ITEM_FILE)
    interactions = pd.read_csv(PROCESSED_INTERACTION_FILE)

    env = CRSEnv(items, interactions, seed=SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, "dqn_crs.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = DQN(checkpoint["state_dim"], checkpoint["action_dim"], HIDDEN_DIM).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_results = []

    for policy in ["random", "heuristic", "dqn"]:
        print(f"[INFO] Evaluating {policy}...")
        df = run_policy(
            env=env,
            policy_name=policy,
            model=model if policy == "dqn" else None,
            device=device,
            num_episodes=EVAL_EPISODES,
        )
        all_results.append(df)

    result_df = pd.concat(all_results, ignore_index=True)
    summary, failure = summarize(result_df)

    result_path = os.path.join(METRIC_DIR, "evaluation_raw.csv")
    summary_path = os.path.join(METRIC_DIR, "evaluation_summary.csv")
    failure_path = os.path.join(METRIC_DIR, "evaluation_failure.csv")

    result_df.to_csv(result_path, index=False)
    summary.to_csv(summary_path, index=False)
    failure.to_csv(failure_path, index=False)

    print("\n[Evaluation Summary]")
    print(summary)

    print("\n[Failure Breakdown]")
    print(failure)

    print(f"[SAVE] {result_path}")
    print(f"[SAVE] {summary_path}")
    print(f"[SAVE] {failure_path}")


if __name__ == "__main__":
    main()