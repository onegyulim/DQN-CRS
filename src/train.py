import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from config import (
    PROCESSED_ITEM_FILE,
    PROCESSED_INTERACTION_FILE,
    CHECKPOINT_DIR,
    METRIC_DIR,
    PLOT_DIR,
    NUM_EPISODES,
    GAMMA,
    LR,
    BATCH_SIZE,
    REPLAY_BUFFER_SIZE,
    MIN_REPLAY_SIZE,
    TARGET_UPDATE_FREQ,
    TRAIN_FREQ,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    HIDDEN_DIM,
    SEED,
)
from env import CRSEnv
from dqn_model import DQN
from replay_buffer import ReplayBuffer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epsilon_by_step(step: int):
    # [실험 비교] 고정 epsilon 방식도 시도함 → 수렴 속도가 느리고 최종 성공률이 낮아 채택하지 않음
    # def epsilon_fixed():
    #     return 0.1
    #
    # 최종 채택: 지수 감소 방식 (EPS_START=1.0 → EPS_END=0.05)
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * step / EPS_DECAY)


def select_action(policy_net, state, valid_mask, epsilon, device):
    valid_actions = np.where(valid_mask > 0)[0].tolist()

    if len(valid_actions) == 0:
        raise RuntimeError("No valid actions available.")

    if random.random() < epsilon:
        return random.choice(valid_actions)

    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(state_t).squeeze(0).cpu().numpy()

    # invalid action masking
    q_values[valid_mask <= 0] = -1e9
    return int(np.argmax(q_values))


def optimize_model(policy_net, target_net, optimizer, replay_buffer, device):
    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return None

    states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(BATCH_SIZE)

    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
    next_masks = torch.tensor(next_masks, dtype=torch.float32, device=device)

    q_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        next_q_values = target_net(next_states)
        next_q_values = next_q_values.masked_fill(next_masks <= 0, -1e9)
        max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
        target = rewards + GAMMA * (1.0 - dones) * max_next_q

    loss = F.mse_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
    optimizer.step()

    return loss.item()


def moving_average(values, window=100):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main():
    set_seed(SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("[INFO] Loading processed data...")
    items = pd.read_csv(PROCESSED_ITEM_FILE)
    interactions = pd.read_csv(PROCESSED_INTERACTION_FILE)

    env = CRSEnv(items, interactions, seed=SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] State dim: {env.state_dim}")
    print(f"[INFO] Action dim: {env.num_actions}")

    policy_net = DQN(env.state_dim, env.num_actions, HIDDEN_DIM).to(device)
    target_net = DQN(env.state_dim, env.num_actions, HIDDEN_DIM).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    episode_rewards = []
    episode_success = []
    episode_turns = []
    losses = []

    global_step = 0

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        last_info = None

        while not done:
            valid_mask = env.get_action_mask()
            epsilon = epsilon_by_step(global_step)
            action = select_action(policy_net, state, valid_mask, epsilon, device)

            next_state, reward, done, info = env.step(action)
            next_mask = env.get_action_mask() if not done else np.zeros(env.num_actions, dtype=np.float32)

            replay_buffer.push(state, action, reward, next_state, done, next_mask)

            state = next_state
            total_reward += reward
            last_info = info

            if global_step % TRAIN_FREQ == 0:
                loss = optimize_model(policy_net, target_net, optimizer, replay_buffer, device)
                if loss is not None:
                    losses.append(loss)

            if global_step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            global_step += 1

        episode_rewards.append(total_reward)
        episode_success.append(1 if last_info and last_info["success"] else 0)
        episode_turns.append(last_info["turn"] if last_info else env.max_turn)

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            success_rate = np.mean(episode_success[-100:])
            avg_turn = np.mean(episode_turns[-100:])
            print(
                f"[EP {episode:5d}] "
                f"avg_reward={avg_reward:.4f} "
                f"success_rate={success_rate:.4f} "
                f"avg_turn={avg_turn:.2f} "
                f"epsilon={epsilon:.4f}"
            )

    ckpt_path = os.path.join(CHECKPOINT_DIR, "dqn_crs.pt")
    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "state_dim": env.state_dim,
            "action_dim": env.num_actions,
        },
        ckpt_path,
    )
    print(f"[SAVE] checkpoint: {ckpt_path}")

    metrics = pd.DataFrame(
        {
            "episode": np.arange(1, NUM_EPISODES + 1),
            "reward": episode_rewards,
            "success": episode_success,
            "turn": episode_turns,
        }
    )
    metrics_path = os.path.join(METRIC_DIR, "train_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"[SAVE] metrics: {metrics_path}")

    plt.figure()
    ma_reward = moving_average(episode_rewards, window=100)
    plt.plot(ma_reward)
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.title("DQN Training Reward")
    plt.tight_layout()
    reward_plot_path = os.path.join(PLOT_DIR, "train_reward.png")
    plt.savefig(reward_plot_path, dpi=200)
    plt.close()

    plt.figure()
    ma_success = moving_average(episode_success, window=100)
    plt.plot(ma_success)
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Success Rate")
    plt.title("DQN Training Success Rate")
    plt.tight_layout()
    success_plot_path = os.path.join(PLOT_DIR, "train_success.png")
    plt.savefig(success_plot_path, dpi=200)
    plt.close()

    print(f"[SAVE] plot: {reward_plot_path}")
    print(f"[SAVE] plot: {success_plot_path}")


if __name__ == "__main__":
    main()