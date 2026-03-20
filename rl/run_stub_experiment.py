import json
import random
from pathlib import Path

import matplotlib.pyplot as plt

from agent_stub import EpsilonGreedyAgent
from environment import DefectControlEnv


def run(num_episodes=80, seed=42):
    rng = random.Random(seed)

    env = DefectControlEnv()
    agent = EpsilonGreedyAgent(actions=env.ACTIONS, seed=seed)

    rewards_per_episode = []
    epsilons = []

    for _ in range(num_episodes):
        env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.choose_action()
            true_is_defect = rng.random() < 0.5
            _, reward, done = env.step(action, true_is_defect)
            ep_reward += reward

        rewards_per_episode.append(ep_reward)
        epsilons.append(agent.epsilon)
        agent.update_epsilon()

    experiments_dir = Path("experiments")
    metrics_dir = Path("metrics")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode)
    plt.title("Reward Per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.plot(epsilons)
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")

    plt.tight_layout()
    plt.savefig(experiments_dir / "rl_learning_curves.png", dpi=150)
    plt.close()

    summary = {
        "episodes": num_episodes,
        "reward_mean": sum(rewards_per_episode) / len(rewards_per_episode),
        "reward_min": min(rewards_per_episode),
        "reward_max": max(rewards_per_episode),
        "note": "Early RL curve is expected to be noisy and unstable.",
    }

    (metrics_dir / "rl_stub_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
