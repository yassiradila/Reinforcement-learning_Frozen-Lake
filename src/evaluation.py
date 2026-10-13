"""Empirical policy evaluation module."""

import numpy as np


def evaluate_policy(env, policy: np.ndarray, n_episodes: int = 1000, seed: int = 42):
    """Empirically evaluate a deterministic policy across multiple episodes.

    Args:
        env: Gymnasium FrozenLake environment.
        policy (np.ndarray): Action mapping for each state.
        n_episodes (int): Total number of evaluation episodes (default 1000).
        seed (int): Base random seed for reproducibility.

    Returns:
        tuple: (success_rate, avg_reward)
            - success_rate (float): Fraction of episodes reaching the goal tile 'G'.
            - avg_reward (float): Average reward obtained across all episodes.
    """
    successes = 0
    total_reward = 0.0

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        episode_reward = 0.0

        while not done:
            action = int(policy[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        if episode_reward > 0.0:
            successes += 1
        total_reward += episode_reward

    success_rate = successes / n_episodes
    avg_reward = total_reward / n_episodes
    return success_rate, avg_reward
