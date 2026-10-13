"""Environment utility functions for FrozenLake-v1."""

import gymnasium as gym


def make_env(map_name: str = "4x4", is_slippery: bool = True, render_mode: str = "rgb_array"):
    """Create and return a configured FrozenLake-v1 environment.

    Args:
        map_name: Map layout identifier, typically '4x4' or '8x8'.
        is_slippery: Whether agent movement is stochastic (True) or deterministic (False).
        render_mode: Rendering mode for Gymnasium (e.g., 'rgb_array', 'ansi', or None).

    Returns:
        gym.Env: Instantiated FrozenLake-v1 Gymnasium environment.
    """
    return gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode=render_mode,
    )
