"""Visualization utilities for FrozenLake state values and policies."""

from typing import Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


ACTION_ARROWS = {
    0: (-0.3, 0.0),   # LEFT
    1: (0.0, 0.3),    # DOWN
    2: (0.3, 0.0),    # RIGHT
    3: (0.0, -0.3),   # UP
}


def plot_grid(env, title: str = "FrozenLake Grid", save_path: Optional[str] = None):
    """Render and display/save the initial environment grid."""
    fig, ax = plt.subplots(figsize=(5, 5))
    env.reset()
    frame = env.render()
    ax.imshow(frame)
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_value_policy(
    V: np.ndarray,
    policy: np.ndarray,
    env,
    title: str = "Value Function & Optimal Policy",
    save_path: Optional[str] = None,
):
    """Visualize state values as a heatmap overlaid with directional policy arrows."""
    n_states = env.observation_space.n
    grid_size = int(np.sqrt(n_states))
    V_grid = V.reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Value heatmap
    im = ax.imshow(V_grid, cmap="YlOrRd", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("State Value", rotation=270, labelpad=20)

    # Annotate numeric values
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            ax.text(
                j,
                i,
                f"{V[state]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10 if grid_size <= 4 else 8,
                weight="bold",
            )

    # Directional arrows for optimal actions
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            action = int(policy[state])
            dx, dy = ACTION_ARROWS[action]

            arrow = FancyArrowPatch(
                (j, i),
                (j + dx, i + dy),
                arrowstyle="->",
                mutation_scale=20 if grid_size <= 4 else 12,
                color="blue",
                linewidth=2,
                alpha=0.75,
            )
            ax.add_patch(arrow)

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels(np.arange(grid_size))
    ax.set_yticklabels(np.arange(grid_size))
    ax.set_title(title, fontsize=14, weight="bold", pad=20)
    ax.grid(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(
    V1: np.ndarray,
    V2: np.ndarray,
    title1: str = "Configuration 1",
    title2: str = "Configuration 2",
    save_path: Optional[str] = None,
):
    """Side-by-side heatmap comparison of two value functions."""
    grid_size1 = int(np.sqrt(len(V1)))
    grid_size2 = int(np.sqrt(len(V2)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, V, gs, title in zip(axes, [V1, V2], [grid_size1, grid_size2], [title1, title2]):
        V_grid = V.reshape(gs, gs)
        im = ax.imshow(V_grid, cmap="YlOrRd", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(gs):
            for j in range(gs):
                state = i * gs + j
                ax.text(
                    j,
                    i,
                    f"{V[state]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10 if gs <= 4 else 8,
                    weight="bold",
                )

        ax.set_xticks(np.arange(gs))
        ax.set_yticks(np.arange(gs))
        ax.set_title(title, fontsize=14, weight="bold", pad=15)
        ax.grid(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_success_rates(
    methods: Sequence[str],
    success_rates: Sequence[float],
    title: str = "Empirical Success Rate Comparison",
    save_path: Optional[str] = None,
):
    """Bar chart visualization of success rates."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
    bars = ax.bar(methods, success_rates, color=colors[: len(methods)], alpha=0.85, edgecolor="black")

    ax.set_ylabel("Success Rate", fontsize=12, weight="bold")
    ax.set_title(title, fontsize=14, weight="bold", pad=15)
    ax.set_ylim(0, 1.1)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
