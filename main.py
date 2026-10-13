"""Main execution script for FrozenLake Dynamic Programming benchmarks."""

import argparse
import os
from src.environment import make_env
from src.dynamic_programming import value_iteration, policy_iteration
from src.evaluation import evaluate_policy
from src.visualization import (
    plot_value_policy,
    plot_comparison,
    plot_success_rates,
)


def run_benchmark(gamma: float = 0.99, theta: float = 1e-8, n_episodes: int = 1000, save_plots: bool = False, assets_dir: str = "assets"):
    """Run full suite of experiments comparing VI and PI across grid sizes and dynamics."""
    print("=" * 70)
    print(" FROZENLAKE-V1 DYNAMIC PROGRAMMING BENCHMARK ")
    print("=" * 70)

    if save_plots:
        os.makedirs(assets_dir, exist_ok=True)

    # 1. 4x4 Stochastic - Value Iteration
    print("\n[1/4] Running Value Iteration on 4x4 Stochastic...")
    env_4x4_stoch = make_env("4x4", is_slippery=True)
    V_vi_4x4, pi_vi_4x4, it_vi_4x4 = value_iteration(env_4x4_stoch, gamma=gamma, theta=theta, verbose=True)
    sr_vi_4x4, ar_vi_4x4 = evaluate_policy(env_4x4_stoch, pi_vi_4x4, n_episodes=n_episodes)

    # 2. 4x4 Stochastic - Policy Iteration
    print("\n[2/4] Running Policy Iteration on 4x4 Stochastic...")
    V_pi_4x4, pi_pi_4x4, it_pi_4x4 = policy_iteration(env_4x4_stoch, gamma=gamma, theta=theta, verbose=True)
    sr_pi_4x4, ar_pi_4x4 = evaluate_policy(env_4x4_stoch, pi_pi_4x4, n_episodes=n_episodes)

    # 3. 4x4 Deterministic - Value Iteration
    print("\n[3/4] Running Value Iteration on 4x4 Deterministic...")
    env_4x4_det = make_env("4x4", is_slippery=False)
    V_det_4x4, pi_det_4x4, it_det_4x4 = value_iteration(env_4x4_det, gamma=gamma, theta=theta, verbose=True)
    sr_det_4x4, ar_det_4x4 = evaluate_policy(env_4x4_det, pi_det_4x4, n_episodes=n_episodes)

    # 4. 8x8 Stochastic - Value Iteration
    print("\n[4/4] Running Value Iteration on 8x8 Stochastic...")
    env_8x8_stoch = make_env("8x8", is_slippery=True)
    V_vi_8x8, pi_vi_8x8, it_vi_8x8 = value_iteration(env_8x8_stoch, gamma=gamma, theta=theta, verbose=True)
    sr_vi_8x8, ar_vi_8x8 = evaluate_policy(env_8x8_stoch, pi_vi_8x8, n_episodes=n_episodes)

    print("\n" + "=" * 70)
    print(f"{'Environment':<12} | {'Algorithm':<16} | {'Slippery':<8} | {'Iters':<6} | {'V(S0)':<8} | {'Success':<8} | {'Avg Reward'}")
    print("-" * 70)
    print(f"{'4x4':<12} | {'Value Iteration':<16} | {'Yes':<8} | {it_vi_4x4:<6} | {V_vi_4x4[0]:<8.4f} | {sr_vi_4x4:<8.1%} | {ar_vi_4x4:.4f}")
    print(f"{'4x4':<12} | {'Policy Iteration':<16} | {'Yes':<8} | {it_pi_4x4:<6} | {V_pi_4x4[0]:<8.4f} | {sr_pi_4x4:<8.1%} | {ar_pi_4x4:.4f}")
    print(f"{'4x4':<12} | {'Value Iteration':<16} | {'No':<8} | {it_det_4x4:<6} | {V_det_4x4[0]:<8.4f} | {sr_det_4x4:<8.1%} | {ar_det_4x4:.4f}")
    print(f"{'8x8':<12} | {'Value Iteration':<16} | {'Yes':<8} | {it_vi_8x8:<6} | {V_vi_8x8[0]:<8.4f} | {sr_vi_8x8:<8.1%} | {ar_vi_8x8:.4f}")
    print("=" * 70)

    if save_plots:
        print("\nSaving figures to", assets_dir)
        plot_value_policy(V_vi_4x4, pi_vi_4x4, env_4x4_stoch, "Value Iteration - 4x4 Stochastic", os.path.join(assets_dir, "value_iteration_4x4.png"))
        plot_value_policy(V_pi_4x4, pi_pi_4x4, env_4x4_stoch, "Policy Iteration - 4x4 Stochastic", os.path.join(assets_dir, "policy_iteration_4x4.png"))
        plot_comparison(V_vi_4x4, V_pi_4x4, "Value Iteration (4x4)", "Policy Iteration (4x4)", os.path.join(assets_dir, "comparison_vi_vs_pi.png"))
        plot_comparison(V_vi_4x4, V_det_4x4, "Stochastic (4x4)", "Deterministic (4x4)", os.path.join(assets_dir, "stochastic_vs_deterministic.png"))
        plot_value_policy(V_vi_8x8, pi_vi_8x8, env_8x8_stoch, "Value Iteration - 8x8 Stochastic", os.path.join(assets_dir, "value_iteration_8x8.png"))
        plot_success_rates(["VI 4x4 Stoch", "PI 4x4 Stoch", "VI 4x4 Det", "VI 8x8 Stoch"], [sr_vi_4x4, sr_pi_4x4, sr_det_4x4, sr_vi_8x8], save_path=os.path.join(assets_dir, "empirical_success_rates.png"))
        print("Done saving figures.")


def main():
    parser = argparse.ArgumentParser(description="Solve FrozenLake-v1 with Dynamic Programming (VI / PI).")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--theta", type=float, default=1e-8, help="Convergence threshold (default: 1e-8)")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of evaluation episodes (default: 1000)")
    parser.add_argument("--save-plots", action="store_true", help="Save visualization plots to assets directory")
    parser.add_argument("--assets-dir", type=str, default="assets", help="Directory where assets are saved")
    args = parser.parse_args()

    run_benchmark(
        gamma=args.gamma,
        theta=args.theta,
        n_episodes=args.episodes,
        save_plots=args.save_plots,
        assets_dir=args.assets_dir,
    )


if __name__ == "__main__":
    main()
