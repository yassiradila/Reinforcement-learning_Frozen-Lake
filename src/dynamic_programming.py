"""Dynamic Programming Algorithms: Value Iteration & Policy Iteration."""

import numpy as np


def value_iteration(env, gamma: float = 0.99, theta: float = 1e-8, max_iterations: int = 1000, verbose: bool = False):
    """Solve MDP using Value Iteration via Bellman Optimality updates.

    Args:
        env: Gymnasium environment with unwrapped transition dynamics P.
        gamma: Discount factor (default 0.99).
        theta: Convergence threshold on maximum value difference (default 1e-8).
        max_iterations: Maximum allowable iterations (default 1000).
        verbose: Whether to log convergence info.

    Returns:
        tuple: (V, policy, iterations)
            - V (np.ndarray): Optimal state-value array of shape (n_states,).
            - policy (np.ndarray): Optimal deterministic policy array of shape (n_states,).
            - iterations (int): Number of iterations until convergence.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)

    iterations = 0
    for it in range(max_iterations):
        delta = 0.0
        V_old = V.copy()

        for s in range(n_states):
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_state, reward, _ in env.unwrapped.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V_old[next_state])

            V[s] = np.max(action_values)
            delta = max(delta, abs(V[s] - V_old[s]))

        iterations = it + 1
        if delta < theta:
            if verbose:
                print(f"[Value Iteration] Converged in {iterations} iterations (delta = {delta:.2e} < {theta:.2e}).")
            break

    # Extract greedy optimal policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, _ in env.unwrapped.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])
        policy[s] = np.argmax(action_values)

    return V, policy, iterations


def policy_evaluation(policy, env, gamma: float = 0.99, theta: float = 1e-8, max_iterations: int = 1000):
    """Evaluate a fixed policy using synchronous Bellman updates.

    Args:
        policy (np.ndarray): Policy mapping each state to an action.
        env: Gymnasium environment with unwrapped transition dynamics P.
        gamma: Discount factor.
        theta: Convergence threshold.
        max_iterations: Maximum allowable iterations.

    Returns:
        np.ndarray: Evaluated state-value array V of shape (n_states,).
    """
    n_states = env.observation_space.n
    V = np.zeros(n_states)

    for _ in range(max_iterations):
        delta = 0.0
        V_old = V.copy()

        for s in range(n_states):
            a = policy[s]
            v = 0.0
            for prob, next_state, reward, _ in env.unwrapped.P[s][a]:
                v += prob * (reward + gamma * V_old[next_state])
            V[s] = v
            delta = max(delta, abs(V[s] - V_old[s]))

        if delta < theta:
            break

    return V


def policy_improvement(V, env, gamma: float = 0.99):
    """Derive an improved greedy policy from current state values.

    Args:
        V (np.ndarray): State-value array.
        env: Gymnasium environment.
        gamma: Discount factor.

    Returns:
        np.ndarray: Improved deterministic policy array of shape (n_states,).
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, _ in env.unwrapped.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])
        policy[s] = np.argmax(action_values)

    return policy


def policy_iteration(env, gamma: float = 0.99, theta: float = 1e-8, max_iterations: int = 1000, seed: int = 42, verbose: bool = False):
    """Solve MDP using Policy Iteration (alternating evaluation and improvement).

    Args:
        env: Gymnasium environment.
        gamma: Discount factor.
        theta: Convergence threshold for policy evaluation.
        max_iterations: Maximum allowable policy iterations.
        seed: Random seed for initial random policy.
        verbose: Whether to log convergence info.

    Returns:
        tuple: (V, policy, iterations)
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    rng = np.random.RandomState(seed)
    policy = rng.randint(0, n_actions, size=n_states)

    iterations = 0
    for it in range(max_iterations):
        V = policy_evaluation(policy, env, gamma=gamma, theta=theta)
        new_policy = policy_improvement(V, env, gamma=gamma)
        iterations = it + 1

        if np.array_equal(policy, new_policy):
            if verbose:
                print(f"[Policy Iteration] Converged in {iterations} iterations.")
            return V, new_policy, iterations

        policy = new_policy

    return V, policy, iterations
