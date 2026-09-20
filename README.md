# FrozenLake-v1 Dynamic Programming Solver

This repository contains an implementation study of Dynamic Programming approaches to solve the Gymnasium FrozenLake-v1 environment[cite: 2]. This project was developed as an academic study at the National School of Applied Sciences of Tétouan (ENSA Tétouan) by Yassir Adila and Aymane Hayoun[cite: 2].

## Overview
The project evaluates the performance of **Value Iteration (VI)** and **Policy Iteration (PI)** on a finite Markov Decision Process (MDP)[cite: 2]. We analyze both deterministic and stochastic (slippery) grid dynamics across standard $4\times4$ and scaled $8\times8$ environments[cite: 2].

## Key Findings
* **Algorithm Efficiency:** Both VI and PI successfully derived the optimal policy, achieving an identical state value of $V(0) \approx 0.5420$ on the $4\times4$ stochastic grid[cite: 2]. PI proved significantly more computationally efficient in terms of outer loops, converging in just 4 iterations compared to VI's 438 iterations[cite: 2].
* **Environmental Impact:** The introduction of stochastic "slipping" mechanics forced the agent into cautious, detour-heavy behavior, reducing the success rate from $100\%$ in deterministic settings to $75.2\%$[cite: 2].
* **Scalability:** Expanding to an $8\times8$ stochastic grid exacerbated the curse of dimensionality, requiring 516 iterations for VI to converge and reducing the overall success rate to $62.9\%$ due to increased path complexity[cite: 2].

## Repository Contents
* `notebooks/`: Contains the complete Python implementation, including custom value heatmaps and directional policy visualizations[cite: 1].
* `docs/`: Includes the full technical report detailing the Bellman equation implementations, transition probability mechanics, and convergence properties[cite: 2].