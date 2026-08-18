# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description







## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm



## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
Here is the entire code from cell g7N5_c8OVZ4U:

# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start

for i_episode in range(num_episodes):
    episode = generate_episode(epsilon)

    # Total reward for the episode
    total_reward = sum([reward for state, action, reward in episode])
    episode_rewards.append(total_reward)

    # Update Q-table
    G = 0  # Discounted return
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        G = reward + gamma * G

        # Check if (state, action) pair has been visited for the first time in this episode
        # This is a common simplification for Monte Carlo control
        first_occurrence = next(i for i, (s, a, r) in enumerate(episode) if s == state and a == action)
        if first_occurrence == t:
            # Simple average or incremental update
            Q[state, action] = Q[state, action] + alpha * (G - Q[state, action])

    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (i_episode + 1) % 1000 == 0:
        print(f"Episode {i_episode + 1}/{num_episodes}, Epsilon: {epsilon:.4f}, Avg Reward: {np.mean(episode_rewards[-1000:]):.2f}")
```

---

## Output

```text


Final Q-table:
[[0.625 0.566 0.912 0.662]
 [0.631 0.    0.922 0.691]
 [0.765 0.932 0.74  0.754]
 [0.752 0.    0.227 0.571]
 [0.717 0.785 0.    0.736]
 [0.    0.    0.    0.   ]
 [0.    0.954 0.    0.771]
 [0.    0.    0.    0.   ]
 [0.688 0.    0.792 0.759]
 [0.836 0.969 0.877 0.   ]
 [0.946 0.989 0.    0.952]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.927 0.99  0.918]
 [0.973 0.99  1.    0.979]
 [0.    0.    0.    0.   ]]


Estimated State-Value Function:
[[0.912 0.922 0.932 0.752]
 [0.785 0.    0.954 0.   ]
 [0.792 0.969 0.989 0.   ]
 [0.    0.99  1.    0.   ]]


Learned Policy:
[['R' 'R' 'D' 'L']
 ['D' 'L' 'D' 'L']
 ['R' 'D' 'D' 'L']
 ['L' 'R' 'R' 'L']]


Average reward over last 1000 episodes: 0.936


```


---

## Result
```text



```
---

## Inference
```text



```





---

