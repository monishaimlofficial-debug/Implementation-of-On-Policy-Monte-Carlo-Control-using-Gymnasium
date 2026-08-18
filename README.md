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

Environment: FrozenLake-v1

The project utilizes the FrozenLake-v1 environment from Gymnasium, a classic grid-world reinforcement learning problem. In this setup, the agent navigates a 4x4 grid. The objective is to move from a starting position ('S') to a goal ('G') while avoiding frozen holes ('H'). For simplicity and to focus on the core Monte Carlo Control algorithm, the is_slippery parameter is set to False, meaning the agent's actions deterministically move it in the chosen direction without random slips. This provides a clear, discrete state and action space suitable for demonstrating value-based learning methods.

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


1. **Initialize the Environment and Parameters**
   Set up the FrozenLake environment and create a Q-table filled with zeros for all states and actions.
   Define the learning rate $\alpha$, discount factor $\gamma$, initial exploration rate $\epsilon$, decay rate, and total episodes.
   
---

2. **Generate Episodes using Epsilon-Greedy**
   Run the training loop for the defined number of episodes from start to finish.
   Use the $\epsilon$-greedy policy to balance random exploration with greedy exploitation at each step.
   Store the entire sequence of $(S_t, A_t, R_{t+1})$ transitions encountered in the episode.

---

3. **Calculate Discounted Returns**
   Set the return accumulator $G = 0$ after the episode terminates.
   Iterate backwards through the trajectory from the final step to the start.
   Compute the discounted return at each step using the update rule:
   $$G \leftarrow R_{t+1} + \gamma G$$

---

4. **Update Q-Values with First-Visit Check**
   Check if the current state-action pair $(S_t, A_t)$ occurred earlier in the episode.
   If it is the first visit, update its action-value in the Q-table using:
   $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big(G - Q(S_t, A_t)\big)$$

---

5. **Decay Epsilon and Extract Optimal Policy**
   Reduce $\epsilon$ after each episode until reaching $\epsilon_{min}$ to favor exploitation over time.
   Extract the final optimal policy by picking the action with the highest Q-value:
   $$\pi^*(s) = \arg\max_a Q(s, a)$$
   Compute moving average rewards across sliding windows to confirm convergence.

---


## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python

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
    G = 0  
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        G = reward + gamma * G

        first_occurrence = next(i for i, (s, a, r) in enumerate(episode) if s == state and a == action)
        if first_occurrence == t:

            Q[state, action] = Q[state, action] + alpha * (G - Q[state, action])

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
 Name: MONISH N
 Register Number: 212223240097


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
The Monte Carlo Control successfully trained an agent, achieving a high average reward of approximately 93% over 
the final 1000 episodes. The learning curve clearly shows rapid convergence to an optimal policy that efficiently 
navigates the FrozenLake environment.


```
---

## Inference
```text

The Monte Carlo Control experiment successfully trained an agent to navigate the FrozenLake environment. Over 20,000 episodes, the agent's average reward steadily increased, achieving a high success rate of approximately 91.9% in the last 1000 episodes. This demonstrates that off-policy Monte Carlo Control, using an epsilon-greedy exploration strategy, effectively learned an optimal greedy policy to guide the agent to the goal, avoiding holes, and achieving consistent positive rewards.

```





---

