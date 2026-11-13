# Reinforcement Learning for Biological Neural Networks

Can an AI agent learn to "steer" the activity of living neurons?

This project explores that question. We connect a reinforcement learning (RL) agent to neuronal cultures grown in a dish (*in vitro*). The agent sends electrical stimulation patterns to the culture, observes how the neurons fire in response, and gradually learns which stimulations produce the firing patterns we want.

This work was developed as part of the **"AI in Science" hackathon at the University of Chicago, 2025**.

---

## The idea in plain terms

Reinforcement learning is a trial-and-error way of learning, similar to how an animal learns from rewards:

1. **The agent acts** – it chooses a stimulation pattern to deliver to the culture.
2. **The environment responds** – the neurons fire (spike), and the spikes are recorded on a set of electrodes.
3. **A reward is computed** – a custom reward function scores how close the recorded activity is to the desired pattern.
4. **The agent learns** – over many repetitions, it favours the stimulations that earn higher rewards.

```
      ┌──────────────┐   stimulation pattern   ┌──────────────────────┐
      │              │ ──────────────────────► │  Neuronal culture    │
      │   RL agent   │                         │  (recorded spikes)   │
      │              │ ◄────────────────────── │                      │
      └──────────────┘   state + reward        └──────────────────────┘
```

As the environment uses data recorded dynamically from real cultures, the agent has to cope with the noise and variability of living biological systems (which is basically the challenge here).

---

## Goal: a "clockwise" firing pattern

To test the framework, we designed a reward that encourages activity to **travel in a circle** across four electrodes:

- Electrodes are arranged in a ring (0 → 1 → 2 → 3 → 0).
- When two consecutive spikes occur within **5 ms** and move one step **clockwise**, the agent earns a positive reward.
- If they move **counter-clockwise**, the agent is penalised.
- Consecutive moves in the same direction form a **streak**, and the reward (or penalty) **doubles** with each step of the streak. Breaking the streak, or a gap longer than 5 ms, resets it.

This "clockwise streak" reward strongly favours sustained, rotating waves of activity rather than isolated spikes.

---

## How the pieces fit together

### 1. Turning spikes into a compact state (UMAP)
Raw spike recordings are high-dimensional: many electrodes × many time bins. To give the agent a manageable view of the network, each response is:

1. Converted into a binary grid (electrode × time bin → spike / no spike).
2. Compressed into a small **4-dimensional state** using **UMAP**, a dimensionality-reduction technique.

Before training begins, the system collects about 1,000 responses to random (live) stimulations and fits UMAP on them. The trained reducer can be saved and reloaded later.

### 2. Learning approaches compared

| Approach | How it works | Uses the network state? |
|---|---|---|
| **Deep Q-Network (DQN)** | A neural network estimates how rewarding each possible stimulation is, given the current state of the culture. | Yes |
| **Multi-armed bandit (MAB)** | Treats each stimulation option like a slot-machine "arm" and tracks its average reward, mostly picking the best arm while still exploring. | No |

**DQN details** – the agent uses several well-established techniques to learn stably from noisy biological data:
- *Dueling architecture* – separately estimates how good a state is and how much better each action is.
- *Double DQN with a target network* – reduces over-optimistic value estimates.
- *Prioritized experience replay* – replays the most informative past experiences more often.
- *Reward shaping, clipping and normalization* – keeps rewards within a stable range.
- *Epsilon-greedy exploration* – starts by exploring randomly, then increasingly relies on what it has learned.

In both approaches, the agent chooses among **5 stimulation options** for each of **5 stimuli** at every step.

---

## Results

We evaluated the clockwise-streak reward using both Deep Q-learning and multi-armed bandit approaches. **Deep Q-learning achieved higher cumulative rewards than the bandit-based methods**, suggesting that taking the current state of the network into account helps the agent choose more effective stimulations.

---

## Repository contents

| File | Purpose |
|---|---|
| `dqn_real_umap_final_2.py` | Trains the Deep Q-Network agent on the real-network environment, using UMAP for state reduction. |
| `mab_3.py` | Trains the multi-armed bandit agent, used as a baseline for comparison. |
| `DynamicStateUMAP.py` | Converts spike recordings into compact states with UMAP; supports training, saving and loading. |
| `ClockwiseStreakReward.py` | The clockwise-streak reward function described above. |

---

## Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/) (a GPU is used automatically if available)
- `gymnasium`, `numpy`, `pandas`
- `umap-learn`
- `matplotlib`, `seaborn`, `tqdm`

Install them with:

```bash
pip install torch gymnasium numpy pandas umap-learn matplotlib seaborn tqdm
```

---

## Running the experiments

```bash
# Deep Q-Network agent
python dqn_real_umap_final_2.py

# Multi-armed bandit baseline
python mab_3.py
```

Each run creates a time-stamped results folder containing:
- **Training statistics** (`training_stats.csv`) – reward, rewards per action, loss, exploration rate and learning rate for every episode.
- **Plots** – a training overview, learning-rate curves and (for DQN) a 2-D view of the UMAP state space.
- **Saved models** – the best-performing agent found during training.

---

## Customizing the reward

Reward functions are small, self-contained classes with a `reward(response)` method, where `response` is an array of spikes with one row per spike: `[time_ms, electrode_id]`. To try a different goal, write a new class with the same method and pass it to the environment in place of `ClockwiseStreakReward`. The `penalty_factor` in `ClockwiseStreakReward` controls how harshly counter-clockwise activity is penalised.
