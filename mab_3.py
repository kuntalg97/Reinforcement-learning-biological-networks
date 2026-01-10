import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import torch.amp as amp
from torch.multiprocessing import Process, Queue
import torch.multiprocessing as mp
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from datetime import datetime
import os

# Add parent directory to path for imports
current_dir = Path().resolve()
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
sys.path.append("../../../")
from Gyms.RealNetworkSync import RealNetworkSync
from StateReduction.DynamicStatePCA import DynamicStatePCA

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta = beta  # Importance sampling correction factor
        self.beta_increment = beta_increment
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == 0:
            return None, None, None

        # Calculate sampling probabilities
        probs = self.priorities[:len(self.memory)] ** self.alpha
        probs /= probs.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(device)

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = [self.memory[idx] for idx in indices]
        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority.item())  # Convert to float scalar

class MultiArmedBanditAgent:
    def __init__(self, state_dim, action_dim, num_stimuli):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_stimuli = num_stimuli

#       Hyperparameters to be played around with!
        # MAB parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.arm_q_values = np.zeros(action_dim)  # Q-values for each arm
        self.arm_counts = np.zeros(action_dim, dtype=int)  # Number of times each arm has been pulled
        self.learning_rate = 1e-4  # Learning rate for updating Q-values
        self.gamma = 0.99  # Discount factor

    def act(self, state):
        if random.random() < self.epsilon:
            # Exploration: choose a random action for each stimuli
            actions = np.random.randint(self.action_dim, size=self.num_stimuli)
            return actions
        else:
            # Exploitation: choose the best action (highest Q-value) for each stimuli
            best_action = np.argmax(self.arm_q_values)
            actions = np.array([best_action for _ in range(self.num_stimuli)])
            return actions

    def train(self, state, action, reward, next_state, done):
        # Update counts and Q-values for the chosen action
        for a in action:
            self.arm_counts[a] += 1
            # Simple averaging update
            self.arm_q_values[a] = self.arm_q_values[a] + (self.learning_rate * (reward - self.arm_q_values[a]))

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return 0 #Loss is not important since this is not deep learning

    def get_statistics(self):
        """Get current agent statistics"""
        return {
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate #For simplicity, show this value as learning rate in stats
        }

def train_on_realsync_network():
    # Environment parameters
    state_dim = 4
    action_dim = 5
    num_stimuli = 5
    circuit_id = 1

    # Create state reduction object
    state_reduction = DynamicStatePCA(state_dim=state_dim)

    # Create environment
    env = RealNetworkSync(action_dim=num_stimuli,
                          state_dim=state_dim,
                          circuit_id=circuit_id,
                          state_object=state_reduction)
    print("Collecting initial experiences for state reduction training...")
    spikes = []
    elecs = []

    # Collect initial experiences with progress bar
    with tqdm(total=1000, desc="Collecting experiences") as pbar:
        for _ in range(1000):
            try:
                action = env.action_space.sample()
                _, _, _, _, info = env.step(action)
                spikes.append(info['spikes'])
                elecs.append(info['elecs'])
                pbar.update(1)
            except Exception as e:
                print(f"Error during experience collection: {e}")
                continue

    # Train the state reduction directly on the state_reduction object
    print("Training state reduction...")
    X = state_reduction.fit(spikes, elecs)

    # Create MAB agent
    agent = MultiArmedBanditAgent(state_dim=state_dim,
                                  action_dim=action_dim,
                                  num_stimuli=num_stimuli)

    # Training parameters
    episodes = 28
    best_reward = float('-inf')
    rewards_history = []
    losses_history = []

    # Create directory for saving results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'real_network_training_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    # Training statistics
    stats = {
        'episode': [],
        'reward': [],
        'avg_reward': [],
        'rewards_per_action': [],
        'loss': [], #Loss not needed anymore
        'epsilon': [],
        'learning_rate': [], # Learning rate is a static attribute of the MAB
        'episode_length': []
    }

    print("\nStarting MAB training on real network...")
    progress_bar = tqdm(range(episodes), desc="Training Episodes")
    for episode in progress_bar:
        try:
            state, _ = env.reset()
            state = np.array(state).flatten()
            episode_reward = 0
            episode_loss = 0
            episode_steps = 0
            done = False

            while not done:
                # Get action from agent
                action = agent.act(state)

                # Take action in environment
                try:
                    next_state, reward, terminated, truncated, info = env.step(action)
                    next_state = np.array(next_state).flatten()
                    done = terminated or truncated

                    # Train agent
                    loss = agent.train(state, action, reward, next_state, done)

                    if loss is not None:
                        episode_loss += loss
                        episode_reward += reward
                        episode_steps += 1
                    state = next_state

                    print(f"Info: {info}")
                    done = (episode_steps % 1280 == 0)

                except Exception as e:
                    print(f"Error during step: {e}")
                    done = True
                    continue

            # Calculate statistics
            avg_reward = np.mean(rewards_history[-10:] + [episode_reward]) if rewards_history else episode_reward
            avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0
            rewards_per_action = episode_reward / episode_steps if episode_steps > 0 else 0

            # Update progress bar
            progress_bar.set_postfix({
                'reward': f'{episode_reward:.2f}',
                'avg_reward': f'{avg_reward:.2f}',
                'rewards_per_action': f'{rewards_per_action:.2f}',
                'epsilon': f'{agent.epsilon:.3f}',
                'lr': f'{agent.learning_rate:.2e}' #Show learning rate in progress bar
            })

            # Record statistics
            rewards_history.append(episode_reward)
            losses_history.append(avg_loss)
            stats['episode'].append(episode)
            stats['reward'].append(episode_reward)
            stats['avg_reward'].append(avg_reward)
            stats['rewards_per_action'].append(rewards_per_action)
            stats['loss'].append(avg_loss) # still record the loss for plotting
            stats['epsilon'].append(agent.epsilon)
            stats['learning_rate'].append(agent.get_statistics()['learning_rate']) #Show learning rate in stats
            stats['episode_length'].append(episode_steps)

            # Save intermediate results
            if (episode + 1) % 5 == 0:
                df = pd.DataFrame(stats)
                df.to_csv(os.path.join(results_dir, 'training_stats_intermediate.csv'), index=False)
                create_training_visualization(df, results_dir, is_final=False)

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                #There is no model anymore, so simply store the results instead.
                results = {
                    'arm_q_values': agent.arm_q_values,
                    'arm_counts': agent.arm_counts,
                    'best_reward': best_reward
                }
                torch.save(results, os.path.join(results_dir, 'best_results.pth'))

        except Exception as e:
            print(f"Error during episode {episode}: {e}")
            continue

    # Create final visualizations
    print("\nGenerating final visualizations...")
    df = pd.DataFrame(stats)
    create_training_visualization(df, results_dir, is_final=True)
    df.to_csv(os.path.join(results_dir, 'training_stats.csv'), index=False)

    # Print final statistics
    print("\nTraining Complete!")
    print(f"Results saved in: {results_dir}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final average reward: {avg_reward:.2f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Final learning rate: {agent.get_statistics()['learning_rate']:.2e}")
    print(f"Final rewards per action: {rewards_per_action:.2f}")

    return rewards_history, results_dir

def create_training_visualization(df, results_dir, is_final=True):
    """Create comprehensive training visualizations"""
    # Set up the plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = [12, 8]

    # 1. Training Progress Overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MAB Training Progress Overview', fontsize=16)

    # Rewards plot
    sns.lineplot(data=df, x='episode', y='reward', alpha=0.3, color='blue', ax=axes[0, 0])
    sns.lineplot(data=df, x='episode', y='avg_reward', color='green', ax=axes[0, 0])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend(['Episode Reward', '10-Episode Average'])

    # Rewards per action plot
    sns.lineplot(data=df, x='episode', y='rewards_per_action', color='red', ax=axes[0, 1])
    axes[0, 1].set_title('Rewards per Action')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Rewards per Action')

    # Loss plot
    sns.lineplot(data=df, x='episode', y='loss', color='green', ax=axes[1, 0])
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')

    # Epsilon plot
    sns.lineplot(data=df, x='episode', y='epsilon', color='purple', ax=axes[1, 1])
    axes[1, 1].set_title('Epsilon (Exploration Rate)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_overview.png'))
    plt.close()

    # 2. Learning Rate and Memory Size
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Learning rate
    sns.lineplot(data=df, x='episode', y='learning_rate', color='blue', ax=axes[0])
    axes[0].set_title('Learning Rate Schedule')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Learning Rate')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'learning_rate_memory.png'))
    plt.close()

def analyze_training_results(results_dir):
    """Analyze the training results"""
    # Load training statistics
    stats_path = os.path.join(results_dir, 'training_stats.csv')
    if not os.path.exists(stats_path):
        print(f"Statistics file not found: {stats_path}")
        return

    df = pd.read_csv(stats_path)

    # Print basic statistics
    print("\nBasic Statistics:")
    print(df[['episode', 'reward', 'avg_reward', 'loss', 'epsilon', 'learning_rate']].describe())

    # Plotting (optional): Load the training overview image
    img_path = os.path.join(results_dir, 'training_overview.png')
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.title('Training Overview')
        plt.axis('off')  # Hide axis
        plt.show()
    else:
        print(f"Training overview image not found: {img_path}")

# Example usage:
if __name__ == '__main__':
    rewards_history, results_dir = train_on_realsync_network()
    analyze_training_results(results_dir)

