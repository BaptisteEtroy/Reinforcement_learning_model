import os
import math
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
import gym_race
from collections import deque
import pygame
import time

# Configuration
VERSION_NAME = 'DQN_pytorch'
REPORT_EPISODES = 1
DISPLAY_EPISODES = 5
SAVE_EPISODES = 100

# Hyperparameters
LEARNING_RATE = 0.0001
GAMMA = 0.99
MEMORY_SIZE = 50000
BATCH_SIZE = 32
TRAINING_FREQUENCY = 8
EPSILON_START = 0.3
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9998

# Training settings
NUM_EPISODES = 10000
MAX_STEPS = 2000
UPDATE_TARGET_EVERY = 10
RENDER_EVERY = 1

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dueling DQN architecture (separates value and advantage streams)
class EnhancedDQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(EnhancedDQNetwork, self).__init__()
        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),
        )
        
        # Value stream (estimates state value)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        
        # Advantage stream (estimates advantage of each action)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), 
            nn.LeakyReLU(0.1),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Combine using dueling architecture formula
        return value + (advantages - advantages.mean(dim=1, keepdim=True))

# Prioritized replay buffer for more efficient learning
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        self.expected_state_size = None
        self.warning_count = 0
    
    def push(self, state, action, reward, next_state, done):
        try:
            state_np = np.array(state, dtype=np.float32).flatten()
            next_state_np = np.array(next_state, dtype=np.float32).flatten()
            
            if self.expected_state_size is None and len(self.buffer) == 0:
                self.expected_state_size = state_np.shape[0]
                print(f"Setting expected state size to {self.expected_state_size}")
            
            if self.expected_state_size is not None:
                if state_np.shape[0] != self.expected_state_size:
                    self.warning_count += 1
                    if self.warning_count <= 3 or self.warning_count % 1000 == 0:
                        print(f"Warning: State size mismatch. Expected {self.expected_state_size}, got {state_np.shape[0]}")
                        if self.warning_count == 3:
                            print("Suppressing further warnings...")
                    
                    state_np = self._normalize_state(state_np, self.expected_state_size)
                
                if next_state_np.shape[0] != self.expected_state_size:
                    next_state_np = self._normalize_state(next_state_np, self.expected_state_size)
            
            if len(self.buffer) < self.capacity:
                self.buffer.append((state_np, action, reward, next_state_np, done))
            else:
                self.buffer[self.position] = (state_np, action, reward, next_state_np, done)
            
            self.priorities[self.position] = self.max_priority
            self.position = (self.position + 1) % self.capacity
        except Exception as e:
            print(f"Error in replay buffer push: {e}")
    
    def _normalize_state(self, state, target_size):
        if len(state) < target_size:
            padded = np.zeros(target_size, dtype=np.float32)
            padded[:len(state)] = state
            return padded
        else:
            return state[:target_size]
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None
        
        try:
            if len(self.buffer) < self.capacity:
                probs = self.priorities[:len(self.buffer)]
            else:
                probs = self.priorities
            
            probs = probs ** self.alpha
            probs = probs / probs.sum()
            
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            samples = [self.buffer[idx] for idx in indices]
            
            weights = (len(self.buffer) * probs[indices]) ** (-beta)
            weights = weights / weights.max()
            
            states, actions, rewards, next_states, dones = [], [], [], [], []
            for s, a, r, ns, d in samples:
                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(ns)
                dones.append(d)
            
            try:
                states_tensor = torch.FloatTensor(np.array(states, dtype=np.float32)).to(device)
                next_states_tensor = torch.FloatTensor(np.array(next_states, dtype=np.float32)).to(device)
                actions_tensor = torch.LongTensor(np.array(actions)).to(device)
                rewards_tensor = torch.FloatTensor(np.array(rewards, dtype=np.float32)).to(device)
                dones_tensor = torch.FloatTensor(np.array(dones, dtype=np.float32)).to(device)
                weights_tensor = torch.FloatTensor(weights).to(device)
                
                return (
                    states_tensor,
                    actions_tensor,
                    rewards_tensor,
                    next_states_tensor,
                    dones_tensor,
                    weights_tensor,
                    indices
                )
            except Exception as e:
                print(f"Error converting tensors: {e}")
                return None
        except Exception as e:
            print(f"Error in replay buffer sample: {e}")
            return None
    
    def update_priorities(self, indices, priorities):
        try:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
        except Exception as e:
            print(f"Error updating priorities: {e}")
    
    def __len__(self):
        return len(self.buffer)

# Agent implementing Double DQN with prioritized replay
class EnhancedDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.policy_net = EnhancedDQNetwork(state_size, action_size).to(device)
        self.target_net = EnhancedDQNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE)
        
        self.epsilon = EPSILON_START
        self.steps = 0
        self.beta = 0.4
        self.beta_increment = 0.001
        self.recent_actions = deque(maxlen=100)
    
    def act(self, state, training=True):
        try:
            state_array = np.array(state, dtype=np.float32).flatten()
            
            if state_array.shape[0] != self.state_size:
                if state_array.shape[0] < self.state_size:
                    padded_state = np.zeros(self.state_size, dtype=np.float32)
                    padded_state[:state_array.shape[0]] = state_array
                    state_array = padded_state
                else:
                    state_array = state_array[:self.state_size]
            
            if training and random.random() < self.epsilon:
                if random.random() < 0.2:
                    action = random.randrange(self.action_size)
                else:
                    action_counts = np.zeros(self.action_size)
                    for a in self.recent_actions:
                        action_counts[a] += 1
                    
                    action_counts = action_counts + 1.0
                    probs = 1.0 / action_counts
                    probs = probs / np.sum(probs)
                    action = np.random.choice(self.action_size, p=probs)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(device)
                    q_values = self.policy_net(state_tensor)
                    action = q_values.max(1)[1].item()
            
            self.recent_actions.append(action)
            return action
        except Exception as e:
            print(f"Error in act method: {e}")
            action = random.randrange(self.action_size)
            self.recent_actions.append(action)
            return action
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = self.memory.sample(BATCH_SIZE, self.beta)
        if batch is None:
            print("Skipping training step - invalid batch")
            return 0.0
            
        states, actions, rewards, next_states, dones, weights, indices = batch
        
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use policy net to select actions, target net to evaluate them
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + GAMMA * next_q_values * (1 - dones.unsqueeze(1))
        
        td_errors = torch.abs(target_q_values - q_values)
        loss = (weights.unsqueeze(1) * F.smooth_l1_loss(q_values, target_q_values, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        
        self.optimizer.step()
        self.steps += 1
        
        with torch.no_grad():
            self.memory.update_priorities(indices, (td_errors.detach().cpu().numpy() + 1e-6).flatten())
        
        return loss.item()
    
    def update_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * \
                          math.exp(-self.steps / 10000)
    
    def save(self, filepath):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'beta': self.beta
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        try:
            checkpoint = torch.load(filepath, map_location=device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            if 'beta' in checkpoint:
                self.beta = checkpoint['beta']
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


# Training loop for the DQN agent
def train():
    if not os.path.exists(f'models_{VERSION_NAME}'):
        os.makedirs(f'models_{VERSION_NAME}')
    
    pygame.init()
    
    try:
        env = gym.make("Pyrace-v3", render_mode="human").unwrapped
        print("Using enhanced Pyrace-v3 environment")
    except Exception as e:
        print(f"Error loading Pyrace-v3: {e}")
        print("Falling back to Pyrace-v1...")
        try:
            env = gym.make("Pyrace-v1", render_mode="human").unwrapped
            print("Using Pyrace-v1 environment")
        except Exception as e2:
            print(f"Error loading environment: {e2}")
            env = gym.make("Pyrace-v1")
            print("Using wrapped Pyrace-v1 environment")
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"State size: {state_size}, Action size: {action_size}")
    
    agent = EnhancedDQNAgent(state_size, action_size)
    
    pygame.display.init()
    screen = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Enhanced PyTorch DQN Racing")
    
    rewards = []
    losses = []
    best_reward = -float('inf')
    episode_lengths = []
    checkpoint_counts = []
    action_distribution = np.zeros(action_size)
    
    # Try to load previous checkpoint if exists
    start_episode = 0
    if os.path.exists(f'models_{VERSION_NAME}/latest.pt'):
        if agent.load(f'models_{VERSION_NAME}/latest.pt'):
            if os.path.exists(f'models_{VERSION_NAME}/progress.npy'):
                progress = np.load(f'models_{VERSION_NAME}/progress.npy', allow_pickle=True).item()
                start_episode = progress.get('episode', 0)
                rewards = progress.get('rewards', [])
                losses = progress.get('losses', [])
                best_reward = progress.get('best_reward', -float('inf'))
                print(f"Resuming from episode {start_episode}")
    
    start_time = time.time()
    
    # Main training loop
    for episode in range(start_episode, NUM_EPISODES):
        state, _ = env.reset()
        
        should_render = (episode % DISPLAY_EPISODES == 0)
        env.set_view(should_render)
        
        if hasattr(env, 'pyrace') and hasattr(env.pyrace, 'mode'):
            env.pyrace.mode = int(0)
        
        pygame.event.pump()
        
        total_reward = 0
        episode_loss = 0
        loss_count = 0
        episode_actions = []
        max_checkpoint = 0
        episode_start_time = time.time()
        
        # Episode step loop
        for step in range(MAX_STEPS):
            if step % 10 == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        agent.save(f'models_{VERSION_NAME}/interrupted.pt')
                        pygame.quit()
                        return
            
            try:
                state_flat = np.array(state, dtype=np.float32).flatten()
                action = agent.act(state_flat)
                episode_actions.append(action)
                
                next_state, reward, done, _, info = env.step(action)
                
                next_state_flat = np.array(next_state, dtype=np.float32).flatten()
                
                agent.memory.push(state_flat, action, reward, next_state_flat, done)
                
                total_reward += reward
                action_distribution[action] += 1
                
                current_check = info.get('check', 0)
                max_checkpoint = max(max_checkpoint, current_check)
                
                state = next_state_flat
            except Exception as e:
                print(f"Error during action execution: {e}")
                if 'next_state_flat' in locals():
                    state = next_state_flat
                done = True
            
            if should_render and step % RENDER_EVERY == 0:
                debug_info = [
                    'Enhanced PyTorch DQN',
                    f'Episode: {episode}',
                    f'Step: {step}',
                    f'Reward: {total_reward:.1f}',
                    f'Epsilon: {agent.epsilon:.3f}',
                    f'Check: {current_check}',
                    f'Max: {best_reward:.1f}'
                ]
                
                env.set_msgs(debug_info)
                env.render()
                pygame.display.update()
            
            if step % TRAINING_FREQUENCY == 0:
                loss = agent.train()
                if loss > 0:
                    episode_loss += loss
                    loss_count += 1
            
            if done:
                for _ in range(10):
                    agent.train()
                break
        
        # Update statistics
        episode_lengths.append(step + 1)
        rewards.append(total_reward)
        checkpoint_counts.append(max_checkpoint)
        
        if loss_count > 0:
            losses.append(episode_loss / loss_count)
        else:
            losses.append(0)
        
        if episode % UPDATE_TARGET_EVERY == 0:
            agent.update_target()
        
        agent.update_epsilon()
        
        episode_time = time.time() - episode_start_time
        total_time = time.time() - start_time
        
        if episode_actions:
            episode_action_dist = np.zeros(action_size)
            for a in episode_actions:
                episode_action_dist[a] += 1
            episode_action_dist = episode_action_dist / len(episode_actions)
            
            action_str = ' '.join([f"{i}:{p:.2f}" for i, p in enumerate(episode_action_dist)])
        else:
            action_str = "No actions"
        
        if episode % 10 == 0:
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            print(f"Episode {episode}: reward={total_reward:.1f}, avg={avg_reward:.1f}, ε={agent.epsilon:.4f}, steps={step+1}, ckpt={max_checkpoint}, time={episode_time:.2f}s")
            print(f"Actions: {action_str}")
        
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(f'models_{VERSION_NAME}/best.pt')
            print(f"New best model: {best_reward:.1f}")
        
        # Save progress periodically
        if episode % SAVE_EPISODES == 0 and episode > 0:
            agent.save(f'models_{VERSION_NAME}/latest.pt')
            
            progress = {
                'episode': episode + 1,
                'rewards': rewards,
                'losses': losses,
                'best_reward': best_reward,
                'action_distribution': action_distribution,
                'episode_lengths': episode_lengths,
                'checkpoint_counts': checkpoint_counts
            }
            np.save(f'models_{VERSION_NAME}/progress.npy', progress)
            
            if len(rewards) > 10:
                plt.figure(figsize=(15, 10))
                
                plt.subplot(2, 2, 1)
                plt.plot(rewards)
                plt.title(f'Rewards - Episode {episode}')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                
                plt.subplot(2, 2, 2)
                plt.plot(losses)
                plt.title('Average Loss per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Loss')
                
                plt.subplot(2, 2, 3)
                plt.plot(episode_lengths)
                plt.title('Episode Lengths')
                plt.xlabel('Episode')
                plt.ylabel('Steps')
                
                plt.subplot(2, 2, 4)
                plt.bar(range(action_size), action_distribution)
                plt.title('Action Distribution')
                plt.xlabel('Action')
                plt.ylabel('Count')
                
                plt.tight_layout()
                plt.savefig(f'models_{VERSION_NAME}/training_plot.png')
                plt.close()
            
            total_time_min = total_time / 60
            avg_100 = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            print(f"\nTraining Summary - Episode {episode}")
            print(f"Best reward: {best_reward:.1f}")
            print(f"Average reward (last 100): {avg_100:.1f}")
            print(f"Average episode length: {np.mean(episode_lengths[-100:]):.1f}")
            print(f"Maximum checkpoint reached: {max(checkpoint_counts[-100:] if checkpoint_counts else [0])}")
            print(f"Training time: {total_time_min:.1f} minutes")
    
    agent.save(f'models_{VERSION_NAME}/final.pt')
    return agent, rewards

# Run evaluation with a trained agent
def test(model_path=None):
    pygame.init()
    
    try:
        env = gym.make("Pyrace-v3").unwrapped
        print("Using enhanced Pyrace-v3 environment for testing")
    except Exception as e:
        print(f"Error loading Pyrace-v3: {e}")
        env = gym.make("Pyrace-v1").unwrapped
        print("Using Pyrace-v1 environment for testing")
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = EnhancedDQNAgent(state_size, action_size)
    
    if model_path is None:
        model_path = f'models_{VERSION_NAME}/best.pt'
    
    if not agent.load(model_path):
        print("Could not load model for testing")
        return
    
    agent.epsilon = 0.01
    
    pygame.display.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("PyTorch DQN Racing - Testing")
    
    for episode in range(10):
        state, _ = env.reset()
        env.set_view(True)
        
        if hasattr(env, 'pyrace') and hasattr(env.pyrace, 'mode'):
            env.pyrace.mode = 2
        
        total_reward = 0
        steps = 0
        done = False
        
        pygame.event.pump()
        
        while not done and steps < MAX_STEPS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            action = agent.act(state, training=False)
            
            next_state, reward, done, _, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            debug_info = [
                'PyTorch DQN Testing',
                f'Episode: {episode}',
                f'Step: {steps}',
                f'Reward: {total_reward:.1f}',
                f'Check: {info.get("check", 0)}',
                f'Crash: {info.get("crash", False)}'
            ]
            env.set_msgs(debug_info)
            env.render()
            pygame.display.update()
            
            state = next_state
        
        print(f"Test episode {episode}: Reward = {total_reward:.1f}, Steps = {steps}")


if __name__ == "__main__":
    mode = 'test'
    
    try:
        if mode == 'train':
            agent, rewards = train()
        else:
            test()
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        pygame.quit()
        print("Program completed")