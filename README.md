# PyRace DQN - Reinforcement Learning for Racing

This project implements a Deep Q-Network (DQN) agent to play a 2D racing game using PyTorch. The agent learns to navigate a race track, avoid walls, and complete laps as quickly as possible.

## Features

- **DQN implementation**:
  - Dueling DQN architecture (separate value and advantage streams)
  - Double DQN for more stable learning
  - Prioritized Experience Replay for efficient learning
  - Smart exploration strategy that balances random exploration with targeted exploration

- **Environment**:
  - 9 radar sensors
  - action space with fine-grained control
  - car state information
  - Reward system for speed and completion

## Installation

1. Clone the repository

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure the gym-race environment is installed and accessible

## Usage

### Training a model

To train a new agent:

1. Edit `Pyrace_RL_DQN.py` and set `mode = 'train'` in the main section at the end
2. Run:
   ```
   python Pyrace_RL_DQN.py
   ```

The training process will:
- Save the best model to `models_DQN_pytorch/best.pt`
- Save periodic checkpoints to `models_DQN_pytorch/latest.pt`
- Generate training plots every 100 episodes

### Testing a trained model

To test your trained agent:

1. Edit `Pyrace_RL_DQN.py` and set `mode = 'test'` in the main section
2. Run:
   ```
   python Pyrace_RL_DQN.py
   ```

By default, running `Pyrace_RL_DQN.py` will run my best model. Unless you start training yourself.

## Hyperparameters

There important hyperparameters that can be adjusted at the top of the script:

- `LEARNING_RATE`: Controls how quickly the network adapts (default: 0.0001)
- `GAMMA`: Discount factor for future rewards (default: 0.99)
- `EPSILON_START`: Initial exploration rate (default: 0.3)
- `EPSILON_MIN`: Minimum exploration rate (default: 0.05)
- `BATCH_SIZE`: Number of experiences to learn from at once (default: 32)

## Project Structure

- `Pyrace_RL_DQN.py`: Main DQN script
- `gym_race/`: The racing environment
  - `envs/pyrace_2d.py`: Core racing engine
  - `envs/race_env.py`: Gym environment wrapper

## License

MIT License