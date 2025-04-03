import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_race.envs.pyrace_2d import PyRace2D

class RaceEnv(gym.Env):
    metadata = {'render_modes' : ['human'], 'render_fps' : 30}
    def __init__(self, render_mode="human", ):
        print("init")
        # Enhanced action space with 7 possible actions instead of 3
        self.action_space = spaces.Discrete(7)
        
        # Enhanced observation space:
        # 9 radar readings + speed + angle + checkpoint distance
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0]), 
            high=np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1, 1]),
            dtype=np.float32
        )
        
        self.is_view = True
        self.pyrace = PyRace2D(self.is_view)
        self.memory = []
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        mode = self.pyrace.mode
        del self.pyrace
        self.is_view = True
        self.msgs=[]
        self.pyrace = PyRace2D(self.is_view, mode = self.render_mode)
        obs = self.pyrace.observe()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done = self.pyrace.is_done()
        obs = self.pyrace.observe()
        
        # Convert observation to numpy array with correct dtype
        obs_array = np.array(obs, dtype=np.float32)
        
        return obs_array, reward, done, False, {
            'dist': self.pyrace.car.distance, 
            'check': self.pyrace.car.current_check, 
            'crash': not self.pyrace.car.is_alive,
            'speed': self.pyrace.car.speed,
            'time': self.pyrace.car.time_spent
        }

    # def render(self, close=False , msgs=[], **kwargs): # gymnasium.render() does not accept other keyword arguments
    def render(self): # gymnasium.render() does not accept other keyword arguments
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs

    def save_memory(self, file):
        # print(self.memory) # heterogeneus types
        # np.save(file, self.memory)
        np.save(file, np.array(self.memory, dtype=object))
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
