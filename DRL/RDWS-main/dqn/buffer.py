from typing import Dict

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, batch_size: int = 32):
        self.state_buf = np.zeros([capacity, state_dim], dtype=np.float32)
        self.next_state_buf = np.zeros([capacity, state_dim], dtype=np.float32)
        self.action_buf = np.zeros([capacity], dtype=np.int8)
        self.reward_buf = np.zeros([capacity], dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        #--------------------
        self.delta_buf = np.zeros([capacity], dtype=np.float32)
        self.delta2_buf = np.zeros([capacity], dtype=np.float32)
        # self.is_running_buf = np.zeros(capacity, dtype=np.float32)
        # self.is_child_buf = np.zeros(capacity, dtype=np.float32)
        # self.parent_num_buf = np.zeros(capacity, dtype=np.int)
        # self.reward2_buf = np.zeros([capacity], dtype=np.float32)

        self.batch_size = batch_size
        self.capacity = capacity
        self.index = 0
        self.buffer_size = 0

    def store(
        self,
        done: bool,
        state: np.ndarray,
        action: int, 
        reward: float, 
        next_state: np.ndarray,
        delta: float,
        delta2: float,
        # parent_num: int,
        # is_running: bool,
        # is_child: bool,
        # reward2: float = 0,        
    ):
        self.state_buf[self.index] = state
        self.next_state_buf[self.index] = next_state
        self.action_buf[self.index] = action
        self.reward_buf[self.index] = reward
        self.done_buf[self.index] = done
        self.delta2_buf[self.index] = delta2
        self.delta_buf[self.index] = delta

        # self.is_running_buf[self.index] = is_running
        # self.is_child_buf[self.index] = is_child
        # self.parent_num_buf[self.index] = parent_num
        # self.reward2_buf[self.index] = reward2


        self.index = (self.index + 1) % self.capacity
        self.buffer_size = min(self.buffer_size + 1, self.capacity)

    def sample(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.buffer_size, size=self.batch_size, replace=False)
        return dict(states=self.state_buf[idxs],
                    next_states=self.next_state_buf[idxs],
                    actions=self.action_buf[idxs],
                    rewards=self.reward_buf[idxs],
                    done=self.done_buf[idxs],
                    deltas = self.delta_buf[idxs],
                    delta2s = self.delta2_buf[idxs],
                    #---------------------
                    # is_child = self.is_child_buf[idxs],
                    # parent_nums = self.parent_num_buf[idxs],
                    # is_running = self.is_running_buf[idxs],
                    # rewards2=self.reward2_buf[idxs],
                    )

    def __len__(self) -> int:
        return self.buffer_size
