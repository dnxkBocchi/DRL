import random
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from env.task import TaskStatus


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, batch_size: int = 32):
        self.state_buf = np.zeros([capacity, state_dim], dtype=np.float32)
        self.next_state_buf = np.zeros([capacity, state_dim], dtype=np.float32)
        self.action_buf = np.zeros([capacity], dtype=np.int8)
        self.reward_buf = np.zeros([capacity], dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)

        self.batch_size = batch_size
        self.capacity = capacity
        self.index = 0
        self.buffer_size = 0

    def add(
        self,
        done: bool,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
    ):
        self.state_buf[self.index] = state.cpu() if state.is_cuda else state
        self.next_state_buf[self.index] = (
            next_state.cpu() if next_state.is_cuda else next_state
        )
        self.action_buf[self.index] = action
        self.reward_buf[self.index] = reward
        self.done_buf[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.buffer_size = min(self.buffer_size + 1, self.capacity)

    def sample(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.buffer_size, size=self.batch_size, replace=False)
        return dict(
            states=self.state_buf[idxs],
            next_states=self.next_state_buf[idxs],
            actions=self.action_buf[idxs],
            rewards=self.reward_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.buffer_size


class Network(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, device):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.Tanh(),  # ReLU
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),  # ReLU
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),  # ReLU
            nn.Linear(hidden_size, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)


class DQN:
    def __init__(
        self,
        state_dim,
        action_num,
        batch_size,
        buffer_size,
        hidden_size,
        lr,
        device,
        discount_factor,
        l2_reg: float = 0,
    ):

        self.action_num = action_num
        self.state_dim = state_dim
        self.buffer = ReplayBuffer(buffer_size, state_dim, batch_size)
        self.batch_size = batch_size
        # 折扣因子γ，用于平衡未来奖励与当前奖励的重要性
        self.discount_factor = discount_factor
        self.device = device

        self.net = Network(state_dim, action_num, hidden_size, self.device).to(
            self.device
        )
        self.dqn_target_net = Network(
            state_dim, action_num, hidden_size, self.device
        ).to(self.device)
        # 将主网络 net 的权重复制到目标网络 dqn_target_net
        self.dqn_target_net.load_state_dict(self.net.state_dict())
        # 将目标网络设置为评估模式
        self.dqn_target_net.eval()

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=l2_reg
        )
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.95, last_epoch=-1, verbose=True);

    def get_action(self, state, epsilon):
        # epsilon greedy policy
        # if self.net.training and self.epsilon > random.uniform(0, 1):  # [0 , 1)
        #     return random.randint(0, self.action_num - 1)
        if self.net.training and epsilon > random.uniform(0, 1):  # [0 , 1)
            return random.randint(0, self.action_num - 1)
        else:
            state = state.to(self.device)  # 将状态移动到同一设备
            action = self.net(state).argmax().detach().cpu().item()

            return action

    def updateModel(self, samples):
        loss = self.computeLoss(samples)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def computeLoss(self, samples):
        state_batch = torch.FloatTensor(samples["states"]).to(self.device)
        next_state_batch = torch.FloatTensor(samples["next_states"]).to(self.device)
        action_index = torch.LongTensor(samples["actions"].reshape(-1, 1)).to(
            self.device
        )
        reward = torch.FloatTensor(samples["rewards"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        # gather 操作，基于智能体在每个状态下实际选择的动作（由 action_index 指定），
        # 提取出这些动作对应的 Q 值，存储在 curr_q_value 中。
        curr_q_value = self.net(state_batch).gather(1, action_index)

        # DQN
        # 用于估计从当前状态转移到下一个状态后的预期回报，并根据当前奖励和未来回报更新 Q 值。
        # 折扣因子，决定了未来奖励的权重
        if self.discount_factor:
            next_q_value = (
                self.dqn_target_net(next_state_batch)
                .max(dim=1, keepdim=True)[0]
                .detach()
            )

            # Double DQN 改进: 使用主网络选择动作
            # next_action = self.net(next_state_batch).argmax(dim=1, keepdim=True)
            # next_q_value = (
            #     self.dqn_target_net(next_state_batch).gather(1, next_action).detach()
            # )
            target = (reward + self.discount_factor * next_q_value * (1 - done)).to(
                self.device
            )
        # 算法只关心当前的奖励，不考虑未来的回报。
        else:
            target = (reward).to(self.device)
        # loss = F.mse_loss(curr_q_value, target)
        loss = F.smooth_l1_loss(curr_q_value, target)
        return loss

    def learn(self, example, gamma):
        return self.updateModel(example)

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path))
