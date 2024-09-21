import time
import torch.optim as optim
import torch
import math
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from torch.nn.utils import clip_grad_norm_
import copy
import torch.nn.functional as F
from collections import deque, namedtuple
import random

# from dqn.buffer import ReplayBuffer


class ReplayBuffer:
    """用于存储经验元组的固定大小缓冲区。s."""

    def __init__(self, buffer_size, batch_size, device):
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["done", "state", "action", "reward", "next_state"],
        )

    def add(
        self,
        done: bool,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
    ):
        """为memory增添新的ex。"""
        e = self.experience(done, state, action, reward, next_state)
        self.memory.append(e)

    def sample(self):
        """从记memory中随机抽取一批经验"""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = (
            torch.from_numpy(np.stack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.stack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# ---*SAC*---
class Actor(nn.Module):
    """Actor Model."""

    def __init__(self, state_dim, action_num, hidden_size=32):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_num)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs

    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities

    def get_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities

    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Critic(nn.Module):
    """Critic Model."""

    def __init__(self, state_dim, action_num, hidden_size=32, seed=3):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_num)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))


class SAC(nn.Module):
    def __init__(
        self, state_dim, action_num, batch_size, buffer_size, beta, device, debug
    ):
        super(SAC, self).__init__()
        self.state_dim = state_dim
        self.action_num = action_num
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.debug = debug
        self.gamma = 0.96
        self.tau = 1e-2
        hidden_size = 256
        learning_rate = 1e-3
        self.clip_grad_param = 1

        self.target_entropy = -action_num  # -dim(A)
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)

        # Actor Network

        self.actor_local = Actor(state_dim, action_num, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=learning_rate
        )

        # Critic Network (w/ Target Network)

        self.critic1 = Critic(state_dim, action_num, hidden_size, 2).to(device)
        self.critic2 = Critic(state_dim, action_num, hidden_size, 1).to(device)

        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = Critic(state_dim, action_num, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_dim, action_num, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

    def get_action(self, state, epsilon):
        # state = torch.from_numpy(state).float().to(self.device)
        state = state.float().to(self.device)
        if epsilon > random.uniform(0, 1):  # [0 , 1)
            return random.randint(0, self.action_num - 1)

        with torch.no_grad():
            action = self.actor_local.get_det_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states)
        q2 = self.critic2(states)
        min_Q = torch.min(q1, q2)
        actor_loss = (action_probs * (alpha * log_pis - min_Q)).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi

    def learn(self, experiences, gamma, d=1):

        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(
            states, current_alpha.to(self.device)
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Compute alpha loss
        alpha_loss = -(
            self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (
                torch.min(Q_target1_next, Q_target2_next)
                - self.alpha.to(self.device) * log_pis
            )
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (
                gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1)
            )

        # Compute critic loss
        q1 = self.critic1(states).gather(1, actions.long())
        q2 = self.critic2(states).gather(1, actions.long())

        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        return critic1_loss.item() + critic2_loss.item()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_model(self, path):
        # 保存模型的 state_dict
        torch.save(
            {
                "actor_state_dict": self.actor_local.state_dict(),
                "critic1_state_dict": self.critic1.state_dict(),
                "critic2_state_dict": self.critic2.state_dict(),
                "critic1_target_state_dict": self.critic1_target.state_dict(),
                "critic2_target_state_dict": self.critic2_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
                "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha,
            },
            path,
        )
