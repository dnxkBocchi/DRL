import time
import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
from transformer import MemTransformer
from model import StateEncoder
from torch.nn.utils import clip_grad_norm_
import copy
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


class Memory:
    def __init__(self):
        self.ts = []
        self.actions = []
        self.rep_states = []
        self.images = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.ts[:]
        del self.actions[:]
        del self.rep_states[:]
        del self.images[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class StateRepresentation(nn.Module):
    def __init__(self, H):
        super(StateRepresentation, self).__init__()

        self.H = H
        self.state_rep = H.state_rep
        self.action_dim = H.action_dim
        self.device = H.device

        # inp_dim = H.n_latent_var + H.action_dim + 1 # 动作！！ current state, previous action and reward
        inp_dim = H.n_latent_var  # + H.action_dim + 1
        out_dim = H.n_latent_var

        self.resnet = StateEncoder(
            input_state_dim=H.input_state_dim, img_enc_dim=H.n_latent_var
        )

        if H.state_rep == "lstm":
            self.layer = nn.LSTMCell(inp_dim, out_dim)
            self.h0 = nn.Parameter(torch.rand(H.n_latent_var))
            self.c0 = nn.Parameter(torch.rand(H.n_latent_var))
        elif H.state_rep == "trxl":
            self.layer = MemTransformer(
                inp_dim,
                n_layer=H.n_layer,
                n_head=H.n_head,
                d_model=H.n_latent_var,
                d_head=H.n_latent_var // H.n_head,
                d_inner=H.n_latent_var,
                dropout=H.dropout,
                dropatt=H.dropout,
                pre_lnorm=True,
                tgt_len=1,
                ext_len=0,
                mem_len=H.mem_len,
                attn_type=0,
            )
        elif H.state_rep == "gtrxl":
            raise NotImplemented

        self.init_action = nn.Parameter(torch.rand(H.action_dim))  # 动作！！
        self.init_reward = nn.Parameter(torch.rand(1))

    def forward(self, t, img, _prev_action=None, _prev_reward=None):

        # img = torch.from_numpy(img).float().to(self.device).unsqueeze(0).permute(0,3,1,2)
        # print(img.dtype)

        if type(img) is not np.ndarray:
            img = np.array(img)
        img = torch.from_numpy(img.astype(float))
        img = img.float().to(self.device)
        state = self.resnet(img).squeeze()  # state为一维的tensor//////////////////
        if self.state_rep == "none":
            return state
        if t == 0:
            prev_action = self.init_action
            prev_reward = self.init_reward
            if self.state_rep == "lstm":
                self.h = self.h0.unsqueeze(0)
                self.c = self.c0.unsqueeze(0)
            if self.state_rep in ["trxl", "gtrxl"]:
                self.inputs = []
                self.mems = tuple()
        else:
            prev_action = torch.zeros(self.action_dim).to(self.device)
            prev_action[_prev_action] = 1
            prev_reward = (
                torch.from_numpy(np.array([_prev_reward])).float().to(self.device)
            )

        # [1, inp_dim]
        # inp = torch.cat([state, prev_action, prev_reward], dim=0)

        inp = torch.cat([state], dim=0)
        inp = inp.unsqueeze(0)

        if self.state_rep == "lstm":
            h, c = self.layer(inp, (self.h, self.c))
            self.h = h
            self.c = c
            return h[0]
        elif self.state_rep in ["trxl", "gtrxl"]:
            self.inputs.append(inp)
            _inputs = torch.stack(self.inputs, dim=0)
            pred, _mems = self.layer(_inputs, *self.mems)

            # print(_inputs.shape)
            # print(pred.shape)
            # print(type(_mems),len(_mems))
            # time.sleep(40)

            if t >= (1 + self.H.mem_len):
                self.mems = _mems

            # print('cds',len(pred[0][0]))
            # time.sleep(40)
            return pred[0][0]

        # 使用注意力对状态特征进行提取

    def batch_forward(self, ts, images, actions, rewards):

        if self.state_rep == "none":
            if type(images) is not np.ndarray:
                images = np.array(images)
            # images = torch.from_numpy(images).float().to(self.device).permute(0,3,1,2)
            images = torch.from_numpy(images).float().to(self.device)

            rep_states = self.resnet(images).squeeze()
        else:
            rep_states = []
            for i in range(len(ts)):
                t = ts[i]
                image = images[i]
                action = actions[i]
                reward = rewards[i]

                if i == 0:
                    prev_action = None
                    prev_reward = None
                else:
                    prev_action = actions[i - 1]
                    prev_reward = rewards[i - 1]

                if t == 0:
                    rep_states.append(self.forward(t, image))
                else:
                    rep_states.append(self.forward(t, image, prev_action, prev_reward))

            rep_states = torch.stack(rep_states, dim=0)

        return rep_states


class ActorCritic(nn.Module):
    def __init__(self, model, H):
        super(ActorCritic, self).__init__()

        self.model = model
        self.device = H.device
        self.state_rep = H.state_rep
        inp_dim = H.n_latent_var  # 输入的大小

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(inp_dim, H.action_dim), nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(inp_dim, H.n_latent_var),
            nn.Tanh(),
            nn.Linear(H.n_latent_var, H.n_latent_var),
            nn.Tanh(),
            nn.Linear(H.n_latent_var, 1),
        )

        # shared state representation module
        self.shared_layer = StateRepresentation(H)  # 替换成状态特征类

    def forward(self):
        raise NotImplementedError

    def act(self, t, img, memory):

        if t == 0:
            rep_state = self.shared_layer(t, img)  # 看一下他这里的状态返回是什么
        else:

            rep_state = self.shared_layer(
                t, img, memory.actions[-1], memory.rewards[-1]
            )

        action_probs = self.action_layer(
            rep_state
        )  # dim=inp_dim = H.n_latent_var:rep_state的大小

        dist = Categorical(action_probs)
        action = dist.sample()  # action输出是什么-->怎么映射到我们那个动作

        memory.ts.append(t)
        memory.images.append(img)
        memory.rep_states.append(rep_state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, ts, images, actions, rewards):

        rep_states = self.shared_layer.batch_forward(
            ts, images, actions, rewards
        )  # 更新状态函数估计

        action_probs = self.action_layer(rep_states.detach())
        # print(action_probs)
        # print(action_probs.shape)

        dist = Categorical(action_probs)
        # print('distshape',dist)
        # print('actions',actions)

        action_logprobs = dist.log_prob(actions)

        # print(action_logprobs)
        # print(action_logprobs.shape)
        # time.sleep(40)

        if self.model == "ppo":
            dist_entropy = dist.entropy()
        elif self.model == "vmpo":
            dist_probs = dist.probs

        state_value = self.value_layer(rep_states)  # 输入大小一致

        if self.model == "ppo":
            return action_logprobs, torch.squeeze(state_value), dist_entropy
        elif self.model == "vmpo":
            return action_logprobs, torch.squeeze(state_value), dist_probs


class PPO:
    def __init__(self, H):
        self.lr = H.lr
        self.betas = H.betas
        self.gamma = H.gamma
        self.eps_clip = H.eps_clip
        self.K_epochs = H.K_epochs
        self.device = H.device

        self.policy = ActorCritic("ppo", H).to(H.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=H.lr, betas=H.betas
        )
        self.policy_old = ActorCritic("ppo", H).to(H.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list to tensor
        old_ts = memory.ts
        old_images = memory.images
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_rewards = memory.rewards
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        # Get old probs and old advantages
        with torch.no_grad():
            _, old_state_values, old_dist_probs = self.policy_old.evaluate(
                old_ts, old_images, old_actions, old_rewards
            )
            advantages = rewards - old_state_values.detach()

        # Optimize policy for K epochs:
        for i in range(self.K_epochs):
            # Evaluating sampled actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_ts, old_images, old_actions, old_rewards
            )

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            loss = (
                -torch.min(surr1, surr2).unsqueeze(-1)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
