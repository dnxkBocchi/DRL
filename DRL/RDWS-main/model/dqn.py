import datetime
import math
import pickle
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

    def store(
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
    def __init__(self, in_dim: int, out_dim: int, device):
        """Initialization."""
        super(Network, self).__init__()
        dim = 256

        # 256 128 256 BAD
        # 128 128 128 BAD

        self.layers = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.Linear(dim, dim),
            # nn.LayerNorm(dim),
            # nn.Dropout(p=0.2),
            nn.Tanh(),  # ReLU
            nn.Linear(dim, dim),
            # nn.LayerNorm(dim),
            # nn.Dropout(p=0.2),
            nn.Tanh(),  # ReLU
            nn.Linear(dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)


class DQNScheduler:
    def __init__(
        self,
        action_num: int,
        state_dim: int,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float = 5e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        discount_factor: float = 0.9,
        learning_rate: float = 1e-4,
        l2_reg: float = 0,
        next_q: bool = True,
        reward_num: int = 1,
        alpha: float = 0.5,
    ):

        self.action_num = action_num
        self.next_q = next_q
        self.state_dim = state_dim

        self.memory = ReplayBuffer(memory_size, state_dim, batch_size)
        # 只是将函数 schedule1 本身（即函数对象）赋值给 self.schedule，并没有执行这个函数

        self.batch_size = batch_size
        # ε-贪婪策略中ε的起始值和最终值。
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        # ε-贪婪策略中ε衰减的速率，控制探索与利用的平衡
        self.epsilon_decay = epsilon_decay
        # 更新目标网络的频率
        self.target_update = target_update
        # 折扣因子γ，用于平衡未来奖励与当前奖励的重要性
        self.discount_factor = discount_factor

        self.losses = []
        self.mean_losses = []
        self.mean_rewards = []
        self.all_rewards = []
        self.all_losses = []
        self.rewards = []
        self.epsilons = []
        self.update = []
        self.step_counter = 0
        self.update_counter = 0
        self.epsilon = 1
        self.transition = list()
        self.last_time = 0
        self.last_task = None
        self.makespan = []
        self.cost = []
        self.time_rate = []
        self.cost_rate = []
        self.succes_both_rate = []
        self.episode = []

        self.episode_counter = 0
        self.abc = 0

        self.df = torch.zeros([batch_size, 1])
        self.next_reward = torch.zeros([batch_size, 1])

        # device: cpu / gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("DQN: device is", self.device)

        self.dqn_net = Network(state_dim, action_num, self.device).to(self.device)
        self.dqn_target_net = Network(state_dim, action_num, self.device).to(
            self.device
        )
        # 将主网络 dqn_net 的权重复制到目标网络 dqn_target_net
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())
        # 将目标网络设置为评估模式
        self.dqn_target_net.eval()

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.dqn_net.parameters(), lr=learning_rate, weight_decay=l2_reg
        )
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.95, last_epoch=-1, verbose=True);
        self.alpha = alpha
        self.reward_num = reward_num
        if reward_num == 1:
            self.reward = self.reward1
        elif reward_num == 2:
            self.reward = self.reward2

    # 根据当前的 state 和采取的 action 计算成本奖励 cost_r。
    # 它考虑了预算 (budget)、截至日期 (deadline)、开始时间 (bft)、结束时间 (lft)、时间和成本等因素。
    def costReward(self, state, action):
        budget = state[0]
        cost = state[6 + self.action_num : 6 + self.action_num * 2]

        if cost[action] == 0:
            cost_r = 1
        elif cost[action] <= budget:
            if budget != min(cost):
                cost_r = (budget - cost[action]) / (budget - min(cost))
                cost_r = cost_r.item()
            else:
                cost_r = 1  # (min(cost) - cost[action])/(max(cost) - min(cost))
        else:
            if max(cost) != budget:
                cost_r = (budget - cost[action]) / (max(cost) - budget)
                cost_r = cost_r.item()
            else:
                cost_r = -1  # (max(cost) - cost[action])/(max(cost) - min(cost))

        return cost_r

    def timeReward(self, state, action):
        deadline = state[1]
        time = state[6 : 6 + self.action_num]

        if time[action] <= deadline:
            if deadline != min(time):
                time_r = (deadline - time[action]) / (deadline - min(time))
                time_r = time_r.item()
            else:
                time_r = 1  # (min(time) - time[action])/(max(time) - min(time))
        else:
            if max(time) != deadline:
                time_r = (deadline - time[action]) / (max(time) - deadline)
                time_r = time_r.item()
            else:
                time_r = -1  # (max(time) - time[action])/(max(time) - min(time))
        return time_r

    def reward1(self, state, action, task):
        cost_r = self.costReward(state, action)
        time_r = self.timeReward(state, action)
        r = (1 - self.alpha) * cost_r + self.alpha * time_r
        return r

    def reward2(self, state, action, task):
        cost_r = self.costReward(state, action)
        time_r = self.timeReward(state, action)

        if time_r <= 0:
            if cost_r <= 0:
                r = (1 - self.alpha) * cost_r + self.alpha * time_r
            else:
                r = self.alpha * time_r
        else:
            if cost_r <= 0:
                r = (1 - self.alpha) * cost_r
            else:
                r = (1 - self.alpha) * cost_r + self.alpha * time_r
        return r

    def createState(self, task, vm_list, now_time):
        # 这个状态向量会捕捉任务和虚拟机的多个特征
        x = torch.zeros(self.state_dim, dtype=torch.float)
        # ???
        if len(task.succ) == 0:
            return x

        # 计算最大值,任务预算、执行时间和成本
        budget = max((task.workflow.budget - task.workflow.cost), 0)
        index = 0
        t = []
        c = []
        u = []

        for v in vm_list:
            t.append(task.vref_time_cost[v][0])
            c.append(task.vref_time_cost[v][1])

        for child in task.succ:
            u.append(child.uprank)

        max_t = max(t + [task.deadline, task.BFT, task.LFT])
        max_c = max(c + [budget])
        max_u = max(u)

        # 通过一系列归一化的操作，将任务、虚拟机和工作流的特征信息填入状态向量 x
        x[index] = (
            budget / max_c if max_c else 0
        )  # /task.workflow.remained_length)*task.rank_exe
        index += 1
        x[index] = task.deadline / max_t
        index += 1
        x[index] = task.BFT / max_t
        index += 1
        x[index] = task.LFT / max_t
        index += 1
        x[index] = (max_u) / (task.workflow.entry_task.uprank)
        index += 1

        x[index] = (
            len(task.workflow.tasks) - 2 - len(task.workflow.finished_tasks)
        ) / (len(task.workflow.tasks) - 2)
        index += 1

        for v in vm_list:
            x[index] = task.vref_time_cost[v][0] / max_t
            index += 1
        for v in vm_list:
            x[index] = task.vref_time_cost[v][1] / max_c if max_c else 0
            index += 1
        for v in vm_list:
            x[index] = max(
                max(task.workflow.deadline - now_time, 0) - task.vref_time_cost[v][0], 0
            ) / (task.workflow.deadline)
            index += 1

        # for v in vm_list:
        #     x[index] = v.unfinished_tasks_number
        #     index += 1;
        # for v in vm_list:
        #     x[index] = 1 if v.isVMType() else 0;
        #     index += 1;

        # x[index] = 1 if len(ready_queue) else 0;
        # x[index] = len(ready_queue)/all_task_num;
        # index += 1;

        # print(x)
        return x

    def schedule(
        self,
        last_part,
        task,
        vm_list,
        now_time,
        done,
    ):
        # vm_list.sort(key=lambda x: task.vref_time_cost[v][0])
        # 根据当前任务、虚拟机列表、队列、剩余任务、总任务数和当前时间创建当前的状态。
        state = self.createState(task, vm_list, now_time).to(self.device)
        action = self.selectAction(state, now_time, task.id)

        if self.dqn_net.training:
            # 如果 last_part 为 False，说明当前任务还没有结束或还在处理中。
            # 不需要执行后续的奖励计算和存储操作。因为奖励的计算往往是基于任务的完成状态，
            # 只有在任务完成时，才会根据其执行效果（如时间、资源使用等）给予奖励。
            if not last_part:
                return vm_list[action]

            r = self.reward(state, action, task)

            # 如果存在先前的过渡信息（状态-动作-奖励序列），则计算时间差 delta 和 delta2，
            # 并将这些信息和新的状态、奖励存储在经验回放缓冲区
            if self.transition:
                delta = now_time - self.last_time
                if delta < 0:
                    print("*error*", now_time, self.last_task.estimate_finish_time)
                if self.last_task.status == TaskStatus.done:
                    delta2 = self.last_task.finish_time - now_time
                    # -
                else:
                    delta2 = self.last_task.estimate_finish_time - now_time
                    # +

                self.transition += [
                    state,
                    # task in self.last_task.succ,
                    # len(task.pred),
                    # self.last_task.status != TaskStatus.done ,
                    # r
                ]
                self.memory.store(*self.transition)

            self.rewards.append(r)
            self.all_rewards.append(r)
            self.update.append(self.episode_counter)

            self.transition = [done, state, action, r]
            self.train()

            self.last_time = now_time
            self.last_task = task

            if done:
                self.transition += [state]
                self.transition[0] = done
                self.memory.store(*self.transition)
                self.transition = []
                self.last_time = 0
                self.last_task = None
                self.episode_counter += 1

        return vm_list[action]

    def train(self):
        if len(self.memory) >= self.batch_size:
            loss = self.updateModel()
            self.losses.append(loss)
            self.all_losses.append(loss)

            self.step_counter += 1

            # linearly decrease epsilon
            # self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay);

            # exponentialy decrease epsilon
            self.epsilon = self.epsilon_end + (
                self.epsilon_start - self.epsilon_end
            ) * math.exp(
                -1.0
                * ((self.update_counter + 1) * self.step_counter)
                * self.epsilon_decay
            )

            if self.step_counter == self.target_update:
                # self.lr_scheduler.step();
                self.step_counter = 0
                self.update_counter += 1
                self.epsilons.append(self.epsilon)
                self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())
                self.trainPlot()

    def selectAction(self, state, now_time=0, tid=0, last_part=True):
        # epsilon greedy policy
        if self.dqn_net.training and self.epsilon > random.uniform(0, 1):  # [0 , 1)
            return random.randint(0, self.action_num - 1)
        else:
            # action = self.dqn_net(state).argmax().detach().cpu().item()
            action = self.dqn_net(state).argmax().item()
            # if not self.dqn_net.training and last_part:
            #     budget = state[0]
            #     deadline = state[1]
            #     bft = state[2]
            #     lft = state[3]
            #     time = state[9:15]
            #     cost = state[15:21]
            #     print(deadline.item(), budget, action, "----------------------------------------------")
            #     print(time)
            #     print(cost)

            #     print(now_time, tid,"action", action)
            #     print(state)
            #     print("---------------------------------------------------------------------------------")

            return action

    def updateModel(self):
        samples = self.memory.sample()
        loss = self.computeLoss(samples)
        self.optimizer.zero_grad()
        loss.backward()

        # DQN gradient clipping:
        # for param in self.dqn_net.parameters():
        #     print(param.grad.data)
        #     # param.grad.data.clamp_(-1, 1);
        #     # print(param.grad.data)
        # print("===================================================")
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
        curr_q_value = self.dqn_net(state_batch).gather(1, action_index)

        # # DQN
        # 用于估计从当前状态转移到下一个状态后的预期回报，并根据当前奖励和未来回报更新 Q 值。
        # 折扣因子，决定了未来奖励的权重
        if self.discount_factor:
            next_q_value = (
                self.dqn_target_net(next_state_batch)
                .max(dim=1, keepdim=True)[0]
                .detach()
            )
            target = (reward + self.discount_factor * next_q_value * (1 - done)).to(
                self.device
            )

        # 如果没有折扣因子（即 γ = 0），目标 Q 值就只等于即时奖励 reward。
        # 这种情况下，算法只关心当前的奖励，不考虑未来的回报。
        else:
            target = (reward).to(self.device)

        loss = F.mse_loss(curr_q_value, target)
        # smooth_l1_loss   mse_loss
        return loss

    def trainPlot(self):
        mean = sum(self.losses) / len(self.losses)
        self.mean_losses.append(mean)
        self.losses = []

        mean = sum(self.rewards) / len(self.rewards)
        self.mean_rewards.append(mean)
        self.rewards = []

        if len(self.mean_losses) == 19 or len(self.mean_losses) % 50 == 0:
            print("pictures")
            plt.plot(self.mean_losses, "-o", linewidth=1, markersize=2)
            # plt.xlabel(str(self.target_update * len(self.mean_losses)) + "iterations")
            plt.xlabel(str(len(self.mean_losses)) + "iterations")
            plt.ylabel("Mean Losses")
            plt.show()

            plt.plot(self.epsilons, linewidth=1)
            # plt.title("epsilons");
            plt.ylabel("Epsilon")
            plt.show()

            plt.plot(self.mean_rewards, "-o", linewidth=1, markersize=2)
            # plt.xlabel(str(self.target_update * len(self.mean_rewards)) + "iterations")
            plt.xlabel(str(len(self.mean_rewards)) + "iterations")
            plt.ylabel("Mean Rewards")
            plt.show()

    def trainSave(
        self,
        more_text="",
        mean_makespan=[],
        mean_cost=[],
        succes_deadline_rate=[],
        succes_budget_rate=[],
        succes_both_rate=[],
    ):
        time_str = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M "))
        print("final pictures")

        plt.plot(self.mean_losses, "-o", linewidth=1, markersize=2)
        plt.xlabel(str(len(self.mean_losses)) + "iterations")
        plt.ylabel("Mean Losses")
        plt.savefig("logs/" + time_str + "_loss.png", facecolor="w")
        # transparent=False
        plt.show()
        plt.clf()

        plt.plot(self.epsilons, linewidth=1)
        # plt.title("epsilons");
        plt.ylabel("Epsilon")
        plt.xlabel(str(len(self.mean_rewards)) + "iterations")
        plt.savefig("logs/" + time_str + "_eps.png", facecolor="w")
        # transparent=False
        plt.show()
        plt.clf()

        plt.plot(self.mean_rewards, "-o", linewidth=1, markersize=2)
        plt.xlabel(str(len(self.mean_rewards)) + "iterations")
        plt.ylabel("Mean Rewards")
        plt.savefig("logs/" + time_str + "_reward.png", facecolor="w")
        # transparent=False
        plt.show()
        plt.clf()

        if mean_cost:
            self.cost += mean_cost
            plt.plot(mean_cost, "-o", linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Cost")
            plt.savefig("logs/" + time_str + "_cost.png", facecolor="w")
            # transparent=False
            plt.show()
            plt.clf()

        if mean_makespan:
            self.makespan += mean_makespan
            plt.plot(mean_makespan, "-o", linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Makespan")
            plt.savefig("logs/" + time_str + "_makespan.png", facecolor="w")
            # transparent=False
            plt.show()
            plt.clf()

        if succes_budget_rate:
            self.cost_rate += succes_budget_rate
            plt.plot(succes_budget_rate, "-o", linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Cost Rate")
            plt.savefig("logs/" + time_str + "_bsr.png", facecolor="w")
            # transparent=False
            plt.show()
            plt.clf()

        if succes_deadline_rate:
            self.time_rate += succes_deadline_rate
            plt.plot(succes_deadline_rate, "-o", linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Time Rate")
            plt.savefig("logs/" + time_str + "_dsr.png", facecolor="w")
            # transparent=False
            plt.show()
            plt.clf()

        if succes_both_rate:
            self.succes_both_rate += succes_both_rate
            plt.plot(succes_both_rate, "-o", linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Success Rate")
            plt.savefig("logs/" + time_str + "_both.png", facecolor="w")
            # transparent=False
            plt.show()
            plt.clf()

        file_name = "logs/" + str(self.reward_num) + "_" + str(self.alpha)
        with open(file_name, "wb") as f:
            pickle.dump(self, f)
