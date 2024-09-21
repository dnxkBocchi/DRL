from model.sac import ReplayBuffer

import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from env.task import TaskStatus


class Scheduler:
    def __init__(
        self,
        agent,
        action_num: int,
        state_dim: int,
        buffer_size: int,
        batch_size: int,
        discount_factor: float = 0.9,
        learning_rate: float = 1e-4,
        l2_reg: float = 0,
        constant_df: bool = True,
        df2: float = 0,
        next_q: bool = True,
        reward_num: int = 1,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):

        self.constant_df = constant_df
        self.action_num = action_num
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.agent = agent
        self.buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.agent.device,
        )

        self.beta = beta
        self.losses = []
        self.rewards = []
        self.mean_losses = []
        self.mean_rewards = []
        self.step_counter = 0
        self.time_counter = 0
        self.update_counter = 0
        self.target_update = 100
        self.transition = list()
        self.epsilons = []
        self.epsilon_decay = 5e-4
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon = 1

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
        r = (1 - self.beta) * cost_r + self.beta * time_r
        return r

    def update_epsilon(self):
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(
            -1.0 * ((self.update_counter + 1) * self.step_counter) * self.epsilon_decay
        )

    def trainPlot(self):
        mean = sum(self.losses) / len(self.losses)
        self.mean_losses.append(mean)
        self.losses = []

        mean = sum(self.rewards) / len(self.rewards)
        self.mean_rewards.append(mean)
        self.rewards = []

        if len(self.mean_rewards) % 50 == 0:
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

    def schedule(
        self,
        last_part,
        task,
        vm_list,
        now_time,
        done,
    ):
        state = self.createState(task, vm_list, now_time)
        action = self.agent.get_action(state, self.epsilon)

        if self.agent.actor_local.train():
            if not last_part:
                return vm_list[action]
            r = self.reward1(state, action, task)
            self.rewards.append(r)
            if self.transition:
                self.transition += [state]
                self.buffer.add(*self.transition)
            self.transition = [done, state, action, r]
            self.time_counter += 1
            if len(self.buffer) >= self.batch_size:
                loss = self.agent.learn(self.buffer.sample(), gamma=0.96)
                self.losses.append(loss)
                self.step_counter += 1
                self.update_epsilon()
                if self.step_counter == self.target_update:
                    self.step_counter = 0
                    self.update_counter += 1
                    self.epsilons.append(self.epsilon)
                    self.trainPlot()

        return vm_list[action]
