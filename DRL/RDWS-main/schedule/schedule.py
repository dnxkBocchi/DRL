import math
import datetime
import pickle

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from env.task import TaskStatus
from model.dqn import DQN


class Scheduler:
    def __init__(
        self,
        agent,
        action_num: int,
        state_dim: int,
        batch_size: int,
        beta: float = 0.5,
    ):

        self.action_num = action_num
        self.state_dim = state_dim
        self.batch_size = batch_size

        self.agent = agent
        self.buffer = agent.buffer

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
        x = torch.zeros(self.state_dim, dtype=torch.float)
        # # 如果任务没有后继任务，直接返回全零的状态向量
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
        x[index] = budget / max_c if max_c else 0
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

        # 任务在虚拟机上的时间花费占比
        for v in vm_list:
            x[index] = task.vref_time_cost[v][0] / max_t
            index += 1
        # 任务在每个虚拟机上计算成本的占比
        for v in vm_list:
            x[index] = task.vref_time_cost[v][1] / max_c if max_c else 0
            index += 1
        # 任务在每个虚拟机上执行后的剩余时间占整个截止时间的比值，用来评估任务能否在截止时间前完成。
        for v in vm_list:
            x[index] = (
                max(
                    max(task.workflow.deadline - now_time, 0)
                    - task.vref_time_cost[v][0],
                    0,
                )
                / task.workflow.deadline
            )
            index += 1

        # 6：任务预算、截止时间、最早完成、最晚完成、优先级关系、剩余未完成任务占总任务数的比例
        # 3*action_num：在每个VM上的时间、成本和任务完成的紧迫性
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
            # plt.plot(self.mean_losses, "-o", linewidth=1, markersize=2)
            # # plt.xlabel(str(self.target_update * len(self.mean_losses)) + "iterations")
            # plt.xlabel(str(len(self.mean_losses)) + "iterations")
            # plt.ylabel("Mean Losses")
            # plt.show()

            plt.plot(self.mean_rewards, "-o", linewidth=1, markersize=2)
            # plt.xlabel(str(self.target_update * len(self.mean_rewards)) + "iterations")
            plt.xlabel(str(len(self.mean_rewards)) + "iterations")
            plt.ylabel("Mean Rewards")
            plt.show()

    def run(
        self,
        last_part,
        task,
        vm_list,
        now_time,
        done,
    ):
        state = self.createState(task, vm_list, now_time)
        action = self.agent.get_action(state, self.epsilon)

        if self.agent.net.training:
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
                    if self.agent == DQN:
                        self.agent.dqn_target_net.load_state_dict(
                            self.agent.net.state_dict()
                        )
                    self.trainPlot()
            if done:
                self.transition += [state]
                self.transition[0] = done
                self.buffer.add(*self.transition)
                self.transition = []

        return vm_list[action]

    def trainPlotFinal(
        self,
        mean_makespan=[],
        mean_cost=[],
        succes_deadline_rate=[],
        succes_budget_rate=[],
        succes_both_rate=[],
    ):
        time_str = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M "))
        print("final pictures")

        if mean_cost:
            # self.cost += mean_cost
            plt.plot(mean_cost, "-o", linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Cost")
            plt.savefig("logs/" + time_str + "_cost.png", facecolor="w")
            # transparent=False
            plt.show()
            plt.clf()

        if mean_makespan:
            # self.makespan += mean_makespan
            plt.plot(mean_makespan, "-o", linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Makespan")
            plt.savefig("logs/" + time_str + "_makespan.png", facecolor="w")
            # transparent=False
            plt.show()
            plt.clf()

        if succes_budget_rate:
            # self.cost_rate += succes_budget_rate
            plt.plot(succes_budget_rate, "-o", linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Cost Rate")
            plt.savefig("logs/" + time_str + "_bsr.png", facecolor="w")
            # transparent=False
            plt.show()
            plt.clf()

        if succes_deadline_rate:
            # self.time_rate += succes_deadline_rate
            plt.plot(succes_deadline_rate, "-o", linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Time Rate")
            plt.savefig("logs/" + time_str + "_dsr.png", facecolor="w")
            # transparent=False
            plt.show()
            plt.clf()

        if succes_both_rate:
            # self.succes_both_rate += succes_both_rate
            plt.plot(succes_both_rate, "-o", linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Success Rate")
            plt.savefig("logs/" + time_str + "_both.png", facecolor="w")
            # transparent=False
            plt.show()
            plt.clf()
