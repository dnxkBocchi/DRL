import time

import torch
import numpy as np
from env import SchedulingEnv
from model import baseline_DQN, baselines
from utils import get_args
import matplotlib.pyplot as plt

start = time.time()

args = get_args()
# store result 总共5个不同的方法比较
performance_lamda = np.zeros(args.Baseline_num)
performance_success = np.zeros(args.Baseline_num)
performance_load_variance = np.zeros(args.Baseline_num)

performance_successes = np.zeros((args.Epoch, args.Baseline_num))
# gen env 初始化环境，任务、VM的特征维度
env = SchedulingEnv(args)

# build model
brainRL = baseline_DQN(env.actionNum, env.s_features)  # (10, 20)
brainOthers = baselines(env.actionNum, env.VMtypes)


global_step = 0
my_learn_step = 0
DQN_Reward_list = []
My_reward_list = []
Reward_list = np.zeros(args.Epoch)
for episode in range(args.Epoch):

    print(
        "----------------------------Episode", episode, "----------------------------"
    )
    args.SAIRL_greedy += 0.04
    job_c = 1  # job counter
    performance_c = 0

    # 每次迭代，将所有环境，任务重新设置，共10次
    env.reset(
        args
    )  # attention: whether generate new workload, if yes, don't forget to modify reset() function

    performance_respTs = []
    while True:
        # baseline DQN
        global_step += 1

        # finish只有到最后一个才会变false， job_attrs得到该任务的特征
        finish, job_attrs = env.workload(job_c)

        # ？ 把该任务信息与每台虚拟机的信息做计算，得到VM的归一化速度和负载信息
        DQN_state = env.getState(job_attrs, 5)
        if global_step != 1:
            brainRL.store_transition(last_state, last_action, last_reward, DQN_state)

        # 一个整数，返回选定的动作，无论是基于Q值选择的最优动作还是随机选择的动作。
        action_DQN = brainRL.choose_action(DQN_state)  # choose action

        # 获取任务的运行性能参数，具体描述任务的启动时间、等待时间、执行时间、响应时间、任务负载情况，并判断任务是否成功或被拒绝。
        # 计算reward
        reward_DQN = env.feedback(job_attrs, action_DQN, 5)

        Reward_list[episode] += reward_DQN
        if episode == 1:
            DQN_Reward_list.append(reward_DQN)
        if (global_step > args.Dqn_start_learn) and (
            global_step % args.Dqn_learn_interval == 0
        ):  # learn
            brainRL.learn()
        last_state = DQN_state
        last_action = action_DQN
        last_reward = reward_DQN
        # print("step {}， reward: {}".format(global_step , reward_DQN))

        # random policy
        state_Random = env.getState(job_attrs, 1)
        action_random = brainOthers.random_choose_action()
        reward_random = env.feedback(job_attrs, action_random, 1)
        # round robin policy
        state_RR = env.getState(job_attrs, 2)
        action_RR = brainOthers.RR_choose_action(job_c)
        reward_RR = env.feedback(job_attrs, action_RR, 2)
        # EITF
        state_EITF = env.getState(job_attrs, 3)
        idleTimes = env.get_VM_idleT(3)  # get VM state
        action_EITF = brainOthers.EITF_choose_action(idleTimes)
        reward_EITF = env.feedback(job_attrs, action_EITF, 3)
        # BEST_FIT
        state_BEST_FIT = env.getState(job_attrs, 4)
        action_BEST_FIT = brainOthers.BEST_FIT_choose_action(
            env.CPU[3], env.Memory[3], env.IO[3], job_attrs
        )
        reward_BEST_FIT = env.feedback(job_attrs, action_BEST_FIT, 4)

        if job_c % 500 == 0:
            acc_Rewards = env.get_accumulateRewards(
                args.Baseline_num, performance_c, job_c
            )
            # avg_load_variance = env.get_load_variance(args.Baseline_num, performance_c, job_c)
            performance_c = job_c

        job_c += 1
        if episode > 2:
            args.SAIRL_learn_interval += 2
        if finish:
            break

    # episode performance
    # startP = 2000
    startP = 0

    total_Rewards = env.get_totalRewards(args.Baseline_num, startP)
    avg_allRespTs = env.get_totalResponseTs(args.Baseline_num, startP)
    total_success = env.get_totalSuccess(args.Baseline_num, startP)
    avg_load_variance = env.get_total_load_variance(args.Baseline_num, startP)
    print("total performance (after {} jobs):".format(startP))
    for i in range(len(args.Baselines)):
        name = "[" + args.Baselines[i] + "]"
        print(
            name + " reward:",
            total_Rewards[i],
            " avg_responseT:",
            avg_allRespTs[i],
            "success_rate:",
            total_success[i],
            "avg_load_variance:",
            avg_load_variance[i],
        )
    performance_successes[episode] = env.get_totalSuccess(args.Baseline_num, 0)

    if episode != 0:
        performance_lamda[:] += env.get_totalResponseTs(args.Baseline_num, 0)
        performance_success[:] += env.get_totalSuccess(args.Baseline_num, 0)
        performance_load_variance[:] += env.get_total_load_variance(
            args.Baseline_num, 0
        )
print("")

print("---------------------------- Final results ----------------------------")
performance_lamda = np.around(performance_lamda / (args.Epoch - 1), 3)
performance_success = np.around(performance_success / (args.Epoch - 1), 3)
performance_load_variance = np.around(performance_load_variance / (args.Epoch - 1), 3)
print("avg_responseT:")
print(performance_lamda)
print("success_rate:")
print(performance_success)
print("load variance:")
print(performance_load_variance)
