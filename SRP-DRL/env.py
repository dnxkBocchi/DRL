import numpy as np
from scipy import stats
import queue
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

np.random.seed(3)


class SchedulingEnv:
    def __init__(self, args):
        # Environment Settings
        self.policy_num = len(args.Baselines)
        self.VMnum = args.VM_Num
        # assert self.VMnum == len(self.VMtypes)
        # self.s_features = 1 + args.VM_Num  # VMnum and job length and job arrival_time
        self.s_features = 2 * args.VM_Num  # VMnum and job length
        self.speed = [500] * self.VMnum
        self.CPU = np.ones((self.policy_num, self.VMnum))
        self.Memory = np.ones((self.policy_num, self.VMnum))
        self.IO = np.ones((self.policy_num, self.VMnum))

        self.load = np.zeros((self.policy_num, self.VMnum))
        self.actionNum = self.VMnum
        self.VMtypes = 4
        self.load_threshold = 0.95

        # Job Setting
        self.jobMI = args.Job_len_Mean
        self.jobMI_std = args.Job_len_Std
        self.jobNum = args.Job_Num
        self.lamda = args.lamda
        self.arrival_Times = np.zeros(self.jobNum)
        self.jobsMI = np.zeros(self.jobNum)
        self.lengths = np.zeros(self.jobNum)
        self.ddl = np.ones(self.jobNum) * args.Job_ddl  # 250ms = waitT + exeT
        self.reqCPUs = np.zeros(self.jobNum)
        self.reqMemorys = np.zeros(self.jobNum)
        self.reqIOs = np.zeros(self.jobNum)
        # generate workload
        self.gen_workload(self.lamda)

        # SAIRL
        # 1-VM id  2-start time  3-wait time 4-exe time 5-response time  6-actual load  7-reward  8-load_variance（running before）
        # 9- success  10-reject
        # 1-waiting time 2-select count 3-job_id of the head of the running queue
        # 4-job_id of the head of the waiting queue
        # Random
        self.RAN_events = np.zeros((10, self.jobNum))
        self.RAN_VM_events = np.zeros((4, self.VMnum), dtype=int)
        # Round Robin
        self.RR_events = np.zeros((10, self.jobNum))
        self.RR_VM_events = np.zeros((4, self.VMnum), dtype=int)
        # Earliest
        self.EITF_events = np.zeros((10, self.jobNum))
        self.EITF_VM_events = np.zeros((4, self.VMnum), dtype=int)
        # BEST_FIT
        self.BEST_FIT_events = np.zeros((10, self.jobNum))
        self.BEST_FIT_VM_events = np.zeros((4, self.VMnum), dtype=int)
        # DQN
        self.DQN_events = np.zeros((10, self.jobNum))
        self.DQN_VM_events = np.zeros((4, self.VMnum), dtype=int)

    def gen_workload(self, lamda):
        # Generate task required CPU、memory、IO
        # CPU, memory, and IO all obey a random distribution
        self.reqCPUs = np.random.uniform(0, self.load_threshold, self.jobNum)
        self.reqMemorys = np.random.uniform(0, self.load_threshold, self.jobNum)
        self.reqIOs = np.random.uniform(0, self.load_threshold, self.jobNum)
        # Generate arrival time of jobs (poisson distribution)
        intervalT = stats.expon.rvs(scale=1 / lamda, size=self.jobNum)
        print(
            "intervalT mean: ",
            round(np.mean(intervalT), 3),
            "  intervalT SD:",
            round(np.std(intervalT, ddof=1), 3),
        )
        self.arrival_Times = np.around(intervalT.cumsum(), decimals=3)
        last_arrivalT = self.arrival_Times[-1]
        print("last job arrivalT:", round(last_arrivalT, 3))

        # Generate jobs' length(Normal distribution)
        self.jobsMI = np.random.normal(self.jobMI, self.jobMI_std, self.jobNum)
        self.jobsMI = self.jobsMI.astype(int)
        print(
            "MI mean: ",
            round(np.mean(self.jobsMI), 3),
            "  MI SD:",
            round(np.std(self.jobsMI, ddof=1), 3),
        )
        # self.lengths = self.jobsMI / self.VMcapacity
        self.lengths = self.jobsMI
        print(
            "length mean: ",
            round(np.mean(self.lengths), 3),
            "  length SD:",
            round(np.std(self.lengths, ddof=1), 3),
        )

    def reset(self, args):
        # New
        self.load = np.zeros((self.policy_num, self.VMnum))
        self.CPU = np.ones((self.policy_num, self.VMnum))
        self.Memory = np.ones((self.policy_num, self.VMnum))
        self.IO = np.ones((self.policy_num, self.VMnum))
        # if each episode generates new workload
        self.arrival_Times = np.zeros(self.jobNum)
        self.jobsMI = np.zeros(self.jobNum)
        self.lengths = np.zeros(self.jobNum)
        self.ddl = np.ones(self.jobNum) * args.Job_ddl  # 250ms = waitT + exeT
        self.gen_workload(args.lamda)

        # reset all records
        self.RAN_events = np.zeros((10, self.jobNum))
        self.RAN_VM_events = np.zeros((4, self.VMnum), dtype=int)
        self.RR_events = np.zeros((10, self.jobNum))
        self.RR_VM_events = np.zeros((4, self.VMnum), dtype=int)
        self.EITF_events = np.zeros((10, self.jobNum))
        self.EITF_VM_events = np.zeros((4, self.VMnum), dtype=int)
        self.BEST_FIT_events = np.zeros((10, self.jobNum))
        self.BEST_FIT_VM_events = np.zeros((4, self.VMnum), dtype=int)
        self.DQN_events = np.zeros((10, self.jobNum))
        self.DQN_VM_events = np.zeros((4, self.VMnum), dtype=int)

    def workload(self, job_count):
        arrival_time = self.arrival_Times[job_count - 1]
        length = self.lengths[job_count - 1]
        reqCPU = self.reqCPUs[job_count - 1]
        reqMemory = self.reqMemorys[job_count - 1]
        reqIO = self.reqIOs[job_count - 1]
        ddl = self.ddl[job_count - 1]
        if job_count == self.jobNum:
            finish = True
        else:
            finish = False
        job_attributes = [
            job_count - 1,
            arrival_time,
            length,
            reqCPU,
            reqMemory,
            reqIO,
            ddl,
        ]
        return finish, job_attributes

    # Get the current real-time load information of each virtual machine
    def get_load(self, job_attrs, policyID, jobs, signs):
        # 为当前策略下的所有虚拟机初始化负载信息
        self.load[policyID - 1] = [0] * self.VMnum
        self.CPU[policyID - 1] = [1] * self.VMnum
        self.Memory[policyID - 1] = [1] * self.VMnum
        self.IO[policyID - 1] = [1] * self.VMnum

        job_id = job_attrs[0]
        # The head job_id of each virtual machine's running queue、
        # 获取每个虚拟机运行队列中最前面的作业ID，用于追踪虚拟机上正在运行的作业。
        run_queue_job_id = signs[2]

        # 遍历每台虚拟机的运行队列
        for index, head in enumerate(run_queue_job_id):
            # Virtual machine run queue head
            # 遍历该虚拟机上从队列头部开始到当前作业ID之前的所有作业，head 是运行队列的头，job_id 是当前作业ID
            for job_index in range(head, job_id):

                # Start time <= current job arrival time and completion time (waiting + execution time) = 0 and satisfy ddl
                # print(jobs[1, job_index] + jobs[3, job_index] > job_attrs[1])
                # print(jobs[7, job_index] == 1)

                # 获取当前作业的虚拟机ID，即该作业是在哪台虚拟机上执行的
                VM_id = int(jobs[0, job_index])

                # Current virtual machine
                # 判断当前作业是否属于正在遍历的虚拟机
                if VM_id == index:

                    # 检查作业的到达时间（jobs[1, job_index]）加上作业的等待时间（jobs[3, job_index]）是否大于当前作业的到达时间（job_attrs[1]）
                    # 并且该作业尚未执行（jobs[8, job_index] == 1）
                    if (
                        jobs[1, job_index] + jobs[3, job_index] > job_attrs[1]
                        and jobs[8, job_index] == 1
                    ):
                        # Not executed yet ,Current load <= 95%
                        # 检查当前虚拟机的负载是否低于负载阈值
                        if (
                            self.load[policyID - 1, VM_id] + jobs[5, job_index]
                            <= self.load_threshold
                        ):
                            # 更新当前虚拟机的负载信息
                            self.load[policyID - 1, VM_id] += jobs[5, job_index]
                            self.CPU[policyID - 1, VM_id] -= self.reqCPUs[job_index]
                            self.Memory[policyID - 1, VM_id] -= self.reqMemorys[
                                job_index
                            ]
                            self.IO[policyID - 1, VM_id] -= self.reqIOs[job_index]
                            # print(self.CPU[policyID - 1, VM_id], self.Memory[policyID - 1, VM_id], self.IO[policyID - 1, VM_id])
                        else:
                            # 如果当前虚拟机的负载超过阈值，则跳出循环，不再处理该虚拟机的后续作业。
                            # Update the head of the preparation queue
                            # signs[3, VM_id] = job_index
                            break
        return self.load[policyID - 1]

    @staticmethod
    def compuLoad(reqCPU, reqMemory, reqIO):
        # return 0.5 * reqCPU + 0.3 * reqMemory + 0.2 * reqIO
        return 0.6 * reqCPU + 0.3 * reqMemory + 0.1 * reqIO

    # Get task running performance parameters
    def update_current_job_information(self, job_attrs, action, policyId, jobs, signs):
        # ID of the current job
        job_id = job_attrs[0]
        # The first run of the job or the current load can satisfy the current job run
        # Load (running time)
        # resources request
        reqCPU = job_attrs[3]
        reqMemory = job_attrs[4]
        reqIO = job_attrs[5]
        # remaining resources
        remaining_CPU = self.CPU[policyId - 1, action]
        remaining_Memory = self.Memory[policyId - 1, action]
        remaining_IO = self.IO[policyId - 1, action]
        # nonlinear change in load and performance
        exe_time = job_attrs[2] / (
            self.speed[action] * (1 - np.power(self.load[policyId - 1, action], 0.5))
        )
        job_load = self.compuLoad(reqCPU, reqMemory, reqIO)
        if job_id == 0 or (
            remaining_CPU >= reqCPU
            and remaining_Memory >= reqMemory
            and remaining_IO >= reqIO
            and (self.load[policyId - 1, action] + job_load <= self.load_threshold)
        ):
            start_time = job_attrs[1]
            wait_time = 0
            response_time = wait_time + exe_time
            actual_load = job_load
            success = 1 if response_time <= job_attrs[6] else 0
            reject = 0 if response_time <= job_attrs[6] else 1
        else:
            # The head of the run queue of the virtual machine corresponding to the action
            run_head = signs[2, action]
            # Run task list
            run_list = []
            # current load
            load = 0
            # remaining resources
            remaining_CPU = 1
            remaining_Memory = 1
            remaining_IO = 1
            # Update the running list
            update = False
            # 这段代码实现了在虚拟机的运行队列中选择任务、更新虚拟机资源和负载状态的功能。
            # 它通过判断任务的结束时间和资源占用情况来动态决定任务是否可以被加入运行队列。
            # 如果虚拟机的负载超过阈值，则任务会被放入准备队列。
            for job_index in range(run_head, job_id):
                if jobs[0, job_index] == action:
                    # End time (start + exe) > current job arrival time and satisfy ddl (task running in the current virtual machine)
                    if (
                        jobs[1, job_index] + jobs[3, job_index] > job_attrs[1]
                        and jobs[8, job_index] == 1
                    ):
                        if update is False:
                            # Update the header of the run list
                            signs[2, action] = job_index
                            update = True

                        if load + jobs[5, job_index] <= self.load_threshold:
                            load += jobs[5, job_index]
                            remaining_CPU -= self.reqCPUs[job_index]
                            remaining_Memory -= self.reqMemorys[job_index]
                            remaining_IO -= self.reqIOs[job_index]
                            # Store the end time 、resources request and load of the currently running task
                            run_list.append(
                                [
                                    jobs[1, job_index] + jobs[3, job_index],
                                    [
                                        self.reqCPUs[job_index],
                                        self.reqMemorys[job_index],
                                        self.reqIOs[job_index],
                                    ],
                                    jobs[5, job_index],
                                ]
                            )
                        else:
                            # There are other tasks stored in the preparation queue
                            signs[3, action] = job_index
                            break

            # Sort the run list in ascending order according to the end time field
            run_list.sort(key=lambda run_job: run_job[0], reverse=False)
            # if len(run_list) > 1:
            #     print('SUCCESS')

            # Process task
            # Current task load
            exe_time = job_attrs[2] / (self.speed[action] * (1 - np.power(load, 0.5)))
            # Find the time point that satisfies the first time when the running task is completed
            # if len(run_list) > 0:
            index = 0
            while index < len(run_list) and (
                ((remaining_CPU + run_list[index][1][0]) < reqCPU)
                or ((remaining_Memory + run_list[index][1][1]) < reqMemory)
                or ((remaining_IO + run_list[index][1][2]) < reqIO)
                or ((self.load_threshold - load + run_list[index][2]) < job_load)
            ):
                load -= run_list[index][2]
                remaining_CPU += run_list[index][1][0]
                remaining_Memory += run_list[index][1][1]
                remaining_IO += run_list[index][1][2]
                exe_time = job_attrs[2] / (
                    self.speed[action] * (1 - np.power(load, 0.5))
                )
                index += 1

            start_time = (
                run_list[index][0]
                if run_list[index][0] > job_attrs[1]
                else job_attrs[1]
            )
            wait_time = (
                0
                if run_list[index][0] > job_attrs[1]
                else run_list[index][0] - job_attrs[1]
            )
            exe_time = exe_time
            response_time = wait_time + exe_time
            actual_load = job_load
            success = 1 if response_time <= job_attrs[6] else 0
            reject = 0 if response_time <= job_attrs[6] else 1

        return (
            round(start_time, 4),
            round(wait_time, 4),
            round(exe_time, 4),
            round(response_time, 4),
            round(actual_load, 4),
            success,
            reject,
        )

    # Update the earliest completion time of each strategy
    @staticmethod
    def update_idle_time(action, new_idleT, success, signs):
        if signs[0, action] != 0:
            if success == 1:
                return min(new_idleT, signs[0, action])
        return new_idleT

    # Reward
    def feedback(self, job_attrs, action, policyID):
        job_id = job_attrs[0]
        arrival_time = job_attrs[1]
        length = job_attrs[2]
        reqCPU = job_attrs[3]
        reqMemory = job_attrs[4]
        reqIO = job_attrs[5]
        ddl = job_attrs[6]
        reward = 0

        if policyID == 1:
            (
                start_time,
                wait_time,
                exe_time,
                response_time,
                actual_load,
                success,
                reject,
            ) = self.update_current_job_information(
                job_attrs, action, policyID, self.RAN_events, self.RAN_VM_events
            )
        elif policyID == 2:
            (
                start_time,
                wait_time,
                exe_time,
                response_time,
                actual_load,
                success,
                reject,
            ) = self.update_current_job_information(
                job_attrs, action, policyID, self.RR_events, self.RR_VM_events
            )
        elif policyID == 3:
            (
                start_time,
                wait_time,
                exe_time,
                response_time,
                actual_load,
                success,
                reject,
            ) = self.update_current_job_information(
                job_attrs, action, policyID, self.EITF_events, self.EITF_VM_events
            )
        elif policyID == 4:
            (
                start_time,
                wait_time,
                exe_time,
                response_time,
                actual_load,
                success,
                reject,
            ) = self.update_current_job_information(
                job_attrs,
                action,
                policyID,
                self.BEST_FIT_events,
                self.BEST_FIT_VM_events,
            )
        elif policyID == 5:
            (
                start_time,
                wait_time,
                exe_time,
                response_time,
                actual_load,
                success,
                reject,
            ) = self.update_current_job_information(
                job_attrs, action, policyID, self.DQN_events, self.DQN_VM_events
            )

        # When the load of the current task is not executed
        loads = self.load[policyID - 1]

        # differ = np.abs(loads[action] - np.mean(loads))
        differ = loads[action] - np.mean(loads)
        # self.load_variance[job_id] = np.var(loads)

        # if success == 1:
        #     reward = length * 0.01 * actual_load / (response_time * np.exp(differ))
        #
        # if reject == 1:
        #     reward = -length * 0.01 * np.exp(differ)
        new_idleT = start_time + exe_time
        if success == 1:
            if differ > 0:
                reward = 0.1 / response_time
            else:
                reward = np.abs(differ) / response_time

        if reject == 1:
            reward = -1
        # whether success

        if policyID == 1:
            self.RAN_events[0, job_id] = action
            self.RAN_events[1, job_id] = start_time
            self.RAN_events[2, job_id] = wait_time
            self.RAN_events[3, job_id] = exe_time
            self.RAN_events[4, job_id] = response_time
            self.RAN_events[5, job_id] = actual_load
            self.RAN_events[6, job_id] = reward
            self.RAN_events[7, job_id] = round(np.var(loads), 3)
            self.RAN_events[8, job_id] = success
            self.RAN_events[9, job_id] = reject
            # update VM info
            self.RAN_VM_events[1, action] += 1
            self.RAN_VM_events[0, action] = self.update_idle_time(
                action, new_idleT, success, self.RAN_VM_events
            )
            # print('VMC_after:', self.RAN_VM_events[0, action])
        elif policyID == 2:
            self.RR_events[0, job_id] = action
            self.RR_events[1, job_id] = start_time
            self.RR_events[2, job_id] = wait_time
            self.RR_events[3, job_id] = exe_time
            self.RR_events[4, job_id] = response_time
            self.RR_events[5, job_id] = actual_load
            self.RR_events[6, job_id] = reward
            self.RR_events[7, job_id] = round(np.var(loads), 3)
            self.RR_events[8, job_id] = success
            self.RR_events[9, job_id] = reject
            # update VM info
            self.RR_VM_events[1, action] += 1
            self.RR_VM_events[0, action] = self.update_idle_time(
                action, new_idleT, success, self.RR_VM_events
            )
        elif policyID == 3:
            self.EITF_events[0, job_id] = action
            self.EITF_events[1, job_id] = start_time
            self.EITF_events[2, job_id] = wait_time
            self.EITF_events[3, job_id] = exe_time
            self.EITF_events[4, job_id] = response_time
            self.EITF_events[5, job_id] = actual_load
            self.EITF_events[6, job_id] = reward
            self.EITF_events[7, job_id] = round(np.var(loads), 3)
            self.EITF_events[8, job_id] = success
            self.EITF_events[9, job_id] = reject
            # update VM info
            self.EITF_VM_events[1, action] += 1
            self.EITF_VM_events[0, action] = self.update_idle_time(
                action, new_idleT, success, self.EITF_VM_events
            )
        elif policyID == 4:
            self.BEST_FIT_events[0, job_id] = action
            self.BEST_FIT_events[1, job_id] = start_time
            self.BEST_FIT_events[2, job_id] = wait_time
            self.BEST_FIT_events[3, job_id] = exe_time
            self.BEST_FIT_events[4, job_id] = response_time
            self.BEST_FIT_events[5, job_id] = actual_load
            self.BEST_FIT_events[6, job_id] = reward
            self.BEST_FIT_events[7, job_id] = round(np.var(loads), 3)
            self.BEST_FIT_events[8, job_id] = success
            self.BEST_FIT_events[9, job_id] = reject
            # update VM info
            self.BEST_FIT_VM_events[1, action] += 1
            self.BEST_FIT_VM_events[0, action] = self.update_idle_time(
                action, new_idleT, success, self.BEST_FIT_VM_events
            )
        elif policyID == 5:
            self.DQN_events[0, job_id] = action
            self.DQN_events[1, job_id] = start_time
            self.DQN_events[2, job_id] = wait_time
            self.DQN_events[3, job_id] = exe_time
            self.DQN_events[4, job_id] = response_time
            self.DQN_events[5, job_id] = actual_load
            self.DQN_events[6, job_id] = reward
            self.DQN_events[7, job_id] = round(np.var(loads), 3)
            self.DQN_events[8, job_id] = success
            self.DQN_events[9, job_id] = reject
            # update VM info
            self.DQN_VM_events[1, action] += 1
            self.DQN_VM_events[0, action] = self.update_idle_time(
                action, new_idleT, success, self.DQN_VM_events
            )
        return reward

    def get_VM_idleT(self, policyID):
        if policyID == 1:
            idleTimes = self.RAN_VM_events[0, :]
        elif policyID == 2:
            idleTimes = self.RR_VM_events[0, :]
        elif policyID == 3:
            idleTimes = self.EITF_VM_events[0, :]
        elif policyID == 4:
            idleTimes = self.BEST_FIT_VM_events[0, :]
        elif policyID == 5:
            idleTimes = self.DQN_VM_events[0, :]
        return idleTimes

    def getState(self, job_attrs, policyID):
        arrivalT = job_attrs[1]
        length = job_attrs[2]
        reqCPU = job_attrs[3]
        reqMemory = job_attrs[4]
        reqIO = job_attrs[5]
        # state_job = [length]
        # state_job = [arrivalT, length]

        # Get the real-time load of the current virtual machine
        if policyID == 5:  # DQN
            loads = self.get_load(
                job_attrs, policyID, self.DQN_events, self.DQN_VM_events
            )
        elif policyID == 1:  # random
            loads = self.get_load(
                job_attrs, policyID, self.RAN_events, self.RAN_VM_events
            )
        elif policyID == 2:  # RR
            loads = self.get_load(
                job_attrs, policyID, self.RR_events, self.RR_VM_events
            )
        elif policyID == 3:  # used for as
            loads = self.get_load(
                job_attrs, policyID, self.EITF_events, self.EITF_VM_events
            )
        elif policyID == 4:
            loads = self.get_load(
                job_attrs, policyID, self.BEST_FIT_events, self.BEST_FIT_VM_events
            )
        current_speed = np.multiply(1 - np.power(loads, 0.5), self.speed)
        speed_norm = (current_speed - np.min(current_speed)) / (
            max(np.max(current_speed) - np.min(current_speed), 1)
        )
        state = np.hstack((speed_norm.tolist(), loads))
        return state

    def get_accumulateRewards(self, policies, start, end):

        rewards = np.zeros(policies)
        rewards[0] = sum(self.RAN_events[6, start:end])
        rewards[1] = sum(self.RR_events[6, start:end])
        rewards[2] = sum(self.EITF_events[6, start:end])
        rewards[3] = sum(self.BEST_FIT_events[6, start:end])
        rewards[4] = sum(self.DQN_events[6, start:end])
        return np.around(rewards, 2)

    def get_totalRewards(self, policies, start):
        rewards = np.zeros(policies)
        rewards[0] = sum(self.RAN_events[6, start : self.jobNum])
        rewards[1] = sum(self.RR_events[6, start : self.jobNum])
        rewards[2] = sum(self.EITF_events[6, start : self.jobNum])
        rewards[3] = sum(self.BEST_FIT_events[6, start : self.jobNum])
        rewards[4] = sum(self.DQN_events[6, start : self.jobNum])
        return np.around(rewards, 2)

    def get_totalResponseTs(self, policies, start):
        respTs = np.zeros(policies)
        respTs[0] = np.mean(self.RAN_events[4, start : self.jobNum])
        respTs[1] = np.mean(self.RR_events[4, start : self.jobNum])
        respTs[2] = np.mean(self.EITF_events[4, start : self.jobNum])
        respTs[3] = np.mean(self.BEST_FIT_events[4, start : self.jobNum])
        respTs[4] = np.mean(self.DQN_events[4, start : self.jobNum])
        return np.around(respTs, 3)

    def get_totalSuccess(self, policies, start):
        successT = np.zeros(
            policies
        )  # sum(self.RAN_events[7, 3000:-1])/(self.jobNum - 3000)
        successT[0] = sum(self.RAN_events[8, start : self.jobNum]) / (
            self.jobNum - start + 1
        )
        successT[1] = sum(self.RR_events[8, start : self.jobNum]) / (
            self.jobNum - start + 1
        )
        successT[2] = sum(self.EITF_events[8, start : self.jobNum]) / (
            self.jobNum - start + 1
        )
        successT[3] = sum(self.BEST_FIT_events[8, start : self.jobNum]) / (
            self.jobNum - start + 1
        )
        successT[4] = sum(self.DQN_events[8, start : self.jobNum]) / (
            self.jobNum - start + 1
        )
        return np.around(successT, 3)

    def get_total_load_variance(self, policies, start):
        load_variance = np.zeros(policies)
        load_variance[0] = sum(self.RAN_events[7, start : self.jobNum]) / (
            self.jobNum - start + 1
        )
        load_variance[1] = sum(self.RR_events[7, start : self.jobNum]) / (
            self.jobNum - start + 1
        )
        load_variance[2] = sum(self.EITF_events[7, start : self.jobNum]) / (
            self.jobNum - start + 1
        )
        load_variance[3] = sum(self.BEST_FIT_events[7, start : self.jobNum]) / (
            self.jobNum - start + 1
        )
        load_variance[4] = sum(self.DQN_events[7, start : self.jobNum]) / (
            self.jobNum - start + 1
        )
        return np.around(load_variance, 3)
