import random
from operator import attrgetter

import seaborn as sbn
import simpy

from env import IaaS, Workload
from env.dax_parser import parseDAX
from env.task import TaskStatus
from env.workflow import Workflow
from functions import func, bdf, estimate
from model.sac import SAC

sbn.set_style("darkgrid", {"axes.grid": True, "axes.edgecolor": "black"})

import os
import torch
import numpy


def setRandSeed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def runEnv(
    wf_path,
    taskScheduler,
    seed,
    constant_df=0,
    constant_bf=0,
    arrival_rate=1 / 60,
    merge=False,
    wf_number=1,
    action_num=6,
    debug=False,
):
    # 整个所有工作流的任务剩余数
    global remained_tasks
    #
    global workload_finished
    global running
    # 表示系统或者仿真正在运行
    running = True
    # 表示剩余任务的数量，初始化为 0。
    remained_tasks = 0
    # 表示工作负载是否完成，初始化为 False，表示工作负载还未完成。
    workload_finished = False

    # 是工作流的到达率
    wf_arrival_rate = arrival_rate
    # workflows/secs 表示引导时间为 97 秒，可能是在模拟系统的启动时间
    boot_time = 97
    # VM 的 CPU 一周期所消耗的时间
    cycle_time = 3600
    # sec
    bandwidth = 20000000  # (2**20); # Byte   #20 MBps

    # 使用 SimPy 库设置一个模拟环境，其中包含不同的管道（Store）和资源（Resource 和 Container）
    # 用于模拟工作流提交、任务完成和虚拟机释放的流程。

    # 创建一个 SimPy 环境对象。这个环境是整个模拟的核心，用于协调所有进程和事件
    sim = simpy.Environment()
    # 创建一个 Store 对象，这是一个队列结构，用于存储提交的工作流。
    workflow_submit_pipe = simpy.Store(sim)
    # 当任务完成时，系统会将完成任务的信息发送到这个管道中，其他进程可以监听该管道并响应任务完成事件。
    task_finished_announce_pipe = simpy.Store(sim)
    # 当某个虚拟机的任务执行完毕后，系统会将虚拟机释放的消息发送到此管道，以便其他进程可以使用该虚拟机或进行相应的操作。
    vm_release_announce_pipe = simpy.Store(sim)
    # 这个资源可能用于控制对某个队列（比如准备好执行的任务队列）的访问，确保多个进程不会同时操作该队列，避免并发问题。
    ready_queue_key = simpy.Resource(sim, 1)
    # Container 对象类似于资源容器，可以用于存储和管理某种可消耗的资源（如任务数量、计算能力等）。
    ready_task_counter = simpy.Container(sim, init=0)

    all_task_num = 0
    finished_wfs = []
    workflow_pool = []
    released_vms_info = []
    tasks_ready_queue = []
    all_vms = []

    # 生成16个虚拟机，初始化虚拟机拥有的资源
    iaas = IaaS(sim, bandwidth, debug=False)
    iaas.setVirtualMachineTypeNums(boot_time, cycle_time)

    fastest_vm_type = max(iaas.vm_types_list, key=attrgetter("mips"))
    cheapest_vm_type = min(iaas.vm_types_list, key=lambda v: v.cycle_price)
    setRandSeed(seed * 5)
    workload = Workload(
        sim,
        workflow_submit_pipe,
        wf_path,
        wf_arrival_rate,
        max_wf_number=wf_number,
        debug=0,
    )

    def __poolingProcess():
        global workload_finished
        global remained_tasks
        while running and not workload_finished:
            # 从工作流管道 self.workflow_submit_pipe 得到dax的路径
            dax_path = yield workflow_submit_pipe.get()

            if dax_path == "end":
                workload_finished = True
                return

            # Parse DAX and make a workflow
            # 得到tasks的属性信息（ID，运行时间（长度））
            tasks, files = parseDAX(dax_path, merge=False)

            wf = Workflow(tasks, path=dax_path, submit_time=sim.now)
            for task in wf.tasks:
                task.status = TaskStatus.pool
                # 保存传输时间，表示这个任务从其父任务获取输入文件所需的最大传输时间。
                task.rank_trans = estimate.maxParentInputTransferTime(
                    task, fastest_vm_type
                )
                # 表示任务在该虚拟机类型上运行所需的时间。
                task.rank_exe = estimate.exeTime(task, fastest_vm_type)
                # 表示工作流在调度和执行这些任务之前仍需要的总执行时间。
                wf.remained_length += task.rank_exe

            # 计算每个任务的上下行优先级
            func.setUpwardRank(wf.exit_task, 0)
            func.setDownwardRank(wf.entry_task, 0)

            #       for t in wf.tasks:
            #         print(t.id, t.uprank, t.downrank)
            setRandSeed(seed + int(sim.now))

            # 创建工作流的截止日期以及预算
            bdf.createDeadline(wf, fastest_vm_type, constant_df=constant_df)
            bdf.createBudget(wf, cheapest_vm_type, constant_bf=constant_bf)

            workflow_pool.append(wf)
            remained_tasks += len(wf.tasks) - 2

            # 将入口任务完成分配，即工作流启动
            wf.entry_task.status = TaskStatus.done
            wf.entry_task.start_time = sim.now
            wf.entry_task.finish_time = sim.now

            if debug:
                print(
                    "[{:.2f} - {:10s}] {} (id: {}, deadline: {:.2f}, budget: {:.2f}, df: {:.2f}, bf: {:.2f}) is saved in the pool.\n # current Wf:{} # total Wf:{}".format(
                        sim.now,
                        "Pool",
                        dax_path,
                        wf.id,
                        wf.deadline,
                        wf.budget,
                        wf.deadline_factor,
                        wf.budget_factor,
                        len(workflow_pool),
                        len(workflow_pool) + len(finished_wfs),
                    )
                )

            __addToReadyQueue(wf.entry_task.succ)
            # yield 用于暂停函数的执行，并返回控制权给调用方（例如事件循环或调度器），直到某个条件满足后再次恢复执行。
            # yield 通常在协程、生成器或者异步框架中使用，使得函数可以挂起，而不阻塞整个程序。
            yield ready_task_counter.put(1)

    #       yield task_finished_announce_pipe.put(wf.entry_task);

    # 将一组任务（task_list）添加到一个准备执行的任务队列（tasks_ready_queue）中，并更新这些任务的状态和准备时间。
    def __addToReadyQueue(task_list):
        for t in task_list:
            t.status = TaskStatus.ready
            t.ready_time = sim.now
        request_key = ready_queue_key.request()
        # extend(): 是 Python 列表（list）对象的方法，它会将另一个列表的所有元素添加到当前列表的末尾。
        tasks_ready_queue.extend(task_list)
        ready_queue_key.release(request_key)

        if debug:
            print(
                "[{:.2f} - {:10s}] {} tasks are added to ready queue. queue size: {}.".format(
                    sim.now, "ReadyQueue", len(task_list), len(tasks_ready_queue)
                )
            )

    # 监听任务完成的消息，并根据任务的状态更新工作流的状态。
    def __queueingProcess():
        while running:
            # 获取已完成的任务
            finished_task = yield task_finished_announce_pipe.get()
            finished_task.status = TaskStatus.done
            wf = finished_task.workflow
            # 将已完成的任务加入工作流的完成任务列表中
            wf.finished_tasks.append(finished_task)

            ready_tasks = []
            # 遍历完成任务的后继任务（succ）来检查其是否准备好进行调度
            for child in finished_task.succ:
                if child.isReadyToSch():
                    #             print(child.id)
                    if child != wf.exit_task:
                        if merge:
                            func.mergeOnFly(child)
                        ready_tasks.append(child)
                    else:
                        #                 print("///////////////////")
                        wf.exit_task.status = TaskStatus.done
                        wf.exit_task.start_time = sim.now
                        wf.exit_task.finish_time = sim.now
                        # 计算工作流的 makespan（总执行时间），即出口任务的完成时间减去工作流的提交时间。
                        wf.makespan = wf.exit_task.finish_time - wf.submit_time
                        finished_wfs.append(wf)
                        workflow_pool.remove(wf)
                        if debug:
                            print(
                                "[{:.2f} - {:10s}] Workflow {} is finished.".format(
                                    sim.now, "Finished", wf.id
                                )
                            )
                            print(
                                "Deadline: {} Makespan: {}, Budget: {}, Cost: {}".format(
                                    wf.deadline, wf.makespan, wf.budget, wf.cost
                                )
                            )
                            # print("*" * 40)

            yield sim.timeout(0.2)
            if ready_tasks:
                __addToReadyQueue(ready_tasks)
                # 更新 ready_task_counter，将其值加 1，表示有新的任务可以准备调度。
                yield ready_task_counter.put(1)

    # 计算任务的剩余截止时间，并根据任务的上行优先级（uprank）、任务执行时间（exeTime）
    # 以及数据传输时间（maxParentInputTransferTime）对其截止时间（deadline）进行动态调整。
    # 计算任务的截止时间。
    def threeDeadline(tasks_list, fastest_type, now_time):
        for task in tasks_list:
            # 计算任务的执行长度（时间）
            task_len = estimate.maxParentInputTransferTime(
                task, fastest_type
            ) + estimate.exeTime(task, fastest_type)
            remained_deadline = (
                task.workflow.deadline + task.workflow.submit_time - now_time
            )
            if remained_deadline < 0:
                remained_deadline = 0
            #         task.deadline = task_len * remained_deadline / (task.uprank + task_len)
            task.deadline = (task_len * remained_deadline) / (
                estimate.maxParentInputTransferTime(task, fastest_type) + task.uprank
            )

    # 为任务列表中的每个任务计算其在不同虚拟机（VM）上的执行时间和执行成本，
    # 并将这些信息存储在任务对象的 vref_time_cost 属性中。
    # task.fast_run：任务上最快运行的虚拟机的执行时间
    def estimateRunTimeCost(
        task_list, vm_list, vm_types_list, now_time, changed_vm=None, new_vm=None
    ):
        for task in task_list:
            if changed_vm or new_vm:
                v = changed_vm if changed_vm else new_vm
                a = (
                    estimate.exeTime(task, v)
                    + estimate.maxParentInputTransferTime(task, v)
                    + v.waitingTime()
                )
                b = estimate.exeCost(task, v)
                #             task.vref_time_cost.update({weakref.ref(v): a})
                task.vref_time_cost.update({v: [a, b]})
            else:
                # 计算所有虚拟机的时间成本
                task.vref_time_cost = {}
                for v in vm_list + vm_types_list:
                    # 结合任务的输入文件传输时间以及虚拟机的等待时间。
                    a = (
                        estimate.exeTime(task, v)
                        + estimate.maxParentInputTransferTime(task, v)
                        + v.waitingTime()
                    )
                    if a < 0:
                        print("$" * 80)
                        print(
                            estimate.exeTime(task, v),
                            estimate.maxParentInputTransferTime(task, v),
                            v.waitingTime(),
                        )
                    b = estimate.exeCost(task, v)
                    #                 task.vref_time_cost.update({weakref.ref(v): a})
                    # dict.update() 方法会更新字典，如果字典中已经存在相同的键，则覆盖旧值，如果不存在该键，则添加新键值对。
                    task.vref_time_cost.update({v: [a, b]})

            # 确保任务优先选择执行时间最短的虚拟机
            task.vref_time_cost = dict(
                sorted(task.vref_time_cost.items(), key=lambda item: item[1][0])
            )
            # 表示在任务上最快运行的虚拟机的执行时间
            task.fast_run = list(task.vref_time_cost.values())[0][0]

    def prioritizeTasks(task_list):
        #       def slackTime(t):
        #         waiting_time = now_time - task.ready_time;
        #         return (task.deadline - waiting_time) - fast_run;

        task_list.sort(key=lambda t: t.deadline - t.fast_run)

    # 负责在虚拟机完成任务后，释放该虚拟机并更新相关的状态
    def __releasingProcess():
        while running:
            vm = yield vm_release_announce_pipe.get()
            iaas.releaseVirtualMachine(vm)
            all_vms.remove(vm)
            released_vms_info.append(vm)
            if debug:
                print(
                    "[{:.2f} - {:10s}] {} virtual machine is released. start time: {}. VM number: {}".format(
                        sim.now, "Releaser", vm.id, vm.start_time, len(all_vms)
                    )
                )

    def __schedulingProcess():
        global workload_finished
        global remained_tasks
        while running:
            # 等待至少有一个任务进入就绪队列
            # yield ready_task_counter.put(1) 和 yield ready_task_counter.get(1)
            # 分别代表的是将任务计数器（ready_task_counter）的值加 1 和从计数器中减 1。
            # 通常用于异步或并发编程，尤其在模拟调度系统中，这些操作可以帮助管理任务的就绪状态和同步执行。
            yield ready_task_counter.get(1)

            threeDeadline(tasks_ready_queue, fastest_vm_type, sim.now)
            # 只为该指定虚拟机计算指定任务的时间和成本
            changed_vm = None
            new_vm = None
            while len(tasks_ready_queue):
                # 计算任务在最快运行的虚拟机的执行时间
                estimateRunTimeCost(
                    tasks_ready_queue, all_vms, iaas.vm_types_list, sim.now
                )

                # prioritizeTasks should be call after that the deadline distributed
                # 对任务列表排序,按照截止时间-最快时间
                prioritizeTasks(tasks_ready_queue)

                choosed_task = tasks_ready_queue.pop(0)
                choosed_task.schedule_time = sim.now
                remained_tasks -= 1

                BFT, LFT = bdf.calBFT_LFT(
                    choosed_task,
                    sim.now,
                    fast_run=list(choosed_task.vref_time_cost.values())[0][0],
                    slow_run=list(choosed_task.vref_time_cost.values())[-1][0],
                )
                choosed_task.soft_deadline = BFT
                choosed_task.hard_deadline = LFT
                choosed_task.BFT = BFT
                choosed_task.LFT = LFT

                if debug:
                    print(
                        "[{:.2f} - {:10s}] {} task choosed for scheduling. L:{}".format(
                            sim.now, "TaskChooser", choosed_task.id, choosed_task.length
                        )
                    )

                # 计算pool中的所有任务数
                all_task_num = 0
                for w in workflow_pool:
                    all_task_num += len(w.tasks) - 2

                # 从 choosed_task.vref_time_cost 中选择虚拟机，并随机化选择的顺序,选择前6个。
                vlist = list(choosed_task.vref_time_cost.keys()) + []
                random.shuffle(vlist)
                vs = vlist[:action_num] + []
                # 查看目前可选的VM
                if debug:
                    print(
                        "[{:.2f} - {:10s}] total {} VMs, total {} deque tasks .".format(
                            sim.now, "Scheduler", len(vlist), len(tasks_ready_queue)
                        )
                    )

                # 选择VM
                choosed_vm = taskScheduler(
                    len(vlist) == action_num,
                    choosed_task,
                    vs,
                    sim.now,
                    remained_tasks == 0 and workload_finished,
                )

                # 核心功能是通过不断重新选择和随机打乱虚拟机列表，
                # 确保每次调度任务时有多种虚拟机可选，且不会过度使用相同的虚拟机。
                if len(vlist) != action_num:
                    action_step = action_num - 2
                    del vlist[:action_num]
                    while True:
                        if len(vlist) > action_step:
                            vs.remove(choosed_vm)
                            random_vm = random.choice(vs)
                            vs = vlist[:action_step] + [choosed_vm] + [random_vm]
                            random.shuffle(vs)
                            choosed_vm = taskScheduler(
                                False,
                                choosed_task,
                                vs,
                                sim.now,
                                remained_tasks == 0 and workload_finished,
                            )
                            #                 print(choosed_task.id, "------------------2", vs)
                            del vlist[:action_step]
                        else:
                            #                 print(vs, 0)
                            vs.remove(choosed_vm)
                            random_vm = random.choice(vs)
                            vs.remove(random_vm)
                            #                 print(vs, 1)
                            while len(vlist) < action_step:
                                #                   print(vs, 2)
                                random_vm = random.choice(vs)
                                vlist.append(random_vm)
                                vs.remove(random_vm)

                            vs = vlist + [choosed_vm] + [random_vm]
                            random.shuffle(vs)

                            choosed_vm = taskScheduler(
                                True,
                                choosed_task,
                                vs,
                                sim.now,
                                remained_tasks == 0 and workload_finished,
                            )

                            #                 print(choosed_task.id, "--------------3", vs)
                            break

                # 将所选虚拟机（choosed_vm）的执行成本添加到当前任务所属工作流的总成本中。
                choosed_task.workflow.cost += choosed_task.vref_time_cost[choosed_vm][1]
                # 减少工作流的剩余长度，根据已执行的任务长度（rank_exe），表示任务的进展。
                choosed_task.workflow.remained_length -= choosed_task.rank_exe
                # 清空任务的 vref_time_cost，表示任务已经被分配给虚拟机
                choosed_task.vref_time_cost = {}

                if choosed_vm.isVMType():
                    if debug:
                        print(
                            "[{:.2f} - {:10s}] A new VM with type {} is choosed (among {} options) for task {}.".format(
                                sim.now,
                                "Scheduler",
                                choosed_vm.name,
                                len(iaas.vm_types_list) + len(all_vms),
                                choosed_task.id,
                            )
                        )

                    nvm = iaas.provideVirtualMachine(choosed_vm, off_idle=True)
                    # 设置虚拟机的管道，用于在任务完成后通知任务调度器，以及在虚拟机资源释放后进行通信。
                    nvm.task_finished_announce_pipe = task_finished_announce_pipe
                    nvm.vm_release_announce_pipe = vm_release_announce_pipe
                    # 将任务 choosed_task 提交到虚拟机 nvm 上进行处理
                    yield sim.process(nvm.submitTask(choosed_task))
                    all_vms.append(nvm)

                else:
                    if debug:
                        print(
                            "[{:.2f} - {:10s}] {} VM with type {} is choosed for task {}.".format(
                                sim.now,
                                "Scheduler",
                                choosed_vm.id,
                                choosed_vm.type.name,
                                choosed_task.id,
                            )
                        )
                    # 将任务 choosed_task 提交到已有的虚拟机 choosed_vm 上，并暂停当前进程，等待任务提交过程结束。
                    # print(choosed_vm.type.name, choosed_task.id,"o b", choosed_task.budget, "c",estimate.taskExeCost(choosed_task, choosed_vm), "used b:",choosed_task.workflow.used_budget);
                    yield sim.process(choosed_vm.submitTask(choosed_task))

    # 统计了已完成工作流的总体情况，帮助评估执行时间和成本是否满足预算和截止日期的要求，
    # 同时提供调试信息用于分析不满足条件的工作流。
    def lastFunction():

        #     if len(finished_wfs)==1:
        #           wf = finished_wfs[0]
        #           a =  1 if wf.cost<= wf.budget and wf.makespan<= wf.deadline else 0
        #           return wf.makespan, wf.cost, wf.makespan/wf.deadline, wf.cost/wf.budget,  a

        total_time = 0.0
        total_cost = 0.0
        budget_meet = 0.0
        deadline_meet = 0.0
        both_meet = 0.0
        for wf in finished_wfs:
            total_time += wf.makespan
            total_cost += wf.cost

            if wf.cost <= wf.budget:
                budget_meet += 1
            else:
                # print("{}B", wf.budget, wf.cost, wf.budget - wf.cost)
                print(
                    "[Budget] wf.id: {}, wf.budget = {:.2f}, wf.cost = {:.2f}, distance = {:.2f}".format(
                        wf.id, wf.budget, wf.cost, wf.budget - wf.cost
                    )
                )

            if wf.makespan <= wf.deadline:
                deadline_meet += 1
            else:
                # print("XXD", wf.deadline, wf.makespan, wf.deadline - wf.makespan)
                print(
                    "[Deadline] wf.id: {}, wf.deadline = {:.2f}, wf.makespan = {:.2f}, distance = {:.2f}".format(
                        wf.id, wf.deadline, wf.makespan, wf.deadline - wf.makespan
                    )
                )

            if wf.cost <= wf.budget and wf.makespan <= wf.deadline:
                both_meet += 1

        print(
            "cost fail total : {}, makespan fail total : {}".format(
                len(finished_wfs) - budget_meet, len(finished_wfs) - deadline_meet
            )
        )
        total_time /= len(finished_wfs)
        total_cost /= len(finished_wfs)
        budget_meet /= len(finished_wfs)
        deadline_meet /= len(finished_wfs)
        both_meet /= len(finished_wfs)
        return total_time, total_cost, deadline_meet, budget_meet, both_meet

    sim.process(__poolingProcess())
    sim.process(__schedulingProcess())
    sim.process(__queueingProcess())
    sim.process(__releasingProcess())

    # 启动仿真，执行所有已经注册到 SimPy 环境中的进程和事件。
    sim.run()
    return lastFunction()
