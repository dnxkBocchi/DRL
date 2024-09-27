import random

import seaborn as sbn
import simpy

from env.workload import Workload, Workflow
from env.task import TaskStatus, parseDAX
from env.virtual_machine import VirtualMachine
from env.ncp_network import create_ncp_graph
from schedule import estimate

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
    wf_number=1,
    action_num=6,
    debug=False,
):
    # 整个所有工作流的任务剩余数
    global remained_tasks
    global workload_finished
    global running
    global capacity_falty
    # 表示系统或者仿真正在运行
    running = True
    # 表示剩余任务的数量，初始化为 0。
    remained_tasks = 0
    # 表示工作负载是否完成，初始化为 False，表示工作负载还未完成。
    workload_finished = False
    capacity_falty = 0

    # 是工作流的到达率
    wf_arrival_rate = arrival_rate

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

    finished_wfs = []
    workflow_pool = []
    tasks_ready_queue = []
    nvms = []

    NCP_graph, NCPs = create_ncp_graph(node_nums=action_num)
    NCPs_id = NCP_graph.get_nodes()
    for ncp_type in NCPs:
        vm = VirtualMachine(sim, ncp_type, NCP_graph, off_idle=True, debug=debug)
        vm.start()
        nvms.append(vm)

    fastest_ncp_type = max(NCPs, key=lambda v: v.compute_capacity)
    cheapest_ncp_type = min(NCPs, key=lambda v: v.cycle_price)
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
                    task, fastest_ncp_type
                )
                # 表示任务在该虚拟机类型上运行所需的时间。
                task.rank_exe = estimate.exeTime(task, fastest_ncp_type)
                # 表示工作流在调度和执行这些任务之前仍需要的总执行时间。
                wf.remained_length += task.rank_exe

            # 计算每个任务的上下行优先级
            estimate.setUpwardRank(wf.exit_task, 0)
            estimate.setDownwardRank(wf.entry_task, 0)

            #       for t in wf.tasks:
            #         print(t.id, t.uprank, t.downrank)
            setRandSeed(seed + int(sim.now))

            # 创建工作流的截止日期以及预算
            estimate.createDeadline(wf, fastest_ncp_type, constant_df=constant_df)
            estimate.createBudget(wf, cheapest_ncp_type, constant_bf=constant_bf)

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
                        ready_tasks.append(child)
                    else:
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

    # 为任务列表中的每个任务计算一个新的截止时间
    # ??? 不懂这样的设计
    def threeDeadline(tasks_list, fastest_type, now_time):
        for task in tasks_list:
            # 计算任务的执行长度（时间）
            task_len = task.rank_trans + task.rank_exe
            # 计算任务剩余的截止时间
            remained_deadline = (
                task.workflow.deadline + task.workflow.submit_time - now_time
            )
            if remained_deadline < 0:
                remained_deadline = 0

            # 基于动态调度理论、任务优先级和依赖关系、软实时系统中的比例分配原则等。其核心目标是：
            # 保证高优先级任务按时完成，特别是关键路径任务。
            # 避免传输延迟造成的调度瓶颈。
            # 合理分配剩余时间，防止全局工作流的截止时间影响系统的平稳运行。
            task.deadline = (task_len * remained_deadline) / (
                task.rank_trans + task.uprank
            )
            if debug:
                print(
                    "[task deadline] task_len :{}, remained_deadline :{}, deadline :{}".format(
                        task_len, remained_deadline, task.deadline
                    )
                )

    # 为任务列表中的每个任务计算其在不同虚拟机（VM）上的执行时间和执行成本，
    # 并将这些信息存储在任务对象的 vref_time_cost 属性中。
    # task.fast_run：任务上最快运行的虚拟机的执行时间
    def estimateRunTimeCost(task_list):
        for task in task_list:
            # 计算所有虚拟机的时间成本
            task.vref_time_cost = {}
            for ncp in NCPs:
                # 结合任务的输入文件传输时间以及虚拟机的等待时间。
                a = (
                    estimate.exeTime(task, ncp)
                    + estimate.maxParentInputTransferTime(task, ncp)
                    + ncp.waitingTime()
                )
                if a < 0:
                    print("$" * 80)
                b = estimate.exeCost(task, ncp)
                # dict.update() 方法会更新字典，如果字典中已经存在相同的键，则覆盖旧值，如果不存在该键，则添加新键值对。
                task.vref_time_cost.update({ncp: [a, b]})

            # 确保任务优先选择执行时间最短的虚拟机，并重新生成一个有序的字典
            task.vref_time_cost = dict(
                sorted(task.vref_time_cost.items(), key=lambda item: item[1][0])
            )
            # 表示在任务上最快运行的虚拟机的执行时间
            task.fast_run = list(task.vref_time_cost.values())[0][0]

    def prioritizeTasks(task_list):
        task_list.sort(key=lambda t: t.deadline - t.fast_run)

    # 负责在虚拟机完成任务后，释放该虚拟机并更新相关的状态
    def __releasingProcess():
        while running:
            vm = yield vm_release_announce_pipe.get()
            if debug:
                print(
                    "[{:.2f} - {:10s}] {} virtual machine is released. vm tasks num = {}. ".format(
                        sim.now,
                        "Releaser",
                        vm.id,
                        vm.finished_tasks_number,
                    )
                )

    # 选择VM
    def chooseVM(ncp_list, choosed_task):
        random.shuffle(ncp_list)
        choosed_vm = taskScheduler(
            len(ncp_list) == action_num,
            choosed_task,
            ncp_list,
            sim.now,
            remained_tasks == 0 and workload_finished,
        )
        return choosed_vm

    # 找到所选的nvm，并且更新其他vm的管道
    def updateVMs(choosed_vm):
        for i in range(len(nvms)):
            if nvms[i].ncp == choosed_vm:
                nvm = nvms[i]
                break
        # 设置虚拟机的管道，用于在任务完成后通知任务调度器，以及在虚拟机资源释放后进行通信。
        for nvmi in nvms:
            nvmi.task_finished_announce_pipe = task_finished_announce_pipe
            nvmi.vm_release_announce_pipe = vm_release_announce_pipe
            if workload_finished and remained_tasks == 0:
                nvmi.workload_finished = True
        return nvm

    def __schedulingProcess():
        global workload_finished
        global remained_tasks
        global capacity_falty
        while running:
            # 等待至少有一个任务进入就绪队列
            # yield ready_task_counter.put(1) 和 yield ready_task_counter.get(1)
            # 分别代表的是将任务计数器（ready_task_counter）的值加 1 和从计数器中减 1。
            # 通常用于异步或并发编程，尤其在模拟调度系统中，这些操作可以帮助管理任务的就绪状态和同步执行。
            yield ready_task_counter.get(1)
            # 计算工作流中的每个任务截止时间
            threeDeadline(tasks_ready_queue, fastest_ncp_type, sim.now)
            while len(tasks_ready_queue):
                # 计算任务在最快运行的虚拟机的完成时间以及成本
                estimateRunTimeCost(tasks_ready_queue)

                # prioritizeTasks should be call after that the deadline distributed
                # 对任务列表排序,按照截止时间-最快时间
                prioritizeTasks(tasks_ready_queue)

                choosed_task = tasks_ready_queue.pop(0)
                choosed_task.schedule_time = sim.now
                remained_tasks -= 1

                BFT, LFT = estimate.calBFT_LFT(
                    choosed_task,
                    sim.now,
                    fast_run=list(choosed_task.vref_time_cost.values())[0][0],
                    slow_run=list(choosed_task.vref_time_cost.values())[-1][0],
                )
                choosed_task.soft_deadline = BFT
                choosed_task.hard_deadline = LFT
                choosed_task.BFT = BFT
                choosed_task.LFT = LFT

                # 计算pool中的所有任务数
                all_task_num = 0
                for w in workflow_pool:
                    all_task_num += len(w.tasks) - 2

                # 从 choosed_task.vref_time_cost 中选择虚拟机，并随机化选择的顺序,选择前6个。
                ncp_list = list(choosed_task.vref_time_cost.keys()) + []
                choosed_vm = chooseVM(ncp_list, choosed_task)
                # 直接将vm的容量不够情况，重新选择
                # while choosed_task.input_size > choosed_vm.storage_capacity:
                #     capacity_falty += 1
                #     choosed_vm = chooseVM(ncp_list, choosed_task)
                #     if debug:
                #         print(
                #             "choose_vm id:{}, choosed_vm:{}, choose_task:{}".format(
                #                 choosed_vm.node_id,
                #                 choosed_vm.storage_capacity,
                #                 choosed_task.input_size,
                #             )
                #         )

                if debug:
                    print(
                        "[{:.2f} - {:10s}] {} task choosed for scheduling. L:{} to nvm: {}.".format(
                            sim.now,
                            "Scheduler",
                            choosed_task.id,
                            choosed_task.length,
                            choosed_vm.node_id,
                        )
                    )

                # 将所选虚拟机（choosed_vm）的执行成本添加到当前任务所属工作流的总成本中。
                choosed_task.workflow.cost += choosed_task.vref_time_cost[choosed_vm][1]
                # 减少工作流的剩余长度，根据已执行的任务长度（rank_exe），表示任务的进展。
                choosed_task.workflow.remained_length -= choosed_task.rank_exe
                # 清空任务的 vref_time_cost，表示任务已经被分配给虚拟机
                choosed_task.vref_time_cost = {}

                nvm = updateVMs(choosed_vm)
                # 将任务 choosed_task 提交到虚拟机 nvm 上进行处理
                yield sim.process(nvm.submitTask(choosed_task))

    # 统计了已完成工作流的总体情况，帮助评估执行时间和成本是否满足预算和截止日期的要求，
    # 同时提供调试信息用于分析不满足条件的工作流。
    def lastFunction():

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
                    "[Budget] wf.path: {}, wf.budget = {:.2f}, wf.cost = {:.2f}, distance = {:.2f}".format(
                        wf.path, wf.budget, wf.cost, wf.budget - wf.cost
                    )
                )

            if wf.makespan <= wf.deadline:
                deadline_meet += 1
            else:
                # print("XXD", wf.deadline, wf.makespan, wf.deadline - wf.makespan)
                print(
                    "[Deadline] wf.path: {}, wf.deadline = {:.2f}, wf.makespan = {:.2f}, distance = {:.2f}".format(
                        wf.path, wf.deadline, wf.makespan, wf.deadline - wf.makespan
                    )
                )

            if wf.cost <= wf.budget and wf.makespan <= wf.deadline:
                both_meet += 1

        print(
            "cost fail total : {}, makespan fail total : {}, capacity_falty: {}".format(
                len(finished_wfs) - budget_meet,
                len(finished_wfs) - deadline_meet,
                capacity_falty,
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
