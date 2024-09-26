import math
import random
from operator import attrgetter
from env.task import TaskStatus


def exeTime(task, vm):
    return task.length / (vm.compute_capacity)


def exeCost(task, vm):
    # 调用 exeTime 计算任务执行时间，再除以 vm.cycle_time 以确定执行任务需要多少个周期，并用 math.ceil 向上取整。
    return math.ceil(exeTime(task, vm) / vm.cycle_time) * vm.cycle_price


def transferTime(size, bandwidth):
    return size / bandwidth


def trans_time(size, vm1, vm2, g):
    return size / g.get_edges_bandwidth(vm1, vm2)


# 计算给定任务从其父任务传输输入文件所需的最大时间
def maxParentInputTransferTime(task, vm, g):
    transfer_size = 0
    # 遍历当前任务的输入文件 task.input_files，检查这些文件是否存在于父任务 p 的输出文件 p.output_files 中。
    # 如果是，将该文件的大小 f.size 累加到 a 中。
    for p in task.pred:
        a = 0
        for f in task.input_files:
            if f in p.output_files and p.vm_ref is not None:
                a += trans_time(f.size, p.vm_ref, vm, g)
        transfer_size = a if a > transfer_size else transfer_size

    # 计算并返回最大传输时间
    # return transferTime(transfer_size, vm.bandwidth)
    return transfer_size


def setUpwardRank(task, rank):
    if task.uprank < rank:
        task.uprank = rank

    for ptask in task.pred:
        setUpwardRank(ptask, task.uprank + task.rank_trans + ptask.rank_exe)


# 这个函数用于递归计算任务的 下行优先级（downward rank）。
# 下行优先级表示从当前任务开始，所有后继任务中最长路径的总和，类似于任务向前推进时的最长耗时。
def setDownwardRank(task, rank):
    if task.downrank < rank:
        task.downrank = rank

    for ctask in task.succ:
        setDownwardRank(ctask, task.downrank + task.rank_exe + ctask.rank_trans)


# 为工作流 wf 创建截止日期（deadline）。
# 截止日期基于工作流的最快执行时间和一个随机生成的或固定的截止日期因子（deadline_factor）
def createDeadline(
    wf, fastest_vm_type, min_df=1, max_df=20, factor_int=True, constant_df=0
):
    wf.fastest_exe_time = wf.entry_task.uprank

    if constant_df:
        wf.deadline_factor = constant_df
    else:
        wf.deadline_factor = (
            random.randint(min_df, max_df)
            if factor_int
            else random.uniform(min_df, max_df)
        )
    # 四舍五入到小数点后两位
    wf.deadline = round(wf.deadline_factor * wf.fastest_exe_time, 2)


# 为工作流 wf 计算预算（budget）。
# 预算是基于工作流中所有任务在最便宜的虚拟机类型上运行所需的最低成本，并通过乘以一个随机生成或固定的预算因子来得到。
def createBudget(
    wf, cheapest_vm_type, min_bf=1, max_bf=20, factor_int=True, constant_bf=0
):
    # Compute lowest budget for workflow, without data transfer time
    total_time = 0
    for task in wf.tasks:
        total_time += exeTime(task, cheapest_vm_type)

    cycle_num = math.ceil(total_time / cheapest_vm_type.cycle_time)
    wf.cheapest_exe_cost = cycle_num * cheapest_vm_type.cycle_price

    if constant_bf:
        wf.budget_factor = constant_bf
    else:
        wf.budget_factor = (
            random.randint(min_bf, max_bf)
            if factor_int
            else random.uniform(min_bf, max_bf)
        )
    wf.budget = round(wf.budget_factor * wf.cheapest_exe_cost, 2)


# 没搞懂为什么要搞这个深度差，后继肯定都行啊????
def calBFT_LFT(task, now_time, vm_list=[], fast_run=0, slow_run=0):
    # 变量 asap 用来标记任务是否能够立即执行
    asap = True
    succ = []
    # 没搞懂为什么要搞这个深度差，后继肯定都行啊
    for t in task.succ:
        if t.depth - task.depth == 1:
            succ.append(t)
    # 计算后继任务的 EST（Earliest Start Time） 和 EFT（Earliest Finish Time）
    for child in succ:
        child.EST = -1
        child.EFT = -1
        for p in child.pred:
            # 如果后继任务的某个前驱任务还未完成（状态为等待或正在运行），则不能立刻开始执行。
            if (
                p is not task
                and p.status is TaskStatus.wait
                or p.status is TaskStatus.run
            ):
                asap = False
                p.EFT = p.estimate_finish_time - now_time
        child.LP = max(child.pred, key=attrgetter("EFT"))
        child.EST = child.LP.EFT + 0

    if asap:
        BFT = fast_run
        LFT = fast_run
    else:
        BFT = min(succ, key=attrgetter("EST")).EST
        if BFT <= 0:
            BFT = fast_run

        c = max(succ, key=attrgetter("EST"))
        while c.LP.uprank < task.uprank:
            # print(c.id, c.LP.id, c.LP.uprank, task.id ,task.uprank, "---", c.EST)
            succ.remove(c)
            if succ:
                c = max(succ, key=attrgetter("EST"))
            else:
                break

        LFT = c.EST  # LFT
        if BFT > LFT:
            LFT = BFT

    if fast_run and slow_run:
        if BFT < fast_run:
            BFT = fast_run
        elif BFT > slow_run:
            BFT = slow_run

        if LFT < fast_run:
            LFT = fast_run
        elif LFT > slow_run:
            LFT = slow_run
    return BFT, LFT
