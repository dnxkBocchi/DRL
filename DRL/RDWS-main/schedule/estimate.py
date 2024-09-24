import math


def exeTime(task, vm):
    return task.length / (vm.mips if vm.isVMType() else vm.type.mips)


def exeCost(task, vm):
    # a = task.length / (vm.mips if vm.isVMType() else vm.type.mips)
    # 调用 exeTime 计算任务执行时间，再除以 vm.cycle_time 以确定执行任务需要多少个周期，并用 math.ceil 向上取整。
    if vm.isVMType():
        return math.ceil(exeTime(task, vm) / vm.cycle_time) * vm.cycle_price

    else:
        # exe_time = max(exeTime(task, vm) - vm.gap2EndCycle(), 0)
        return math.ceil(exeTime(task, vm) / vm.type.cycle_time) * vm.type.cycle_price


def transferTime(size, bandwidth):
    return size / bandwidth


def parentChildTransferTime(parent_task, child_task, vm):
    transfer_size = 0
    for out_file in parent_task.output_files:
        if out_file in child_task.input_files:
            transfer_size += out_file.size
    return transferTime(
        transfer_size, vm.bandwidth if vm.isVMType() else vm.type.bandwidth
    )


def totalOutputTransferTime(task, vm_list):
    transfer_size = sum(file.size for file in task.output_files)
    bandwidth = max(vm_list, key=lambda v: v.type.mips).type.bandwidth
    return transferTime(transfer_size, bandwidth)


def totalInputTransferTime(task, vm):
    # print("totalInputTransferTimetotalInputTransferTimetotalInputTransferTimetotalInputTransferTime")
    transfer_size = 0
    bandwidth = 0
    if vm.isVMType():
        transfer_size = sum(file.size for file in task.input_files)
        bandwidth = vm.bandwidth
    else:
        bandwidth = vm.type.bandwidth
        for in_file in task.input_files:
            if in_file not in vm.disk_items:
                transfer_size += in_file.size

    return transferTime(transfer_size, bandwidth)


# 计算给定任务从其父任务传输输入文件所需的最大时间
def maxParentInputTransferTime(task, vm):
    transfer_size = 0
    # 遍历当前任务的输入文件 task.input_files，检查这些文件是否存在于父任务 p 的输出文件 p.output_files 中。
    # 如果是，将该文件的大小 f.size 累加到 a 中。
    for p in task.pred:
        size_all = []
        a = 0
        for f in task.input_files:
            if f in p.output_files:
                a += f.size
        transfer_size = a if a > transfer_size else transfer_size

    # 计算并返回最大传输时间
    return transferTime(
        transfer_size, vm.bandwidth if vm.isVMType() else vm.type.bandwidth
    )
