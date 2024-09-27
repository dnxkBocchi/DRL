import weakref
import simpy

from .task import TaskStatus


class VirtualMachine:
    counter = 0

    def __init__(self, env, ncp_type, NCP_network_graph, off_idle=False, debug=False):
        VirtualMachine.counter += 1
        self.id = "vm" + str(ncp_type.node_id)
        self.num = VirtualMachine.counter + 0
        self.debug = debug
        self.env = env
        self.off_idle = off_idle
        self.ncp = ncp_type
        self.g = NCP_network_graph
        self.provision_time = env.now

        # --------------------------

        self.running = False
        self.executing_task = None
        self.workload_finished = False

        self.start_time = 0
        self.release_time = 0
        self.done_tasks = []
        self.disk_items = []
        self.finish_time = 0

        self.unfinished_tasks_number = 0
        self.finished_tasks_number = 0

        # The Store operates in a FIFO (first-in, first-out) order
        self.task_queue = simpy.Store(env)

        self.task_finished_announce_pipe = None
        self.vm_release_announce_pipe = None

    def start(self):
        self.env.process(self.__start())

    # 生成器，用于启动虚拟机（VM）的过程
    def __start(self):

        yield self.env.timeout(self.ncp.startup_delay)
        if self.debug:
            print("[{:.2f} - {:10s}] Start.".format(self.env.now, self.id))
        self.start_time = self.env.now
        self.running = True
        if self.off_idle:
            self.env.process(self.__checkIdle())
        self.env.process(self.__cpu())

    # 检查 VM 是否处于空闲状态
    def __checkIdle(self):
        while self.running:
            yield self.env.timeout(self.ncp.cycle_time)
            if self.isIdle() and self.workload_finished:
                # if self.vm_release_announce_pipe != None:
                self.vm_release_announce_pipe.put(self)
                self.release_time = self.env.now
                self.running = False

    # 提交任务到 VM 的任务队列
    def submitTask(self, task):
        self.unfinished_tasks_number += 1
        self.estimateFinishTime(task)
        yield self.task_queue.put(task)
        task.vqueue_time = self.env.now
        task.star_time_file_transferring = self.env.now
        task.status = TaskStatus.wait
        # 保存 VM 引用， 创建对对象的弱引用
        task.vm_ref = weakref.ref(self)

        if self.debug:
            print(
                "[{:.2f} - {:10s}] {} task is submitted to vm queue, total size {}.".format(
                    self.env.now,
                    self.id,
                    task.id,
                    self.unfinished_tasks_number,
                )
            )

    # 估算给定任务的完成时间
    def estimateFinishTime(self, task):
        vms = []
        transfer_time = []
        # 遍历任务的前驱任务（task.pred），识别与当前 VM 不同的 VM，并将其添加到 vms 列表中
        for ptask in task.pred:
            if not ptask.isEntryTask() and ptask.vm_ref().id != self.id:
                if ptask.vm_ref not in vms:
                    vms.append(ptask.vm_ref)
        for v in vms:
            total_size = 0
            files = task.input_files + []
            for file in files:
                if file in v().disk_items:
                    total_size += file.size
                    files.remove(file)

            transfer_time.append(
                v().ncp.transferTime4Size(total_size, self.ncp, self.g)
            )

        # 确定最大传输时间
        trans_time = max(transfer_time) if transfer_time else 0
        # 计算等待时间和传输时间
        task.estimate_waiting_time = max(trans_time, self.waitingTime())
        task.estimate_transfer_time = max(trans_time - self.waitingTime(), 0)

        task.estimate_finish_time = (
            self.env.now + task.estimate_waiting_time + self.ncp.exeTime(task)
        )
        self.finish_time = task.estimate_finish_time

    # 管理任务的状态和资源的使用
    def __exeProcess(self, task):
        self.executing_task = task
        task.start_time = self.env.now
        task.status = TaskStatus.run
        if self.debug:
            print(
                "[{:.2f} - {:10s}] {} task is start executing.".format(
                    self.env.now, self.id, task.id
                )
            )

        yield self.env.timeout(self.ncp.exeTime(task))

        task.finish_time = self.env.now
        # task.status = TaskStatus.done;
        self.done_tasks.append(task)
        self.finished_tasks_number += 1

        # make output files
        # 更新磁盘项，将任务输出的文件添加到磁盘项中。
        self.disk_items += task.output_files

        if self.debug:
            print(
                "[{:.2f} - {:10s}] {} task is finished, use time: {}.".format(
                    self.env.now, self.id, task.id, task.finish_time - task.start_time
                )
            )

        self.unfinished_tasks_number -= 1
        self.executing_task = None
        self.task_finished_announce_pipe.put(task)

    # 从任务队列中获取任务并执行
    def __cpu(self):
        while self.running:
            task = yield self.task_queue.get()
            task.cpu_disposal_time = self.env.now

            # I/O
            if task.estimate_transfer_time:
                yield self.env.timeout(task.estimate_transfer_time)

            self.disk_items += task.input_files
            task.files_transfered = True

            # CPU
            yield self.env.process(self.__exeProcess(task))

    def currentTaskRunningTime(self):
        return self.env.now - self.executing_task.start_time

    def waitingTime(self):
        return max(self.finish_time - self.env.now, 0)

    def runningTime(self):
        if self.running:
            return self.env.now - self.start_time
        return 0

    def timeToStart(self):
        return (
            0
            if self.running
            else self.ncp.startup_delay - (self.env.now - self.provision_time)
        )

    def gap2EndCycle(self):
        return self.ncp.cycle_time - (self.finish_time % self.ncp.cycle_time)

    def isProvisionedVM(self):
        return True

    def isVMncp(self):
        return False

    def isIdle(self):
        return self.running and self.unfinished_tasks_number == 0

    @staticmethod
    def reset():
        VirtualMachine.counter = 0

    def __repr__(self):
        return "{}".format(self.id)
