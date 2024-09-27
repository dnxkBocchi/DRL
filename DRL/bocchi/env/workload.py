import os
import random

import numpy


class Workload:
    counter = 0

    def __init__(
        self,
        env,
        workflow_submit_pipe,
        wf_path,
        arrival_rate,
        max_wf_number=float("inf"),
        max_time=float("inf"),
        initial_delay=0,
        random_seed=-1,
        dax=None,
        debug=False,
    ):
        Workload.counter += 1
        self.id = "wl" + str(Workload.counter)
        self.env = env
        self.dax = dax
        self.workflow_submit_pipe = workflow_submit_pipe
        self.workflow_path = wf_path
        self.arrival_rate = arrival_rate
        # workflow per min
        self.initial_delay = initial_delay
        # 工作流处理或仿真的总时长限制。
        self.max_time = max_time
        # 限制系统中同时存在的工作流数量。
        self.max_wf_number = max_wf_number
        self.debug = debug
        self.finished = False
        self.rand_seed = random.randint(0, 10000) if random_seed == -1 else random_seed
        self.rand_state = numpy.random.RandomState(self.rand_seed)
        # 使用生成的随机种子初始化 Python 的内置随机数生成器。
        random.seed(self.rand_seed)

        # 初始化已提交工作流的数量为 0
        self.__submitted_wf_number = 0
        # 用于存储工作流延迟数据
        self.delays = []
        # 存储工作流对象。
        self.workflows = []
        self.__times = []

        # starts the __run() method as a SimPy process
        # env.process(self.__run()) 将 __run() 方法注册为一个 SimPy 进程，
        # 这意味着 __run() 方法会被调度执行，并能够在仿真中与其他进程和事件交互。
        # __run() 方法实际上并不会立即被调用。它只是被传递给 env.process() 方法，并注册为一个 SimPy 进程。
        env.process(self.__run())

    def __poissonDistInterval(self):
        # k = 0 and lambda = wf_per_second
        wf_per_interval = self.arrival_rate
        return self.rand_state.poisson(1.0 / wf_per_interval)
        # return numpy.random.RandomState().exponential(1.0 / wf_per_interval);

    # 模拟了在一个 SimPy 环境中提交工作流的过程，直到满足预定的条件（时间或最大工作流数量）
    # 逐步提交工作流的过程
    # 向工作流管道 self.workflow_submit_pipe 发送DAX的文件路径
    def __run(self):
        if self.dax:
            yield self.workflow_submit_pipe.put(self.dax)
            self.finished = True
            yield self.workflow_submit_pipe.put("end")
            return

        yield self.env.timeout(self.initial_delay)

        # submit workflows until reach the max_time or max_wf_number
        while (
            self.env.now < self.max_time
            and self.__submitted_wf_number < self.max_wf_number
        ):
            random.seed(self.rand_seed + self.__submitted_wf_number)
            # Choose random DAX file from workflow_path

            if isinstance(self.workflow_path, list):
                wf_path = random.choice(self.workflow_path)
            elif os.path.isdir(self.workflow_path):
                if not hasattr(self, "cached_dax_files"):
                    # 如果没有缓存的文件列表，先缓存起来
                    self.cached_dax_files = [
                        f for f in os.listdir(self.workflow_path) if f[0] != "."
                    ]
                dax = random.choice(self.cached_dax_files)
                wf_path = self.workflow_path + "/" + dax
            else:
                yield self.workflow_submit_pipe.put(self.workflow_path)
                self.finished = True
                yield self.workflow_submit_pipe.put("end")
                return

            # 生成的随机时间间隔控制工作流提交的频率。这模拟了工作流在系统中被随机提交的情况。
            interval = self.__poissonDistInterval()
            yield self.env.timeout(interval)

            # 每次提交后记录延迟和当前时间，以便统计和分析。
            self.delays.append(interval)
            self.__times.append(self.env.now)

            if self.debug:
                print(
                    "[{:.2f} - {:10s}] workflow {} ({}) submitted.".format(
                        self.env.now, "Workload", self.__submitted_wf_number, dax
                    )
                )

            self.workflows.append(wf_path)
            self.__submitted_wf_number += 1

            yield self.workflow_submit_pipe.put(wf_path)

        # 当所有工作流提交完成，发送 "end" 信号，表示不再有新的工作流提交。
        self.finished = True
        yield self.workflow_submit_pipe.put("end")

    @staticmethod
    def reset():
        Workflow.counter = 0
        Workload.counter = 0

    def __str__(self):
        return "Workload (id: {}, workflow_path: {}, arrival_rate: {})".format(
            self.id, self.workflow_path, self.arrival_rate
        )

    def __repr__(self):
        return "{}".format(self.id)


class Workflow:
    counter = 0

    def __init__(self, tasks, files=None, path="", submit_time=0, union=False):
        Workflow.counter += 1
        self.id = "wf" + str(Workflow.counter)
        self.path = path
        self.user = "not important in this implementation!"

        self.fastest_exe_time = 0
        self.deadline_factor = 0
        self.cheapest_exe_cost = 0
        self.budget_factor = 0

        self.deadline = 0
        self.budget = 0
        self.used_budget = 0
        self.submit_time = submit_time

        self.estimate_cost = 0
        self.remained_length = 0

        self.levels = []

        self.finished_tasks = []
        self.new_ready_tasks = 1

        self.tasks = tasks
        self.files = files

        self.exit_task = None
        self.entry_task = None

        self.cost = 0
        self.makespan = 0

        for task in tasks:
            self.remained_length += task.length
            task.setWorkflow(self)

            if task.isEntryTask():
                self.entry_task = task
            elif task.isExitTask():
                self.exit_task = task

            for input_file in task.input_files:
                input_file.consumer_tasks_id.append(task.id)

            for output_file in task.output_files:
                output_file.producer_task_id = task.id

            # print(task.id, task.height_len)

    def getTaskNumber(self):
        return len(self.tasks)

    @staticmethod
    def reset():
        Workflow.counter = 0

    def __str__(self):
        return "Workflow (id: {}, path: {})".format(self.id, self.path)

    def __repr__(self):
        return "{}".format(self.id)
