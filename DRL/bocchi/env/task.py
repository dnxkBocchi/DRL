import enum
import re
import xml.etree.ElementTree as ET


class TaskStatus(enum.Enum):
    none = 0
    pool = 1
    # 2: add to ready queue to be scheduled
    ready = 2
    wait = 3
    # 4: The task is running
    run = 4
    done = 5


class Task:
    def __init__(self, num, length):
        self.id = "wf?-" + str(num)
        self.num = num
        self.length = length
        self.uprank = 0
        self.downrank = 0
        self.rank_exe = 0
        self.rank_trans = 0
        self.workflow = None
        self.status = TaskStatus.none
        self.depth = -1
        self.height_len = -1
        # task.depth_len 记录的是从源任务到当前任务所经历的所有任务的总执行时间或长度。
        self.depth_len = -1
        self.succ = []
        self.pred = []
        self.input_files = []
        self.not_real_input_files = []
        self.output_files = []
        self.level = -1
        # 关键路径截止时间,即任务的最早完成时间
        self.deadline_cp = 0

        # 任务进入工作流的准备时间
        self.ready_time = 0
        # 任务开始调度的时间
        self.schedule_time = 0
        self.vqueue_time = 0
        self.cpu_disposal_time = 0
        self.start_time = 0
        self.finish_time = 0

        self.store_in_temp = False

        self.LP = -1
        # Last Parent
        self.EST = -1
        self.ECT = -1
        self.EFT = -1

        self.BFT = -1
        self.LFT = -1

        self.next_children = []
        self.estimated_cost = -1
        self.cost = 0
        self.hard_deadline = -1
        self.soft_deadline = -1
        self.deadline = -1
        self.budget = -1

        # 保存 VM 引用， 创建对象的弱引用
        self.vm_ref = None
        self.files_transfered = False
        self.star_time_file_transferring = 0
        self.input_size = 0
        self.estimate_waiting_time = 0
        self.estimate_finish_time = 0

        # Temp: for calculate Slack Time and scheduling
        # 将虚拟机作为键，时间和成本作为值，存储在任务的 vref_time_cost 属性中
        self.fast_run = 0
        self.vref_time_cost = {}  #  dictionary {v1:[2,0.5] v2: [23, 5] ......}

    def setWorkflow(self, wf):
        self.workflow = wf
        self.id = str(wf.id) + "-" + str(self.num)

    def isReadyToSch(self):
        for parent in self.pred:
            if parent.status != TaskStatus.done:
                return False
        return True

    def isAllChildrenDone(self):
        for child in self.succ:
            if child.status != TaskStatus.done:
                return False
        return True

    def isAllChildrenStoredInTemp(self):
        if self.succ[0].isExitTask():
            return True

        for child in self.succ:
            if not child.store_in_temp:
                return False
        return True

    def isEntryTask(self):
        return len(self.pred) == 0

    def isExitTask(self):
        return len(self.succ) == 0

    def __str__(self):
        return "Task (id: {}, depth: {}, length: {},\n pred: {}, succ: {})".format(
            self.id, self.depth, self.length, self.pred, self.succ
        )

    def __repr__(self):
        return "{}".format(self.id)


class File:
    def __init__(self, name, size):
        self.name = str(name)
        self.size = float(size)
        self.real_input = True  # not produced by a task
        self.producer_task_id = None
        self.consumer_tasks_id = []

    def __str__(self):
        return "File (name: {}, size: {}, consumers: {})".format(
            self.name, self.size, self.consumer_tasks_id
        )

    def __repr__(self):
        return self.name


def setTaskDepth(task, d, l):
    # print(task)
    if task.depth < d:
        task.depth = d
    if task.depth_len < l:
        task.depth_len = l

    for child_task in task.succ:
        setTaskDepth(child_task, task.depth + 1, task.depth_len + task.length)


def parseDAX(xmlfile, merge=False):
    tasks = []
    files = []

    def convertTaskRealIdToNum(id_str):
        return int(re.findall("\d+", id_str)[0]) + 1

    def getTask(num):
        for task in tasks:
            if task.num == num:
                return task

    tree = ET.parse(xmlfile)
    root = tree.getroot()
    for node in root:
        if "job" in node.tag.lower():
            # 获取任务 ID 和运行时间，并创建 Task 对象，然后将其添加到 tasks 列表中。
            num = convertTaskRealIdToNum(node.attrib.get("id"))
            runtime = float(node.attrib.get("runtime")) * 1000
            # in WorkflowSim, runtime multiplied by 1000.
            task = Task(num, runtime)
            tasks.append(task)

            # 被用来检查文件的大小、名称，以及它是输入文件还是输出文件。
            for file in node:
                if "uses" in file.tag.lower():
                    file_size = float(file.attrib.get("size"))
                    # Byte
                    file_name = file.attrib.get("name")
                    # DAX v3.3
                    if file_name == None:
                        file_name = file.attrib.get("file")
                        # DAX v3

                    if file.attrib.get("link") == "output":
                        file_alredy_exist = None
                        for file in files:
                            if file_name == file.name:
                                file_alredy_exist = file
                                task.output_files.append(file)
                                file.real_input = False

                        if not file_alredy_exist:
                            file_item = File(file_name, file_size)
                            files.append(file_item)
                            task.output_files.append(file_item)
                            file_item.real_input = False

                    elif file.attrib.get("link") == "input":

                        file_alredy_exist = None
                        for file in files:
                            if file_name == file.name:
                                file_alredy_exist = file
                                task.input_files.append(file)
                                task.input_size += file.size

                        if not file_alredy_exist:
                            file_item = File(file_name, file_size)
                            task.input_files.append(file_item)
                            files.append(file_item)

        # 如果节点标签包含 child，则表示这是一个任务之间的依赖关系。
        elif "child" in node.tag.lower():
            # 将这个 ID 转换为一个实际的任务编号（或任务对象的标识符）。
            child_num = convertTaskRealIdToNum(node.attrib.get("ref"))
            # 根据子任务的编号获取对应的 Task 对象。
            child = getTask(child_num)

            # 在任务之间建立依赖关系。
            for parent in node:
                parent_num = convertTaskRealIdToNum(parent.attrib.get("ref"))
                parent = getTask(parent_num)
                child.pred.append(parent)
                parent.succ.append(child)

    # Add an entry task and an exit task to the workflow
    roots = []
    lasts = []

    # 处理头结点和尾节点的依赖关系
    for task in tasks:
        task.depth = 0
        if len(task.pred) == 0:
            roots.append(task)
        elif len(task.succ) == 0:
            lasts.append(task)

        for f in task.input_files:
            if not f.real_input:
                task.not_real_input_files.append(f)
                # task.original_input_size += f.size

        for f in task.not_real_input_files:
            task.input_files.remove(f)

    entry_num = 0
    exit_num = -1

    entry_task = Task(entry_num, 0)
    # entry_task.depth = 0;
    exit_task = Task(exit_num, 0)
    # exit_task.depth = 0;

    # 处理头结点和尾节点的依赖关系
    for task in roots:
        task.pred.append(entry_task)
        entry_task.succ.append(task)

        for f in task.input_files:
            entry_task.output_files.append(f)

    for task in lasts:
        task.succ.append(exit_task)
        exit_task.pred.append(task)
        for f in task.output_files:
            exit_task.input_files.append(f)

    tasks.append(entry_task)
    tasks.append(exit_task)

    # Calculate each task's depth
    setTaskDepth(entry_task, 0, 0)

    return tasks, files
