import re
import xml.etree.ElementTree as ET

from .file import File
from .task import Task


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

    # merge
    if merge:
        for task in tasks:
            while len(task.succ) == 1 and len(task.succ[0].pred) == 1:
                child = task.succ[0]
                task.num = str(task.num) + "+" + str(child.num)
                task.length += child.length

                for f in child.input_files:
                    if f not in task.output_files and f not in task.input_files:
                        task.input_files.append(f)

                for f in child.output_files:
                    if f not in task.output_files:
                        task.output_files.append(f)

                for t in child.succ:
                    t.pred.remove(child)
                    t.pred.append(task)

                task.succ += child.succ
                task.succ.remove(child)
                tasks.remove(child)

    # 处理头结点和尾节点的依赖关系
    for task in tasks:
        task.depth = 0
        if len(task.pred) == 0:
            roots.append(task)
        elif len(task.succ) == 0:
            lasts.append(task)

        # WORKFLOW SIM
        # * If a input file has an output file it does not need stage-in For
        # * workflows, we have a rule that a file is written once and read many
        # * times, thus if a file is an output file it means it is generated within
        # * this job and then used by another task within the same job (or other jobs
        # * maybe) This is useful when we perform horizontal clustering

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
