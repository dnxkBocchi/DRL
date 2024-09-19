# RDWS
A real-time workflow scheduling system in a cloud computing environment with a deep reinforcement learning approach.

Please download the workflows from https://download.pegasus.isi.edu/misc/SyntheticWorkflows.tar.gz.
Put the DAX files in an appropriate folder in workflows directory.


## 工作流程
### 1.agent = DQNScheduler
DQN的初始化，定义网络，优化函数，reward
### 2.train->runEnv
创建SimPy环境对象，初始化六个虚拟机拥有的资源
### 3.workload = Workload
初始化workload信息，将workload.__run()提交到simply
### 4.workload.__run()
向工作流管道 self.workflow_submit_pipe 发送DAX的文件路径
### 5.runEnv.__poolingProcess()
tasks, files = parseDAX：取出DAX路径，通过parseDAX得到task属性信息
wf = Workflow: 一个workflow对象就是一个任务DAG对象，并且建立task和workflow的相互set
workflow_pool.append(wf)：向pool添加wf的信息，创建基于最快，最便宜VM的工作流的截止日期、预算、总执行时间
__addToReadyQueue：向准备队列添加入口任务的后继，更新这些任务的状态和准备时间
### 6.runEnv.__schedulingProcess()
threeDeadline：计算任务的截止时间
tasks_ready_queue：对队列中的每个任务进行时间最短排序
choosed_task = tasks_ready_queue.pop(0)：最优任务对其选择VM
choosed_vm, q = taskScheduler：通过定义的Scheduler得到选择的VM
sim.process(nvm.submitTask(choosed_task))：将任务 choosed_task 提交到虚拟机 nvm 上进行处理
### 7.scheduler
createState:根据当前任务、虚拟机列表、队列、剩余任务、总任务数和当前时间创建当前的状态。
selectAction：得到基于该状态的动作，reward：计算奖励
memory.store(*self.transition)：存储在经验回放缓冲区
train：当经验缓冲区足够时，开始训练
updateModel：更新模型参数
### 8.runEnv.__queueingProcess()
处理任务完成后的调度逻辑。它负责更新任务的状态，计算工作流的 makespan，并将准备好调度的子任务加入到任务队列中。
### 9.runEnv.__releasingProcess()
负责在虚拟机完成任务后，释放该虚拟机并更新相关的状态
### 10.lastFunction()
统计了已完成工作流的总体情况