class VirtualMachineType:
    def __init__(
        self,
        name,
        mips,
        price,
        bandwidth,
        cycle_time=3600,
        startup_delay=0,
        cpu_factor=1,
        net_factor=1,
    ):
        self.name = name
        # "Million Instructions Per Second"（每秒百万条指令），是 CPU 性能的衡量标准。
        self.mips = mips
        # 表示每个计算周期的费用，反映计算资源的价格。
        self.cycle_price = price
        # 表示一个计算周期所需的时间 ？？？
        self.cycle_time = cycle_time
        # 表示启动延迟，模拟计算资源启动所需的时间。
        self.startup_delay = startup_delay
        self.bandwidth = bandwidth
        self.net_factor = net_factor
        # error
        self.cpu_factor = cpu_factor
        # error
        self.ram = "not important in this implementation!"
        self.disk_size = "not important in this implementation!"
        self.unfinished_tasks_number = 0

    def waitingTime(self):
        return self.startup_delay

    def exeTime(self, task):
        return (task.length / self.mips) * self.cpu_factor

    def getLength(self, time):
        return (time / self.cpu_factor) * self.mips

    def transferTime(self, file):
        return (file.size / self.bandwidth) * self.net_factor

    def transferTime4Size(self, size):
        return (size / self.bandwidth) * self.net_factor

    def getSize(self, time):
        return (time / self.net_factor) * self.bandwidth

    def isVMType(self):
        return True

    def isProvisionedVM(self):
        return False

    def __str__(self):
        return "VM Type (name: {}, mips: {}, cycle_price: {})".format(
            self.name, self.mips, self.cycle_price
        )

    def __repr__(self):
        return "VM Type {}".format(self.name)
