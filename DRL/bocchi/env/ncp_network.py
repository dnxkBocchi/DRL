import networkx as nx
import random
import csv
import random


class Node:
    def __init__(
        self,
        node_id,
        compute_capacity,
        storage_capacity,
        cycle_price,
        cycle_time=3600,
        startup_delay=0,
        bandwidth=20000000,
        latency=1,
    ):
        self.node_id = node_id
        self.compute_capacity = compute_capacity
        self.storage_capacity = storage_capacity
        self.cycle_time = cycle_time
        self.cycle_price = cycle_price
        # 带宽
        self.bandwidth = bandwidth
        # 延迟 = 数据大小 / 带宽
        self.latency = latency
        self.unfinished_tasks_number = 0
        # 表示启动延迟，模拟计算资源启动所需的时间。
        self.startup_delay = startup_delay

    def waitingTime(self):
        return self.startup_delay

    def exeTime(self, task):
        return task.length / self.compute_capacity

    def transferTime4Size(self, size, ncp, g):
        return size / g.get_edges_bandwidth(self.node_id, ncp.node_id)

    def __str__(self):
        return "node Type (node_id: {}, compute_capacity: {}, cycle_price: {})".format(
            self.node_id, self.compute_capacity, self.cycle_price
        )


class NodeNetworkGraph:
    def __init__(self):
        self.graph = nx.Graph()  # 有向图

    def add_node(self, node_id):
        self.graph.add_node(node_id)

    def add_edge(self, node1_id, node2_id, bandwidth):
        self.graph.add_edge(node1_id, node2_id, bandwidth=bandwidth)

    def get_neighbors(self, node_id):
        return list(self.graph.neighbors(node_id))  # 返回某个节点的邻居节点

    def create_adjacency_matrix(self):
        return nx.to_numpy_array(self.graph, weight="bandwidth")  # 返回邻接矩阵

    def get_nodes(self):
        return list(self.graph.nodes)

    def get_edges_bandwidth(self, node1_id, node2_id):
        return self.graph[node1_id][node2_id]["bandwidth"]


def extract_data(file_path, num):
    data = []
    with open(file_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # 跳过第一行 (表头)

        for row in reader:
            if len(row) >= 3:  # 确保至少有三列
                second_value = float(row[1])  # 第二列数据
                third_value = float(row[2])  # 第三列数据
                data.append((second_value, third_value))

    random.shuffle(data)
    return data[:num]


def create_ncp_graph(
    node_nums,
):
    network_graph = NodeNetworkGraph()
    network_num = 1.0
    ncps = []
    CSC = extract_data("E:/python/DRL/bocchi/data/ncp.csv", node_nums)
    for i in range(node_nums):
        node = Node(
            node_id=i + network_num / 10,
            compute_capacity=CSC[i][0] / 1000,
            storage_capacity=CSC[i][1] * 100,
            cycle_price=round(CSC[i][0] / 100000 + CSC[i][1] / 100000, 3),
        )
        network_graph.add_node(node.node_id)
        ncps.append(node)
    for node1_id in network_graph.get_nodes():
        for node2_id in network_graph.get_nodes():
            if node1_id != node2_id:
                bandwidth = random.randint(10000000, 20000000)
                network_graph.add_edge(node1_id, node2_id, bandwidth)
            else:
                network_graph.add_edge(node1_id, node2_id, 0)
    return network_graph, ncps


def create_network_graph(network_num, node_nums):
    network_graph = []
    for i in range(network_num):
        ncp_graph, ncps = create_ncp_graph(node_nums)
        network_graph.append((ncp_graph, ncps))
    return network_graph
