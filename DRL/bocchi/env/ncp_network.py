import networkx as nx
import numpy as np
import random

# from .virtual_machine import VirtualMachine

# 计算能力、存储能力、计算价格
CSC = np.array(
    [
        [151.6, 28288317.563, 0.532],
        [177.0, 18576916.726, 2.461],
        [26.1, 19584156.692, 2.212],
        [84.4, 23337713.24, 1.835],
        [86.2, 1025500.4496, 1.939],
        [167.9, 3120471.7555, 1.583],
        [25.6, 380684647.36, 0.482],
        [195.1, 1715997.7084, 2.314],
        [58.5, 361713.88047, 1.273],
        [135.6, 140482.82333, 1.766],
        [170.4, 2503548.2787, 2.037],
        [76.7, 439538832.38, 1.541],
        [57.7, 43591426.006, 0.994],
        [20.7, 45837862.033, 1.313],
        [36.6, 32398732.388, 1.938],
        [61.9, 262831891.35, 2.909],
        [73.3, 442259655.64, 1.612],
        [124.1, 3676398.4797, 0.67],
        [67.5, 1605682.2543, 2.908],
        [130.1, 4408984.8434, 2.566],
    ]
)


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

    def getLength(self, time):
        return time * self.compute_capacity

    def transferTime(self, file):
        return file.size / self.bandwidth

    def transferTime4Size(self, size):
        return size / self.bandwidth

    def getSize(self, time):
        return time * self.bandwidth

    def isVMType(self):
        return True

    def isProvisionedVM(self):
        return False

    def __str__(self):
        return "node Type (node_id: {}, compute_capacity: {}, cycle_price: {})".format(
            self.node_id, self.compute_capacity, self.cycle_price
        )


class NodeNetworkGraph:
    def __init__(self):
        self.graph = nx.Graph()  # 有向图

    def add_node(self, node):
        self.graph.add_node(node)

    def add_edge(self, node1, node2, bandwidth):
        self.graph.add_edge(node1, node2, bandwidth=bandwidth)

    def get_neighbors(self, node_id):
        return list(self.graph.neighbors(node_id))  # 返回某个节点的邻居节点

    def create_adjacency_matrix(self):
        return nx.to_numpy_array(self.graph, weight="bandwidth")  # 返回邻接矩阵

    def get_nodes(self):
        return list(self.graph.nodes)

    def get_edges_bandwidth(self, node1, node2):
        return self.graph[node1][node2]["bandwidth"]


def create_network_graph(
    node_nums,
):
    np.random.shuffle(CSC)
    network_graph = NodeNetworkGraph()
    network_num = 1.0
    for i in range(node_nums):
        node = Node(
            node_id=i + network_num / 10,
            compute_capacity=CSC[i][0],
            storage_capacity=CSC[i][1],
            cycle_price=CSC[i][2],
        )
        network_graph.add_node(node)
    for node1 in network_graph.get_nodes():
        for node2 in network_graph.get_nodes():
            if node1.node_id != node2.node_id:
                bandwidth = random.randint(10000000, 20000000)
                network_graph.add_edge(node1, node2, bandwidth)
    return network_graph


# node_list = create_network_graph(20).get_nodes()
# for node in node_list:
#     print(node.node_id, node.compute_capacity, node.storage_capacity, node.cycle_price)
G = create_network_graph(20)
v = G.get_nodes()[0]
v2 = v
print(G.get_edges_bandwidth(v2, G.get_nodes()[1]))
