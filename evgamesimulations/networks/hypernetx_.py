import hypernetx as hnx, hypernetx.algorithms.generative_models as gm
import matplotlib.pyplot as plt
import pandas as pd
import random
from evgamesimulations.common.hyper_network import HyperNetwork
import os
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from evgamesimulations.utils.common_utils import CommonUtils
from hypernetx import Hypergraph

"""
"""

__all__ = [
    "read_hypergraph_from_file",
    "node_degree_distribution",
    "show_degree_distribution",
    "show_graph",
]


def erdos_renyi_hypergraph(n, m, p):
    """Erdos-Renyi network.
    generate a graph which has n nodes, m edges, probability p.
    Parameters
    ----------
    n: int
        Number of nodes
    m: int
        Number of edges
    p: float
        The probability that a bipartite edge is created
    node_labels: list, default=None
        Vertex labels
    edge_labels: list, default=None
        Hyperedge labels
    """

    n = 1000
    m = n
    p = 0.01
    hypergraph = gm.erdos_renyi_hypergraph(n, m, p)

    df = hypergraph.incidence_dataframe()
    print(df.head())

    # Make sure the directory exists and if it doesn't, create it
    # if not os.path.exists(DIRECTORY):
    #     os.makedirs(DIRECTORY)
    file_name = f"/datas/hypergraph_er_{n}.csv"
    df.to_csv(CommonUtils.get_project_root_path() + file_name, mode="w", index=False)
    return hypergraph

def chung_lu_hypergraph() -> Hypergraph:
    import hypernetx.algorithms.generative_models as gm
    import random
    n = 1024
    k1 = {i : random.randint(1, 120) for i in range(n)}
    k2 = {i : sorted(k1.values())[i] for i in range(n)}
    hypergraph = gm.chung_lu_hypergraph(k1, k2)
    
    df = hypergraph.incidence_dataframe()
    print(df.head())

    # Make sure the directory exists and if it doesn't, create it
    # if not os.path.exists(DIRECTORY):
    #     os.makedirs(DIRECTORY)
    file_name = f"/datas/chung_lu_hypergraph_{n}.csv"
    df.to_csv(CommonUtils.get_project_root_path() + file_name, mode="w", index=False)
    return hypergraph

def g_uniform_hypergraph() -> Hypergraph:
    """
    A function to generate an Erdos-Renyi hypergraph as implemented by Mirah Shi and described for
    bipartite networks by Aksoy et al. in https://doi.org/10.1093/comnet/cnx001

    Parameters
    ----------
    n: int
        Number of nodes
    m: int
        Number of edges
    p: float
        The probability that a bipartite edge is created
    node_labels: list, default=None
        Vertex labels
    edge_labels: list, default=None
        Hyperedge labels

    Returns
    -------
    HyperNetX Hypergraph object


    Example::

    >>> import hypernetx.algorithms.generative_models as gm
    >>> n = 1000
    >>> m = n
    >>> p = 0.01
    >>> H = gm.erdos_renyi_hypergraph(n, m, p)

    """
    n = 1000  # number of nodes
    g = 5  # number of start nodes
    node_degrees = defaultdict(int)  # Store the hyperdegree of each node

    node_label = lambda index: index
    edge_label = lambda index: index
    bipartite_edges = []
    # existing_nodes = set(
    #     i for i in range(g)
    # )  # 初始化 existing_nodes 集合，包含初始节点

    # Step 1 : 初始化超图，生成 g 个初始节点和一个包含所有初始节点的超边
    for u in range(g):
        bipartite_edges.append((edge_label(0), node_label(u)))
        node_degrees[u] = 1

    edge_index = 1  # 记录超边的数量
    # Step 2 和 Step 3: 超边增长和优先连接
    while len(node_degrees) < n:

        # m = random.randint(1, g)  # 随机生成 m 值，1 <= m <= g
        m = 1  # 固定的m值

        # 添加 m 个新节点
        new_nodes = [len(node_degrees) + i for i in range(m)]

        # 根据优先连接机制选择 g - m 个现有节点
        total_degree = sum(node_degrees.values())
        existing_nodes = set()

        while len(existing_nodes) < (g - m):
            # 计算优先连接概率
            probabilities = [node_degrees[node] / total_degree for node in node_degrees]
            selected_node = random.choices(list(node_degrees.keys()), weights=probabilities, k=1)[0]
            existing_nodes.add(selected_node)

        # 形成新的超边
        new_hyperedge = new_nodes + list(existing_nodes)
        for node in new_hyperedge:
            node_degrees[node] += 1  # 更新所有节点的超度
            bipartite_edges.append((edge_label(edge_index), node_label(node)))

        edge_index += 1

    df = pd.DataFrame(bipartite_edges)

    hypergraph = Hypergraph(df, static=True)

    # 注：直接存储df到csv文件会有问题，原因暂时不详。故先转换为hypergraph对象，再转换为incidence_dataframe
    df_t = hypergraph.incidence_dataframe()
    print(df.tail())

    # Make sure the directory exists and if it doesn't, create it
    # if not os.path.exists(DIRECTORY):
    #     os.makedirs(DIRECTORY)
    file_name = f"/datas/hypergraph_{g}-uniform_{n}.csv"
    df_t.to_csv(CommonUtils.get_project_root_path() + file_name, mode="w", index=False)

    return hypergraph


def generate_g_k_uniform_hypergraph(n: int = 1000, g: int = 4, k: int = 4) -> Hypergraph:
    """
    生成一个g-uniform超图，其中：
    - 超边大小 g = g
    - 节点度 k = k
    - 节点数 n = n

    Returns
    -------
    HyperNetX Hypergraph object
    """

    # 计算需要的超边数量
    total_edges = (n * k) // g

    node_degrees = defaultdict(int)  # 存储节点的超度
    bipartite_edges = []

    node_label = lambda index: index
    edge_label = lambda index: index

    # 初始化所有节点
    for node in range(n):
        node_degrees[node] = 0

    edge_index = 0

    # 生成超边直到所有节点都达到度k
    while any(degree < k for degree in node_degrees.values()):
        # 选择度小于k的节点
        available_nodes = [node for node, degree in node_degrees.items() if degree < k]

        if len(available_nodes) < g:
            break

        # 随机选择g个节点形成超边
        selected_nodes = random.sample(available_nodes, g)

        # 添加超边
        for node in selected_nodes:
            bipartite_edges.append((edge_label(edge_index), node_label(node)))
            node_degrees[node] += 1

        edge_index += 1

        # 检查是否达到目标
        if edge_index >= total_edges:
            break

    # 创建DataFrame
    df = pd.DataFrame(bipartite_edges, columns=['edge', 'node'])

    # 创建超图
    hypergraph = Hypergraph(df, static=True)

    # 转换为incidence_dataframe并保存
    df_t = hypergraph.incidence_dataframe()
    print(f"生成的超图信息:")
    print(f"节点数: {len(hypergraph.nodes)}")
    print(f"超边数: {len(hypergraph.edges)}")
    print(f"最后几条边:")
    print(df_t.tail())

    # 保存到文件
    file_name = f"/datas/hypergraph_k-{k}_g-{g}-uniform_{n}.csv"
    df_t.to_csv(CommonUtils.get_project_root_path() + file_name, mode="w", index=False)

    return hypergraph


def example_graph() -> Hypergraph:
    # 20 nodes, 11 edges
    scenes_dictionary = {
        0: ("0", "1", "2"),
        1: ("2", "3"),
        2: ("4", "1", "5"),
        3: ("3", "6", "7", "4"),
        4: ("6", "7", "8", "9", "10", "3", "4"),
        5: ("2", "11"),
        6: ("11", "12"),
        7: ("13", "11"),
        8: ("1", "2", "14"),
        9: ("13", "15", "16"),
        10: ("11", "12", "17", "18", "19"),
    }

    return hnx.Hypergraph(scenes_dictionary)


def read_hypergraph_from_file(filepath: str) -> Hypergraph:
    df = pd.read_csv(filepath)
    # Display the first few lines of the DataFrame to verify that the reads are correct
    assert df is not None
    # print(f"Graph DataFrame:\n {df.head()}")
    hypergraph = hnx.Hypergraph.from_incidence_dataframe(df)

    # self.node_degree_distribution(self.hypergraph)
    return hypergraph


def node_degree_distribution(hypergraph: Hypergraph):
    """统计超图节点的度分布

    Args:
        hypergraph (Hypergraph): _description_
    """
    node_degrees = [len(edges) for node, edges in hypergraph.incidences.memberships.items()]
    # 统计度值的数量
    degree_counts = {}
    for degree in node_degrees:
        degree_counts[degree] = degree_counts.get(degree, 0) + 1

    # 将统计结果转换为 DataFrame
    df = pd.DataFrame(list(degree_counts.items()), columns=["Degree", "Count"])

    # 按照度值从小到大排序
    df = df.sort_values(by="Degree")

    # 重置索引
    df.reset_index(drop=True, inplace=True)
    print(df)
    show_degree_distribution(df)


def show_degree_distribution(df):
    """绘制节点度分布图

    Args:
        df (pd.DataFrame): 统计后的节点度分布数据
    """
    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Degree"], df["Count"], color="blue", s=100, alpha=0.7)

    # 添加标题和轴标签
    plt.title("Node Degree Distribution", fontsize=14)
    plt.xlabel("Degree", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xscale("log")  # 设置 x 轴为对数刻度
    plt.yscale("log")  # 设置 y 轴为对数刻度

    # 显示网格线
    plt.grid(True, linestyle="--", alpha=0.6)

    # 展示图像
    plt.show()


def show_graph(h: Hypergraph):

    # plot graph
    plt.subplots(figsize=(12, 12))
    hnx.draw(h)
    plt.show()


def get_root_path() -> str:
    """get the root path"""

    cur_path = os.path.abspath(os.path.dirname(__file__))
    # Obtain the root path of the project, which is the name of the current project
    root_path = cur_path[: cur_path.find("EvGameSimulations") + len("EvGameSimulations")]
    return root_path


def main():
    # filepath = CommonUtils.get_project_root_path() + "/datas/hypergraph_5-uniform_1024.csv"
    # hypergraph = read_hypergraph_from_file(filepath)
    
    # hypergraph = g_uniform_hypergraph()
    hypergraph = generate_g_k_uniform_hypergraph(1000, 4, 4)
    # hypergraph = chung_lu_hypergraph()
    
    node_degree_distribution(hypergraph)
    # show_graph(hypergraph)
    print("Done!")


if __name__ == "__main__":
    main()
