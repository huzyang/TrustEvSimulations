#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : huzyang
# @File    : networkx_.py
# @Software: VSCode
# @DATE    : 2024/11/07

import networkx as nx
from networkx.classes.graph import Graph
import pandas as pd
import matplotlib.pyplot as plt
import os

from evgamesimulations.utils.common_utils import CommonUtils

__all__ = [
    "generate_ba_scale_free_network",
    "read_net_from_file",
]


def generate_ba_scale_free_network(n=36, m=6):
    """
    生成一个 BA 无标度网络。

    参数:
    n (int): 节点数
    m (int): 每个新加入的节点连接到 m 个已有节点

    返回:
    G (networkx.Graph): 生成的 BA 无标度网络
    """
    G = nx.barabasi_albert_graph(n, m)
    filepath = CommonUtils.get_project_root_path() + f"/datas/ba_scale_free_{m}_node-{n}.csv"
    save_network_to_csv(G, filepath)
    return G


def save_network_to_csv(G, filepath):
    """
    将网络保存为 CSV 文件。

    参数:
    G (networkx.Graph): 网络
    filepath (str): 保存文件的路径
    """
    edges = list(G.edges())
    df = pd.DataFrame(edges, columns=["source", "target"])
    df.to_csv(filepath, mode="w", index=False)


def draw_network(G):
    """
    绘制网络。

    参数:
    G (networkx.Graph): 网络
    """
    # 计算节点的度数
    degrees = dict(G.degree())
    max_degree = max(degrees.values())

    # 根据度数设置节点颜色和大小
    node_colors = [degrees[node] for node in G.nodes()]
    node_sizes = [v * 100 for v in degrees.values()]

    # 使用spring布局
    pos = nx.spring_layout(G, k=0.3, iterations=50)

    plt.figure(figsize=(12, 12))

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.coolwarm, alpha=0.8)

    # 绘制边
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color="gray")

    # 绘制节点标签
    # nx.draw_networkx_labels(
    #     G, pos, font_size=10, font_family='sans-serif', font_color='black'
    # )

    # 添加颜色条
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=max_degree))
    # sm.set_array([])
    # plt.colorbar(sm, label='Node Degree')

    # 添加标题
    plt.title("BA Scale-Free Network", fontsize=16)

    # 移除坐标轴
    plt.axis("off")

    plt.show()


def calculate_degree_distribution(G):
    """
    计算网络的度分布。

    参数:
    G (networkx.Graph): 网络

    返回:
    degree_distribution (list): 度分布列表
    """
    degree_distribution = nx.degree_histogram(G)
    return degree_distribution


def plot_degree_distribution(degree_distribution):
    """
    绘制度分布图。

    参数:
    degree_distribution (list): 度分布列表
    """
    degrees = range(len(degree_distribution))
    plt.figure(figsize=(10, 6))
    plt.scatter(degrees, degree_distribution, color="b", alpha=0.7)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True)
    plt.show()


def read_net_from_file(filepath: str) -> Graph:
    """
    从 CSV 文件读取网络数据并生成 NetworkX 图。

    参数:
    filepath (str): 包含网络数据的 CSV 文件路径

    返回:
    G (networkx.Graph): 从文件中读取的网络
    """
    # 读取 CSV 文件
    df = pd.read_csv(filepath)
    
    # 创建一个新的空图
    G = nx.Graph()
    
    # 添加边到图中
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'])
    
    return G


def get_root_path() -> str:
    """get the root path"""

    cur_path = os.path.abspath(os.path.dirname(__file__))
    # Obtain the root path of the project, which is the name of the current project
    root_path = cur_path[: cur_path.find("EvGameSimulations") + len("EvGameSimulations")]
    return root_path


def main():
    filepath = CommonUtils.get_project_root_path() + "/datas/hypergraph_3-uniform_1024.csv"
    G = generate_ba_scale_free_network()

    # 绘制网络
    draw_network(G)

    # 计算度分布
    # degree_distribution = calculate_degree_distribution(G)

    # 绘制度分布图
    # plot_degree_distribution(degree_distribution)

    print("Done!")


if __name__ == "__main__":
    main()
