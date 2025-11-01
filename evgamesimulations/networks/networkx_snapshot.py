import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import os
import datetime
from matplotlib.colors import Normalize
from evgamesimulations.utils.common_utils import CommonUtils


# 构建网络
def build_network(network_df):
    G = nx.Graph()
    for index, row in network_df.iterrows():
        node1 = row["source"]
        node2 = row["target"]
        G.add_edge(node1, node2)
    return G


# 绘制策略快照图
def plot_strategy_frame(G, strategy_df, steps, node_pos, is_animation):
    for step in steps:
        node_colors = []
        edge_colors = []
        node_sizes = []
        edge_width = 0.5
        label_font_size = 8
        label_font_color = "black"
        # 定义策略对应的颜色，使用更具区分度的颜色
        strategy_colors = {
            "I": sns.color_palette("husl", 3)[0],
            "T": sns.color_palette("husl", 3)[1],
            "U": sns.color_palette("husl", 3)[2],
        }

        # 计算节点的度
        degrees = dict(G.degree())
        max_degree = max(degrees.values())

        current_step_strategy = strategy_df[strategy_df["Step"] == step]
        strategy_dict = dict(zip(current_step_strategy["AgentID"], current_step_strategy["Strategy"]))

        for node in G.nodes():
            try:
                strategy = strategy_dict.get(node)
                node_colors.append(strategy_colors[strategy])
                # 根据节点度调整节点大小
                node_sizes.append(300 * (degrees[node] / max_degree + 0.2))
            except IndexError:
                print(f"未找到节点 {node} 在时间步 {step} 的策略信息。")
                node_colors.append("gray")
                node_sizes.append(300 * (degrees[node] / max_degree + 0.2))

        # 为每条边根据两端节点度的和来确定颜色深浅
        degree_sums = [degrees[u] + degrees[v] for u, v in G.edges()]
        max_degree_sum = max(degree_sums)
        norm = Normalize(vmin=0, vmax=max_degree_sum)
        rocket_palette = sns.color_palette("rocket", as_cmap=True)

        for degree_sum in degree_sums:
            edge_colors.append(rocket_palette(norm(degree_sum)))

        # 设置图片清晰度
        plt.rcParams["figure.dpi"] = 300
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(G, node_pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(G, node_pos, edge_color=edge_colors, width=edge_width)
        # nx.draw_networkx_labels(G, node_pos, font_size=label_font_size, font_color=label_font_color)
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", label=key, markerfacecolor=strategy_colors[key], markersize=10)
            for key in strategy_colors.keys()
        ]
        plt.legend(handles=legend_elements, loc="best")
        plt.title(f"Strategy Snapshot - Step {step}", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        if is_animation:
            plt.savefig(CommonUtils.get_project_root_path() + f"/outputs/frames/frame_{step}.png")
            plt.close()
        else:
            plt.show()


# 生成动图或视频
def generate_animation(steps):
    """
    绘制所有时间步策略快照图，并合并成一个动图
    """
    frames = []
    for step in steps:
        image_path = CommonUtils.get_project_root_path() + f"/outputs/frames/frame_{step}.png"
        if os.path.exists(image_path):
            frames.append(imageio.imread(image_path))
        else:
            print(f"图片 {image_path} 不存在。")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if len(steps) < 100:  # 可以根据实际情况调整这个阈值
        imageio.mimsave(
            CommonUtils.get_project_root_path() + f"/outputs/frames/{timestamp}_strategy_evolution.gif", frames, fps=2
        )
    else:
        writer = imageio.get_writer(
            CommonUtils.get_project_root_path() + f"/outputs/frames/{timestamp}_strategy_evolution.mp4", fps=2
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()


if __name__ == "__main__":
    network_file = CommonUtils.get_project_root_path() + "/datas/ba_scale_free_4_node-1024.csv"
    strategy_file = CommonUtils.get_project_root_path() + "/outputs/20250416121809_Trust_Game_V1_SF_1024_all_evo.csv"

    # 读取网络 CSV 文件
    network_df = pd.read_csv(network_file)
    strategy_df = pd.read_csv(strategy_file)

    G = build_network(network_df)
    # 计算并保存节点布局
    node_pos = nx.spring_layout(G)
    # 示例 steps 数组，可根据需要修改
    steps = list(range(1000))  #  list(range(0, 1000, 5))

    plot_strategy_frame(G, strategy_df, steps, node_pos, True)
    generate_animation(steps)
