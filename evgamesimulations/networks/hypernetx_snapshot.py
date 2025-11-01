import datetime
import pandas as pd
import hypernetx as hnx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import imageio

# import imageio_ffmpeg
import os
from evgamesimulations.utils.common_utils import CommonUtils


# 读取超图数据文件
def read_hypernetwork_csv(file_path):
    df = pd.read_csv(file_path)
    hyperedges = []
    for col in df.columns:
        hyperedge = tuple(df[col].dropna())
        hyperedges.append(hyperedge)
    return hnx.Hypergraph(hyperedges)


# 读取策略数据文件
def read_strategy_csv(file_path):
    return pd.read_csv(file_path)


# 绘制策略快照图
def plot_strategy_frame(H, strategy_df, steps, pos):
    for step in steps:
        node_colors = []
        # 定义策略对应的颜色，使用更具区分度的颜色
        strategy_colors = {
            "I": sns.color_palette("husl", 3)[0],
            "T": sns.color_palette("husl", 3)[1],
            "U": sns.color_palette("husl", 3)[2],
        }

        current_step_strategy = strategy_df[strategy_df["Step"] == step]

        for node in H.nodes:
            try:
                strategy = current_step_strategy.loc[current_step_strategy["AgentID"] == node, "Strategy"].values[0]
                node_colors.append(strategy_colors[strategy])
            except IndexError:
                print(f"未找到节点 {node} 在时间步 {step} 的策略信息。")
                node_colors.append("gray")

        # 设置图片清晰度
        plt.rcParams["figure.dpi"] = 300
        plt.figure(figsize=(8, 6))
        hnx.draw(H, pos=pos, node_color=node_colors, with_edge_labels=False)
        plt.title(f"Strategy Snapshot - Step {step}", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(CommonUtils.get_project_root_path() + f"/outputs/frames/hyper_frame_{step}.png")
        plt.close()


# 生成动图或视频
def generate_animation(steps):
    frames = []
    for step in steps:
        image_path = CommonUtils.get_project_root_path() + f"/outputs/frames/hyper_frame_{step}.png"
        if os.path.exists(image_path):
            frames.append(imageio.imread(image_path))
        else:
            print(f"图片 {image_path} 不存在。")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if len(steps) < 100:  # 可以根据实际情况调整这个阈值
        imageio.mimsave(
            CommonUtils.get_project_root_path() + f"/outputs/frames/{timestamp}_hyper_strategy_evolution.gif",
            frames,
            fps=2,
        )
    else:
        writer = imageio.get_writer(
            CommonUtils.get_project_root_path() + f"/outputs/frames/{timestamp}_hyper_strategy_evolution.mp4", fps=2
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()


if __name__ == "__main__":
    hypernetwork_file = CommonUtils.get_project_root_path() + "/datas/test_4_node.csv"
    strategy_file = CommonUtils.get_project_root_path() + "/outputs/test_strategy_step.csv"

    H = read_hypernetwork_csv(hypernetwork_file)
    strategy_df = read_strategy_csv(strategy_file)

    # 计算并保存节点布局
    pos = hnx.drawing.layouts.layout(H, layout_algorithm="spring")

    # 示例 steps 数组，可根据需要修改
    steps = list(range(1000)) # list(range(0, 1000, 5))

    plot_strategy_frame(H, strategy_df, steps, pos)
    generate_animation(steps)
