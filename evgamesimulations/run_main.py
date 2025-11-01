import mesa

from models.pd.parameters import Parameters
from networks.hypernetx_ import HyperNet

"""MAIN FUNCTION.
This is the main function to run the simulation.
"""


def main():
    # =====================================
    # 初始化模型参数，开始仿真
    # =====================================
    # 读取配置文件
    pass
    # 设置模型参数
    params = Parameters(100)
    # 调用run.py，开始运行仿真
    # model = init_model(params)
    # run_model(model)

    H = HyperNet(100, 100, 0.01)
    print(H)


if __name__ == "__main__":
    main()
