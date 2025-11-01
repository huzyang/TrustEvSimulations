#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : huzyang
# @Project : EvGameSimulations
# @File    : batch_run.py
# @Software: PyCharm
# @DATE    : 2024/11/10
"""batch_run.py could contain the code to run the model multiple times."""

import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import mesa
import logging
import hypernetx as hnx

from model import NTrustGameV4Model
from evgamesimulations.utils.common_utils import CommonUtils
from evgamesimulations.utils.data_processing import DataProcessing
from parameters import (
    Parameters,
    MODEL_PARAMS,
    BENCH_RUN_SET,
    NET_BASE_TYPE,
    CLASSICALNET_TYPE,
    HYPERNET_TYPE,
    UPDATE_RULE_TYPE,
)

# 配置 logger
console_level = logging.ERROR
log_level = logging.ERROR
logging.basicConfig(
    filename="run.log", filemode="w", level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
)


def setup_logging():
    # 配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 创建文件处理器
    file_handler = logging.FileHandler("run.log", mode="w")
    file_handler.setLevel(log_level)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    # 定义日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# 调用日志配置函数
setup_logging()
logger = logging.getLogger(__name__)


def print_dict(dict):
    for k, v in dict.items():
        print(f"{k}: {v}")


def save_results(results: list[dict], filename):
    """
    Save the results of an experiment.
    """
    all_datas = pd.DataFrame(results)
    # file name should be ends with all.csv

    # 修改后：
    output_path = os.path.join(CommonUtils.get_project_root_path(), "outputs", f"{filename[:-4]}_all.csv")
    all_datas.to_csv(output_path, mode="w", index=False)

    logger.info(f"{all_datas.tail()} \n Batch Results were saved into {filename}")
    print(f"{all_datas.tail()} \n Batch Results were saved into {filename}")


def main():

    params = Parameters()
    iterations = int(BENCH_RUN_SET.get("iterations", 2))
    max_steps = int(BENCH_RUN_SET.get("max_steps", 100))

    # recheck parameters
    print(f"\n /** Trust Game V4 Parametes checking **/")
    print_dict(BENCH_RUN_SET)
    print_dict(MODEL_PARAMS)
    for key, value in params.to_dict().items():
        print(f"{key}: {value}")

    start = time.time()
    # network_constructing()
    results = mesa.batch_run(
        NTrustGameV4Model,
        parameters=params.to_dict(),
        iterations=iterations,
        max_steps=max_steps,
        number_processes=BENCH_RUN_SET.get("number_processes", 1),
        data_collection_period=1,
        display_progress=True,
    )

    end = time.time()
    print(f"程序运行时间为: {end - start:.2f} Seconds")

    # 获取当前时间的时间戳字符串
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_type = MODEL_PARAMS.get("model_type")
    analyzed_params = MODEL_PARAMS.get("analyzed_params", [])
    model_reporter = MODEL_PARAMS.get("collected_datas_static", [])
    file_name = f"{timestamp}_{model_type}_{analyzed_params[0]}_{analyzed_params[1]}.csv"
    # 1.save original results
    save_results(results, file_name)

    # 2.process results to get relation between the average of Cooperating Agents and Step
    data_processor = DataProcessing()
    datas = data_processor.benchresult_to_step_data(
        results=results,
        input_file_name="",
        output_file_name=file_name,
        param_name=analyzed_params,
        model_reporter=model_reporter,
    )

    # 3、advanced process to get the relation between params and Fc
    data_processor.processing_step_two(
        step_datas=datas,
        input_file_name="",
        output_file_name=file_name,
        param_name=analyzed_params,
        model_reporter=model_reporter,
    )

    # 4
    # agent_reporter = ["Reputation", "Fitness"]
    # datas = DataProcessing.benchresult_to_agent_step_data(
    #     results=results,
    #     input_file_name=None,
    #     output_file_name=file_name,
    #     param_name=analyzed_params,
    #     agent_reporter=agent_reporter,
    # )

def network_constructing():
    from mesa.discrete_space.grid import OrthogonalVonNeumannGrid
    from mesa.discrete_space.network import Network
    from evgamesimulations.networks.networkx_ import read_net_from_file
    from evgamesimulations.networks.hypernetx_ import read_hypergraph_from_file
    from evgamesimulations.common.hyper_network import HyperNetwork
    from evgamesimulations.utils.common_utils import CommonUtils

    network_file = MODEL_PARAMS.get("network_file")
    type_of_net_base = MODEL_PARAMS.get("type_of_net_base")
    type_of_network = MODEL_PARAMS.get("type_of_network")
    parameters = Parameters()
    cell_space = None

    if MODEL_PARAMS.get("is_internal_space", True):
        assert (
                type_of_network == CLASSICALNET_TYPE.NET_REGULAR_GRID.value
        ), "Only regular grid is supported for this method."

        cell_space = OrthogonalVonNeumannGrid((parameters.width, parameters.height), torus=True)
        # self.space = HexGrid((self.width, self.height), torus=True)
    else:
        if type_of_net_base != NET_BASE_TYPE.HYPERNET.value:
            raise ValueError("Only hypernet is supported for this method.")

        # read from file or directely generate network
        hypergraph = read_hypergraph_from_file(network_file)
        if isinstance(hypergraph, hnx.Hypergraph):
            raise ValueError("Failed to load or generate hypergraph. 'hypergraph' is None.")

        cell_space = HyperNetwork(hypergraph)

    return cell_space


if __name__ == "__main__":
    main()
