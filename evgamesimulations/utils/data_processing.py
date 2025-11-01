"""data processing"""

import math
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from evgamesimulations.utils.common_utils import CommonUtils

__all__ = ["DataProcessing"]


class DataProcessing:
    @classmethod
    def benchresult_to_agent_step_data(
        cls,
        results: list[dict],
        input_file_name: str,
        output_file_name: str,
        param_name: list[str],
        agent_reporter: list[str],
    ) -> pd.DataFrame:
        """
        处理batch run 的结果，得到各参数下的Agents的属性（payoff、reputation等）的平均值随Step的变化趋势。

        :param results: 批量运行的结果列表，每个元素是一个字典。
        :param input_file_name: 输入CSV文件名，若results为None，则从该文件读取数据。
        :param output_file_name: 输出文件名。
        :param param_name: 参数名称列表。
        :param agent_reporter: 模型报告指标名称列表。
        :return: 包含按Step分组的平均值的DataFrame。
        """

        # 参数合法性检查
        if not results and not input_file_name:
            raise ValueError("参数 results 和 input_file_name 不能同时为 None!")

        if not param_name or len(param_name) != 1:
            raise ValueError("param_name 必须是非空列表且长度为1!")

        if not agent_reporter or not isinstance(agent_reporter, list):
            raise ValueError("agent_reporter 必须是非空列表!")

        # 构造列名
        columns = ["RunId", "iteration", "Step", "AgentID"] + param_name + agent_reporter

        try:
            # 加载数据
            if results is not None:
                all_datas = pd.DataFrame(results)
            else:
                all_datas = pd.read_csv(CommonUtils.get_project_root_path() + input_file_name)
                output_file_name = f"{input_file_name[:-8]}.csv"

            # 确保数据包含所需列
            missing_columns = set(columns) - set(all_datas.columns)
            if missing_columns:
                raise KeyError(f"数据中缺少以下列: {missing_columns}")

            all_datas = all_datas[columns]

            # 计算 Step 和 AgentID 的平均值
            all_datas = (
                all_datas.groupby(param_name + ["Step", "AgentID"])[agent_reporter].mean().round(4).reset_index()
            )

            # 转换为宽格式
            data_results = all_datas.pivot(index=["Step", "AgentID"], columns=param_name, values=agent_reporter)
            # data_results = all_datas.pivot(index=["Step", "AgentID"], columns=param_name, values=agent_reporter)

            # 取最后 20% 的 Step 数据
            unique_steps = data_results.index.get_level_values("Step").unique()
            last_20_percent_steps = unique_steps[int(len(unique_steps) * 0.8) :]
            last_20_percent_df = data_results.loc[
                data_results.index.get_level_values("Step").isin(last_20_percent_steps)
            ]

            # 按照 AgentID 进行分组取平均
            final_result = last_20_percent_df.groupby(["AgentID"]).mean().round(4).reset_index()

            # 重命名列
            final_result.columns = [f"{col}-avgReputation" for col in final_result.columns]

            # 重置索引
            # final_result = final_result.reset_index()

            # 输出结果
            print(final_result.tail())
            output_file_name = f"{output_file_name[:-4]}_agentdata.csv"
            final_result.to_csv(CommonUtils.get_project_root_path() + output_file_name, mode="w", index=False)
            print(f"Results saved into {output_file_name}")
            return final_result

        except Exception as e:
            print(f"benchresult_to_agent_step_data处理数据时发生错误: {e}")
            return pd.DataFrame()  # 返回空DataFrame以满足类型要求

    @classmethod
    def benchresult_to_step_data(
        cls,
        results: list[dict],
        input_file_name: str,
        output_file_name: str,
        param_name: list[str],
        model_reporter: list[str],
    ) -> pd.DataFrame:
        """
        处理batch run 的结果，得到各参数下的平均Cooperating Agents随Step的变化趋势。

        :param results: 批量运行的结果列表，每个元素是一个字典。
        :param input_file_name: 输入CSV文件名，若results为None，则从该文件读取数据。
        :param output_file_name: 输出文件名。
        :param param_name: 参数名称列表。
        :param model_reporter: 模型报告指标名称列表。
        :return: 包含按Step分组的平均值的DataFrame。
        """
        # 参数合法性检查
        if not results and not input_file_name:
            raise ValueError("参数 results 和 input_file_name 不能同时为 None!")

        if not param_name or not isinstance(param_name, list):
            raise ValueError("param_name 必须是非空列表!")

        if not model_reporter or not isinstance(model_reporter, list):
            raise ValueError("model_repoter 必须是非空列表!")


        # 构造列名
        columns = ["RunId", "iteration", "Step"] + param_name + model_reporter

        try:
            # 加载数据
            if results is not None:
                all_datas = pd.DataFrame(results)
            else:
                all_datas = pd.read_csv(os.path.join(CommonUtils.get_project_root_path(), "outputs", input_file_name))
                output_file_name = f"{input_file_name[:-8]}.csv"

            all_datas = all_datas[columns]

            # 分组计算
            grouped = all_datas.groupby(param_name)
            data_results = pd.DataFrame()

            for name, group in grouped:
                for col in model_reporter:
                    # 对同一 Step 下的 Cooperating Agents 求平均
                    data = group.groupby("Step")[col].mean().round(4).reset_index()
                    if data_results.empty:
                        data_results = pd.DataFrame(
                            data=np.arange(0, len(data)),
                            columns=[f"Step"],
                            # data=np.arange(0, len(data)), columns=[f"Step({'-'.join(param_values)})"]
                        )

                    # concat data to data_results
                    formatted_name = tuple(float(val) for val in name)
                    column_name = f"{col}({formatted_name[0]}, {formatted_name[1]})"
                    data_results = pd.concat(
                        [data_results, data[col].rename(column_name)],
                        axis=1,
                    )

            # 输出结果
            print(data_results.tail())
            output_file_name = f"{output_file_name[:-4]}_timestep.csv"
            output_path = os.path.join(CommonUtils.get_project_root_path(), "outputs", output_file_name)
            data_results.to_csv(output_path, mode="w", index=False)
            print(f"Results saved into {output_file_name}")
            return data_results

        except Exception as e:
            print(f"benchresult_to_step_data处理数据时发生错误: {e}")
            return pd.DataFrame()

    @classmethod
    def processing_step_two(
        cls,
        step_datas: pd.DataFrame,
        input_file_name: str,
        output_file_name: str,
        param_name: list[str],
        model_reporter: list[str],
    ) -> pd.DataFrame:
        import re

        # 参数合法性检查
        if step_datas is None and not input_file_name:
            raise ValueError("参数 step_datas 和 input_file_name 不能同时为 None!")

        if not param_name or not isinstance(param_name, list) or len(param_name) != 2:
            raise ValueError("param_name 必须是非空列表且长度为2!")

        if not model_reporter or not isinstance(model_reporter, list):
            raise ValueError("model_reporter 必须是非空列表!")

        # 提取参数值
        param1 = param_name[0]
        param2 = param_name[1]
        try:
            # 加载数据
            if step_datas is None:
                step_datas = pd.read_csv(os.path.join(CommonUtils.get_project_root_path(), "outputs", input_file_name))
                output_file_name = f"{input_file_name[:-13]}.csv"

            # 步骤1：取后20%的数据
            last_20_percent_data = step_datas.iloc[int(len(step_datas) * 0.8) :]

            # 步骤2：排除首列
            last_20_percent_data = last_20_percent_data.drop(columns="Step")

            # 步骤3：计算列平均值
            avg_df = last_20_percent_data.mean().round(4)

            # 步骤4：解析列名
            pattern = r"^([A-Za-z]+)\((\d+\.?\d*),\s*(\d+\.?\d*)\)$"
            # pattern = r"^([A-Za-z]+)(\d+\.?\d*)-(\d+\.?\d*)$"  # 老格式
            parsed = []
            param_values = []
            for col in avg_df.index:
                match = re.match(pattern, col)
                if match:
                    var, v1, v2 = match.groups()
                    param_values.append((v1, v2))
                    parsed.append((var, v1, v2, avg_df[col]))

            # 提取唯一的 r 和 beta 值
            param1_values = sorted(set([val[0] for val in param_values]))
            param2_values = sorted(set([val[1] for val in param_values]))
            # 步骤5：重塑数据
            result_df = pd.DataFrame(parsed, columns=["Variable", param1, param2, "Value"])

            # 步骤6：重新排列列顺序（先按model_reporter名，再按param值）
            result_df = result_df.pivot(index=param2, columns=["Variable", param1], values="Value")
            result_df = result_df.reindex(
                columns=pd.MultiIndex.from_product([model_reporter, param1_values], names=["Variable", param1])
            )

            # 步骤7：整理列名并调整顺序
            result_df.columns = ["{}{}".format(var, r) for var, r in result_df.columns]
            result_df = result_df.reset_index().rename(columns={"index": param2})

            # 步骤8：调整列顺序（确保每个变量的两个r值相邻）
            columns_order = [param2] + [f"{var}{val}" for var in model_reporter for val in param1_values]
            result_df = result_df[columns_order]

            # 输出结果
            print(result_df.tail())
            output_file_name = f"{output_file_name[:-4]}_summary.csv"
            output_path = os.path.join(CommonUtils.get_project_root_path(), "outputs", output_file_name)
            result_df.to_csv(output_path, mode="w", index=False)
            print(f"Results were saved into {output_file_name}!")
            return result_df

        except Exception as e:
            print(f"processing_step_two处理过程中发生错误: {e}")
            return pd.DataFrame()
