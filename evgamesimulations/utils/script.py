import pandas as pd
import numpy as np
import os

from evgamesimulations.utils.common_utils import CommonUtils


def split_timestep_data(file_name: str):
    # 读取CSV文件
    df = pd.read_csv(os.path.join(CommonUtils.get_project_root_path(), "outputs", file_name))

    # 创建第一个文件：包含Step和"FI", "FT", "FU", "GW"所有列
    ki_kt_columns = [
        col for col in df.columns if col.startswith("Step") or col.startswith("FI") or col.startswith("FT") or col.startswith("FU") or col.startswith("GW")
    ]
    df1 = df[ki_kt_columns]

    # 创建第二个文件：包含Step和"FG", "FN", "FB", "RE"的所有列
    kit_gp_columns = [
        col for col in df.columns if col.startswith("Step") or col.startswith("FG") or col.startswith("FN") or col.startswith("FB") or col.startswith("RE")
    ]
    df2 = df[kit_gp_columns]

    # 保存为两个新文件
    output_file_1 = os.path.join(CommonUtils.get_project_root_path(), "outputs", f"{file_name[:-4]}_file1.csv")
    output_file_2 = os.path.join(CommonUtils.get_project_root_path(), "outputs", f"{file_name[:-4]}_file2.csv")
    df1.to_csv(output_file_1, mode="w", index=False)
    df2.to_csv(output_file_2, mode="w", index=False)

    print("文件分割完成！")
    print(f"第一个文件包含 {len(df1.columns)} 列: Step, FI, FT, FU, GW 所有列")
    print(f"第二个文件包含 {len(df2.columns)} 列: Step, FG, FN, FB, RE 所有列")

def get_step_agent_data(file_name: str):
    # 读取CSV文件
    df = pd.read_csv(os.path.join(CommonUtils.get_project_root_path(), "outputs", file_name))
    # 筛选RunId=0的数据
    df_run0 = df[df['RunId'] == 0]

    # 处理Strategy数据
    strategy_pivot = df_run0.pivot(index='AgentID', columns='Step', values='Strategy')
    strategy_pivot.index.name = None
    strategy_pivot.columns.name = None
    output_file_1 = os.path.join(CommonUtils.get_project_root_path(), "outputs", f"{file_name[:-4]}_strategy_data.csv")
    strategy_pivot.to_csv(output_file_1, mode="w", index=False, header=True)
    print("1、完成Strategy数据处理！")

    # 处理State数据
    state_pivot = df_run0.pivot(index='AgentID', columns='Step', values='State')
    state_pivot.index.name = None
    state_pivot.columns.name = None
    state_pivot.to_csv('state_data.csv', header=True)
    output_file_2 = os.path.join(CommonUtils.get_project_root_path(), "outputs", f"{file_name[:-4]}_state_data.csv")
    state_pivot.to_csv(output_file_2, mode="w", index=False, header=True)
    print("2、完成state_data数据处理！")

    # # 处理State_strategy数据
    state_strategy_pivot = df_run0.pivot(index='AgentID', columns='Step', values='State_strategy')
    state_strategy_pivot.index.name = None
    state_strategy_pivot.columns.name = None
    output_file_3 = os.path.join(CommonUtils.get_project_root_path(), "outputs", f"{file_name[:-4]}_state_strategy_data.csv")
    state_strategy_pivot.to_csv(output_file_3, mode="w", index=False, header=True)
    print("3. 完成state_strategy_data数据处理！")


def average_same_columns(file1_name: str, file2_name: str):
    """
    将两个CSV文件中相同列名的数据求和取平均

    Parameters:
    file1_name (str): 第一个CSV文件名
    file2_name (str): 第二个CSV文件名
    output_file_name (str): 输出文件名，如果为None则自动生成
    """
    # 使用示例
    # average_same_columns(
    #     "b-0.1-0.5-0.9_Trust_GameV4_r_beta_timestep-1.csv",
    #     "b-0.1-0.5-0.9_Trust_GameV4_r_beta_timestep-2.csv"
    # )
    # 读取两个CSV文件
    file1_path = os.path.join(CommonUtils.get_project_root_path(), "outputs", file1_name)
    file2_path = os.path.join(CommonUtils.get_project_root_path(), "outputs", file2_name)

    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # 创建结果DataFrame，保持列顺序与原文件一致
    result_df = pd.DataFrame()

    # 按第一个文件的列顺序处理数据
    for column in df1.columns:
        if column in df2.columns:
            # 对共同列求平均并保留4位小数
            result_df[column] = ((df1[column] + df2[column]) / 2).round(4)
        else:
            # 如果第二文件没有该列，则只保留第一文件的数据（这种情况应该不会出现）
            result_df[column] = df1[column].round(4)


    # 从输入文件名生成输出文件名
    base_name1 = file1_name.replace('_timestep-1.csv', '')
    output_file_name = f"{base_name1}_averaged.csv"

    # 保存结果
    output_path = os.path.join(CommonUtils.get_project_root_path(), "outputs", output_file_name)
    result_df.to_csv(output_path, index=False)

    print(f"文件处理完成！结果已保存到: {output_file_name}")

    return result_df


def main():
    file_name = "20251002091636_Trust_GameV4_r_beta_all.csv"
    # split_timestep_data(file_name)
    # get_step_agent_data(file_name)
    # 调用函数处理两个文件
    average_same_columns(
        "20251006172233_Trust_GameV4_initial_proportion_T_initial_proportion_I_summary.csv",
        "20251006172750_Trust_GameV4_initial_proportion_T_initial_proportion_I_summary.csv"
    )

if __name__ == "__main__":
    main()
