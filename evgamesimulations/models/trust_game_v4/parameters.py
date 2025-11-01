#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : huzyang
# @File    : parameters.py
# @Software: PyCharm
# @DATE    : 2025/09/18
"""Parameters of the model of evolutionary dynamics.
This is parameters of the model for the Game simulation.
"""
import math
import os

from evgamesimulations.utils.common_utils import CommonUtils
from enum import Enum

__all__ = ["NET_BASE_TYPE", "CLASSICALNET_TYPE", "HYPERNET_TYPE", "UPDATE_RULE_TYPE", "Parameters", "MODEL_PARAMS"]


# =====================================
# Static
# =====================================
class NET_BASE_TYPE(Enum):
    CLASSICALNET = "ClassicalNet"
    HYPERNET = "HyperNet"


class CLASSICALNET_TYPE(Enum):
    # TYPE OF NETWORK
    NET_WELL_MIXED = "WellMixed"
    NET_REGULAR_GRID = "RegularGrid"
    NET_ER_NETWORK = "ER"
    NET_SF_NETWORK = "SF"


class HYPERNET_TYPE(Enum):
    # TYPE OF HYPERNET
    HYPERNET_ER_NETWORK = "HyperER"
    HYPERNET_SF_NETWORK = "HyperSF"
    HYPERNET_N_UNIFORM = "HyperNUniform"


class UPDATE_RULE_TYPE(Enum):
    # Update rule type for agents
    PROPORTIONAL_UPDATE_RULE = "ProportionalRule"
    PROPORTIONAL_V1_RULE = "ProportionalV1Rule"
    UI_UPDATE_RULE = "UnconditionalImitationRule"
    VOTER_UPDATE_RULE = "VoterModelRule"
    FERMI_UPDATE_RULE = "FermiRule"
    MODIFIED_FERMI_V1_RULE = "FermiRuleV1"
    MODIFIED_FERMI_V2_RULE = "FermiRuleV2"
    MODIFIED_FERMI_V3_RULE = "FermiRuleV3"
    MORAN_UPDATE_RULE = "MoranRule"
    BEST_RESPONSE_UPDATE_RULE = "BestResponseRule"
    Q_LEARNING = "QLearning"


# Static params
BENCH_RUN_SET = {
    "iterations": 2,
    "max_steps": 20000,
    "number_processes": 8,
    "data_collection_period": 1,
    "display_progress": True,
}
MODEL_PARAMS = {
    "model_type": "Trust_GameV4",
    "pay_cost_c": 1,
    "max_degree": 4,
    "reputation_mu": 50,
    "reputation_variance": 10,
    "max_reputation": 100,
    "min_reputation": 1,
    "fermi_kappa": 0.5,
    "epsilon": 0.02,  # Q-learning相关参数
    "is_internal_space": False,  # True False
    "network_file": CommonUtils.get_project_root_path() + "/datas/hypergraph_5-uniform_1024.csv",
    "type_of_net_base": NET_BASE_TYPE.HYPERNET.value,
    "type_of_network": HYPERNET_TYPE.HYPERNET_N_UNIFORM.value,
    # "network_file": CommonUtils.get_project_root_path() + "/datas/ba_scale_free_4_node-1024.csv",
    # "type_of_net_base": NET_BASE_TYPE.CLASSICALNET.value,
    # "type_of_network": CLASSICALNET_TYPE.NET_REGULAR_GRID.value,
    "update_rule": UPDATE_RULE_TYPE.Q_LEARNING.value,
    "activation_order": "Simultaneous",  # ["Sequential", "Random", "Simultaneous"]
    "analyzed_params": ["beta", "r"],  # ["r", "alpha", "beta"]
    "collected_datas_static": ["FI", "FT", "FU", "GW"] #, "FG", "FN", "FB", "RE"],
    # "collected_datas_dynamic": ["FG", "FN", "FB", "RE"],
}


class Parameters(object):
    activation_regimes = ["Sequential", "Random", "Simultaneous"]

    def __init__(self):
        # ********************************
        # Variables
        # ********************************
        self.num_agents = 1024
        self.width = None
        self.height = None
        self.initial_proportion_I = 0.3
        self.initial_proportion_T = 0.3
        # self.initial_proportion_U = 0.4

        self.r_t = 6.0
        self.r = 0.4   # [0.1, 0.3, 0.5, 0.7, 0.9],[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # self.r = [(i + 1) / 10 for i in range(10)]

        self.beta = 0.9 # [(2*i) / 10 for i in range(5)] # [(2*i) / 10 for i in range(5)]  # 声誉关注度

        self.delta = 20  # 声誉阈值控制

        # Q-learning相关参数
        self.learning_rate = 0.8 # [0.2, 0.4, 0.6, 0.8]
        self.discount_factor = 0.8 # [0.2, 0.4, 0.6, 0.8]
        self.temperature = 2

    def params_check(self):
        """Check the parameters of the model."""

        if self.width is None or self.height is None:
            # 计算 width,height 为 num_agents 的平方根，并取整
            self.width = int(math.sqrt(self.num_agents))
            self.height = (self.num_agents + self.width - 1) // self.width

        if 0 >= self.re_threshold_gap or self.re_threshold_gap >= 50:
            raise ValueError("Invalid re_threshold_gap")

    def to_dict(self):
        return self.__dict__
