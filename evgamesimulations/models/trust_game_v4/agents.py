"""Class of the players of the N-trust game"""

import math
import random
import mesa
from mesa.discrete_space.cell_agent import CellAgent
import warnings
import numpy as np
from enum import Enum
import logging

from parameters import (
    MODEL_PARAMS,
    Parameters,
    NET_BASE_TYPE,
    CLASSICALNET_TYPE,
    HYPERNET_TYPE,
    UPDATE_RULE_TYPE,
)

__all__ = ["Agent", "STRATEGY"]

class STRATEGY(Enum):
    # UNDEFINED_STRATEGY = "Unknown"
    INVESTOR = "I"
    TRUSTWORTHY_TRUSTEE = "T"
    UNTRUSTWORTHY_TRUSTEE = "U"

STRATEGY_LIST = [STRATEGY.INVESTOR.value, STRATEGY.TRUSTWORTHY_TRUSTEE.value, STRATEGY.UNTRUSTWORTHY_TRUSTEE.value]
TRINARY_REPUTATION_LIST = ["G", "N", "B"]

class Agent(CellAgent):

    def __init__(self, agent_id: int, strategy: str, model: mesa.Model, reputation: float = 50):
        super().__init__(model)
        # reset unique_id, making it starting from 0
        self.unique_id = agent_id
        self.strategy = strategy
        self.next_strategy = strategy
        self.payoff = 0.0

        # neighbors_group_dict eg:{'0':..., '1':{0: ('0', '1', '2'), 2: ('4', '1', '5'), 8: ('1', '2', '14')},...}
        # where '1' denotes a node, 0 denotes an edge. 0: ('0', '1', '2') denotes nodes '0', '1' and '2' are in edge 0
        self.neighbors_list = []
        self.neighbors_group_dict = {}

        # 配置 logger
        self.logger = logging.getLogger(__name__)

        # addtional characteristics
        self.reputation = reputation

        # Q-learning相关参数
        # self.reward = 0.0
        self.q_table = {
            "G": [0.0, 0.0, 0.0],  # 在I状态下，选择[I,T,U]的Q值
            "N": [0.0, 0.0, 0.0],  # 在T状态下，选择[I,T,U]的Q值
            "B": [0.0, 0.0, 0.0]   # 在U状态下，选择[I,T,U]的Q值
        }
        # 无声誉机制和状态
        # self.q_table = {
        #     "I": [0.0, 0.0, 0.0],  # 在I状态下，选择[I,T,U]的Q值
        #     "T": [0.0, 0.0, 0.0],  # 在T状态下，选择[I,T,U]的Q值
        #     "U": [0.0, 0.0, 0.0]   # 在U状态下，选择[I,T,U]的Q值
        # }
        self.current_state = self.get_current_state()  # 当前状态
        self.last_state = None  # 上一状态

    @property
    def is_cooroperating(self):
        return self.strategy != STRATEGY.UNTRUSTWORTHY_TRUSTEE.value

    @property
    def degree(self):
        degree = 0
        if self.neighbors_group_dict is None:
            return degree

        degree = len(self.neighbors_group_dict)
        return degree

    @property
    def state_strategy(self):
        return f"{self.current_state}-{self.strategy}"

    def get_current_state(self) -> str:
        delta = int(self.model.delta)
        if self.reputation < 50.0 - delta:
            self.current_state = "B"
        elif self.reputation > 50.0 + delta:
            self.current_state = "G"
        else:
            self.current_state = "N"
        return self.current_state

    @property
    def reward(self) -> float:
        # theta = MODEL_PARAMS.get("theta")  # 这行被注释掉了，可能是用于获取某个参数theta
        r = int(0)
        # 根据当前状态，设置奖励值
        match self.last_state:
        # 使用match-case语句（Python 3.10+）根据当前状态设置不同的基础奖励值
            case "G":
                r = int(2)  # 如果状态为"G"，设置奖励值为2
            case "N":
                r = int(1)  # 如果状态为"N"，设置奖励值为1
            case "B":
                r = int(0)
            case _:
                return 0

        # 返回奖励值
        return self.payoff + r * self.model.beta

    def step(self):

        # neighbors = [*list(self.cell.neighborhood.agents), self]
        self.choose_next_strategy()
        self.logger.debug(
            f"Focal agent ID: {self.unique_id}, updated strategy from {self.strategy} to {self.next_strategy}"
        )
        # if MODEL_PARAMS.get("activation_order") != "Simultaneous":
        #     self.advance()

    def advance(self):
        self.calculate_payoff()  # 计算收益
        self.update_reputation() # 更新声誉

    def choose_next_strategy(self):
        """
        Returns:
            bool: True if a strategy update occurs, otherwise False.
        """
        # No strategy update occurs in the first step
        if self.model.steps == 1:
            return

        # Select the update rule based on the model parameters
        update_rule = MODEL_PARAMS.get("update_rule")
        match update_rule:
            case UPDATE_RULE_TYPE.Q_LEARNING.value:
                self.__q_learning_update()
            case _:
                raise ValueError(f"Unknown update rule: {update_rule}")
        self.strategy = self.next_strategy  # 更新策略
        self.model.nodes_strategy[self.unique_id] = self.strategy  # 更新全局策略
        return True

    def calculate_payoff(self):
        type_of_net_base = MODEL_PARAMS.get("type_of_net_base")
        # self.payoff = 0.0  # reset payoff
        if MODEL_PARAMS.get("activation_order") == "Simultaneous":
            # payoff for strategy
            if type_of_net_base == NET_BASE_TYPE.HYPERNET.value:
                self.payoff = self.calculate_payoff_hypernetx(
                    self.strategy,
                    based_on_next_strategy=True,
                )
            else:
                self.payoff = self.calculate_payoff_networkx(
                    self.strategy,
                    based_on_next_strategy=True,
                )
        else:
            # payoff for next strategy
            if type_of_net_base == NET_BASE_TYPE.HYPERNET.value:
                self.payoff = self.calculate_payoff_hypernetx(
                    self.next_strategy,
                    based_on_next_strategy=False,
                )
            else:
                self.payoff = self.calculate_payoff_networkx(
                    self.next_strategy,
                    based_on_next_strategy=False,
                )

        self.model.nodes_payoff[self.unique_id] = self.payoff  # 存储全局收益

    def calculate_payoff_hypernetx(self, self_strategy, based_on_next_strategy=False):
        """
        Calculate the payoff for the agent based on interactions with its neighbors.
        """
        node_edges_payoffs = []  # save agent's payoff in different edges
        for edge_id in self.neighbors_group_dict.keys():
            node_edges_payoffs.append(self.model.edges_nodes_payoff_array[int(edge_id), self.unique_id])

        payoff = np.mean(node_edges_payoffs).round(5)
        self.logger.debug(
            f"Focal agent ID: {self.unique_id}, Strategy: {self.strategy}, update payoff from {self.payoff} to {round(payoff, 5)}"
        )
        return payoff

    def calculate_payoff_networkx(self, self_strategy, based_on_next_strategy=False):
        """
        Calculate the payoff for the agent based on interactions with its neighbors.

        Args:
            model (Model): The model containing the network and parameters.

        Returns:
            float: The payoff for the current step.
        """

        neighbors = list(self.cell.neighborhood.agents)
        if not neighbors:
            self.logger.warning(f"No neighbors found for agent {self.unique_id}")
            return 0.0

        strategy_counts = {
            STRATEGY.INVESTOR.value: 0,
            STRATEGY.TRUSTWORTHY_TRUSTEE.value: 0,
            STRATEGY.UNTRUSTWORTHY_TRUSTEE.value: 0,
        }

        # Count the number of neighbors with each strategy
        if based_on_next_strategy:
            # next strategy
            for agent in neighbors:
                if agent.next_strategy in strategy_counts:
                    strategy_counts[agent.next_strategy] += 1
        else:
            # strategy
            for agent in neighbors:
                if agent.strategy in strategy_counts:
                    strategy_counts[agent.strategy] += 1

        strategy_counts[self_strategy] += 1  # Include focal agent itself

        payoff = 0.0
        num_t_u = (
            strategy_counts[STRATEGY.TRUSTWORTHY_TRUSTEE.value] + strategy_counts[STRATEGY.UNTRUSTWORTHY_TRUSTEE.value]
        )
        # Calculate payoff based on the utility matrix and strategy

        if num_t_u > 0:
            c = MODEL_PARAMS.get("pay_cost_c", 1)
            r_t = self.model.payoff_param_r_t
            r_u = self.model.payoff_param_r_u

            # Calculate payoff based on the agent's current strategy
            # formula_1 = self.calculate_formula_1(omega, strategy_counts[STRATEGY.INVESTOR.value])
            match self_strategy:
                case STRATEGY.INVESTOR.value:
                    # INVESTOR payoff: tv * (R_T * (kT / (kT + kU)) - 1)
                    payoff = c * (r_t * (strategy_counts[STRATEGY.TRUSTWORTHY_TRUSTEE.value] / num_t_u) - 1)

                case STRATEGY.TRUSTWORTHY_TRUSTEE.value:
                    # TRUSTWORTHY_TRUSTEE payoff: R_T * tv * (kI / (kT + kU))
                    payoff = r_t * c * (strategy_counts[STRATEGY.INVESTOR.value] / num_t_u)

                case STRATEGY.UNTRUSTWORTHY_TRUSTEE.value:
                    # UNTRUSTWORTHY_TRUSTEE payoff: R_U * tv * (kI / (kT + kU))
                    payoff = r_u * c * (strategy_counts[STRATEGY.INVESTOR.value] / num_t_u)
                case _:
                    warnings.warn(
                        "Fatal Error: The agent's strategy is unknown!",
                        category=UserWarning,
                        stacklevel=1,
                    )

        self.logger.debug(
            f"Focal agent ID: {self.unique_id}, Strategy: {self.strategy}, neighbors strategy:{[agent.strategy for agent in neighbors]}, update payoff from {self.payoff} to {round(payoff, 5)}"
        )
        return round(payoff, 5)

    def update_reputation(self):
        """
        更新代理的声誉值。

        Returns:
            float: 更新后的声誉值。
        """
        if self.strategy == STRATEGY.UNTRUSTWORTHY_TRUSTEE.value:
            delta = -2.0  # random.gauss(-0.1, 0.1)
        else:
            delta = 2.0  # random.gauss(0.1, 0.1)

        # 计算新的声誉值并限制范围
        new_reputation = max(1.0, min(100.0, round(self.reputation + delta, 2)))  # 确保值在[min,max]范围内

        self.logger.debug(
            f"Focal agent ID: {self.unique_id}, updated reputation from {self.reputation} to {new_reputation}"
        )
        self.reputation = new_reputation
        self.model.nodes_reputation[self.unique_id] = self.reputation # 更新全局声誉
        return new_reputation

    # 新增ε-greedy策略方法
    def _get_q_state(self) -> str:
        """获取当前Q-learning状态 - 根据声誉评价作为状态"""
        return self.get_current_state()
        # 无声誉机制和状态，直接使用当前策略作为状态
        # return self.strategy

    def _select_action_epsilon_greedy(self, state: str) -> str:
        """使用ε-greedy策略选择动作"""
        epsilon = MODEL_PARAMS.get("epsilon", 0.02)
        if random.random() < epsilon:
            # 探索：随机选择动作
            return random.choice(STRATEGY_LIST)

        # 利用当前状态选择当前最优动作
        q_values = self.q_table[state]
        max_q = max(q_values)
        # 多个最大值时随机选择
        max_actions = [i for i, q in enumerate(q_values) if q == max_q]
        action_idx = random.choice(max_actions)
        return STRATEGY_LIST[action_idx]

    def _select_action_boltzmann(self, state: str) -> str:
        """使用玻尔兹曼策略选择动作"""
        # 获取当前状态的Q值
        q_values = np.array(self.q_table[state])

        # 计算玻尔兹曼动作选择概率，根据Q值计算动作选择概率
        max_q = np.max(q_values)
        exp_q = np.exp((q_values - max_q) / self.model.temperature) # 防止数值溢出，减去最大值
        action_probs = exp_q / np.sum(exp_q) # 归一化

        # 根据概率分布选择动作
        action_idx = np.random.choice(len(STRATEGY_LIST), p=action_probs)
        return STRATEGY_LIST[action_idx]
    def __q_learning_update(self):
        """执行Q-learning策略更新"""
        # 获取当前状态
        self.current_state = self._get_q_state()

        # 如果是第一次运行，只初始化状态
        if self.last_state is None:
            self.last_state = self.current_state
            return

        # 获取当前奖励 reward = payoff + alpha * reputation / max_reputation
        reward = self.reward

        # 获取上次动作的索引
        last_action_idx = STRATEGY_LIST.index(self.strategy)

        # Q-learning更新公式: Q(s,a) ← Q(s,a) + α[r + γ·maxQ(s',a') − Q(s,a)]
        last_q = self.q_table[self.last_state][last_action_idx]
        max_future_q = max(self.q_table[self.current_state])

        # 计算新的Q值
        new_q = last_q + self.model.learning_rate * (
                reward + self.model.discount_factor * max_future_q - last_q
        )

        # 更新Q表
        self.q_table[self.last_state][last_action_idx] = round(new_q, 5)

        # 选择下一个动作
        self.next_strategy = self._select_action_epsilon_greedy(self.current_state)  # epsilon_贪婪
        # self.next_strategy = self._select_action_boltzmann(self.current_state) # 玻尔兹曼策略

        # 记录当前状态
        self.last_state = self.current_state

        self.logger.debug(
            f"Focal agent ID: {self.unique_id}, Q-Learning Boltzmann update: "
            f"Strategy from {self.strategy} to {self.next_strategy}, "
            f"State from {self.last_state} to {self.current_state}, "
            f"Reward: {reward:.3f}, New Q-Value: {new_q:.3f}"
        )