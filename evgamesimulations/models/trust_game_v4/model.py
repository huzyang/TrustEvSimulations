"""Simulation core, responsible for scheduling agents of the n-trust game."""

import random as Random
import warnings
import math
import numpy as np
import networkx as nx
import logging
import mesa
from mesa.discrete_space.grid import OrthogonalVonNeumannGrid
from mesa.discrete_space.network import Network

from evgamesimulations.networks.networkx_ import read_net_from_file
from evgamesimulations.networks.hypernetx_ import read_hypergraph_from_file
from evgamesimulations.common.hyper_network import HyperNetwork
from evgamesimulations.utils.common_utils import CommonUtils
from agents import (
    Agent,
    STRATEGY,
)
from parameters import (
    Parameters,
    MODEL_PARAMS,
    NET_BASE_TYPE,
    CLASSICALNET_TYPE,
    HYPERNET_TYPE,
    UPDATE_RULE_TYPE,
)

__all__ = ["NTrustGameV4Model", "HyperEdge"]


class HyperEdge(object):
    """定义超边类"""
    def __init__(self, id: int, nodes: list[int]):
        self.id = id
        self.nodes = nodes



class NTrustGameV4Model(mesa.Model):

    def __init__(
            self,
            # static params
            num_agents: int = 100,
            width: int = 10,
            height: int = 10,
            initial_proportion_I: float = 0.3,
            initial_proportion_T: float = 0.3,
            r_t: float = 3.0,
            r: float = 0.3,
            activation_order: str = "Simultaneous",
            seed: int = None,
            # special params
            beta: float = 0.5,
            delta: int = 25,
            ## Q-learning相关参数
            learning_rate: float = 0.8,
            discount_factor: float = 0.8,
            temperature: float = 2.0,
    ):

        super().__init__(seed=seed)
        # 配置 logger
        self.logger = logging.getLogger(__name__)

        self.num_agents = num_agents
        self.width = width
        self.height = height

        self.initial_proportion_I = initial_proportion_I
        self.initial_proportion_T = initial_proportion_T
        self.initial_proportion_U = round(1 - initial_proportion_I - initial_proportion_T, 2)

        self.k_I = 0
        self.k_T = 0
        self.k_U = 0
        self.__update_ks_from_proportion()

        self.r_t = r_t
        self.r = r
        # Multiplier for what is received by k_U from k_I (R_U * tv)
        self.r_u = round((1 + self.r) * self.r_t, 2)
        self.activation_order = activation_order
        self.nodes_payoff = np.zeros(self.num_agents)
        self.nodes_strategy = (
                [STRATEGY.INVESTOR.value] * self.k_I
                + [STRATEGY.TRUSTWORTHY_TRUSTEE.value] * self.k_T
                + [STRATEGY.UNTRUSTWORTHY_TRUSTEE.value] * self.k_U
        )

        # 模型特殊参数
        self.beta = beta
        self.delta = delta
        # Q-learning相关参数
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.temperature = temperature
        # 声誉初始化和统计
        self.nodes_reputation = self.generate_normal_distribution(
            MODEL_PARAMS.get("reputation_mu", 50),
            MODEL_PARAMS.get("reputation_variance", 10),
            self.num_agents,
        )
        self._fg = self._fn = self._fb = 0

        Random.shuffle(self.nodes_strategy)
        # Create agents and assign strategy
        self.agents.clear()

        network_file = MODEL_PARAMS.get("network_file")
        type_of_net_base = MODEL_PARAMS.get("type_of_net_base")

        if type_of_net_base != NET_BASE_TYPE.HYPERNET.value:
            raise ValueError("Only hypernet is supported for this method.")

        # read from file or directely generate network
        self.network = read_hypergraph_from_file(network_file)
        if self.network is None:
            raise ValueError("Failed to load or generate hypergraph. 'self.network' is None.")

        self.space = HyperNetwork(self.network)
        self.__check_space_and_hyper_network()  # check if the space and network are ready

        # 确保 node_neighbors_dict 和 nodes_neighbors_group_dict 存在
        if not hasattr(self.space, "node_neighbors_dict") or self.space.node_neighbors_dict is None:
            self.space.node_neighbors_dict = {}
        if (
                not hasattr(self.space, "nodes_neighbors_group_dict")
                or self.space.nodes_neighbors_group_dict is None
        ):
            self.space.nodes_neighbors_group_dict = {}

        # 超边属性
        self.hyper_edges = {}
        for key, value in self.space.hypergraph.incidence_dict.items():
            hyper_edge = HyperEdge(key, value)
            self.hyper_edges[key] = hyper_edge
        # 节点属性
        for i in range(self.num_agents):
            strategy = self.nodes_strategy[i]
            reputation = self.nodes_reputation[i]  # reputations[i]

            agent = Agent(i % self.num_agents, str(strategy), self, float(reputation))
            agent.neighbors_list = self.space.node_neighbors_dict.get(i, [])
            agent.neighbors_group_dict = self.space.nodes_neighbors_group_dict.get(i, {})
            agent.cell = self.space._cells[i]

        # DataCollector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "FI": lambda m: round(np.sum(np.array(m.nodes_strategy) == STRATEGY.INVESTOR.value) / self.num_agents, 4),
                "FT": lambda m: round(np.sum(np.array(m.nodes_strategy) == STRATEGY.TRUSTWORTHY_TRUSTEE.value) / self.num_agents, 4),
                # "FIT": lambda m: round(1-(np.sum(np.array(m.nodes_strategy) == STRATEGY.UNTRUSTWORTHY_TRUSTEE.value) / self.num_agents), 4),
                "FU": lambda m: round(np.sum(np.array(m.nodes_strategy) == STRATEGY.UNTRUSTWORTHY_TRUSTEE.value) / self.num_agents, 4),
                "GW": lambda m: round(np.sum(self.nodes_payoff), 2),
                # dynamic datas
                # "FG": lambda m: round(sum(1 for a in m.agents if a.get_current_state() == "G") / self.num_agents, 4),
                # "FN": lambda m: round(sum(1 for a in m.agents if a.get_current_state() == "N") / self.num_agents, 4),
                # "FB": lambda m: round(sum(1 for a in m.agents if a.get_current_state() == "B") / self.num_agents, 4),
                # "RE": lambda m: round(np.sum(self.nodes_reputation) / self.num_agents, 2),
            },
            agent_reporters={"Reputation": "reputation"},
        )
        self.running = True
        self.datacollector.collect(self)

    def __check_space_and_hyper_network(self):
        assert (
                len(self.space.all_cells) == self.num_agents
        ), "Number of cells in the space does not match the number of agents!"
        # for i, cell in self.space._cells.items():
        #     for connection in cell.connections.values():
        #         assert connection.coordinate in self.space.hypergraph.neighbors(i), "Connection not in hypergraph!"

    def update_state_counts(self):
        """一次性计算所有状态的计数并存储到模型属性中"""
        self._fg = self._fn = self._fb = 0
        for a in self.agents:
            state = a.get_current_state
            if state == "G":
                self._fg += 1
            elif state == "N":
                self._fn += 1
            elif state == "B":
                self._fb += 1
    def start(self):
        """Initialize the model."""
        pass

    def run(self, n):
        """Run the model for n steps."""
        model_type = MODEL_PARAMS.get("model_type")
        logging.debug(f"N {model_type} Model: \n{self.to_dict()}")
        for _ in range(n):
            self.step()

    def step(self):
        """Advance the model by one step.
        This function psuedo-randomly reorders the list of agent objects and
        then iterates through calling the function passed in as the parameter
        """
        self.logger.debug(f"************* In step {self.steps} ************")
        # Activate all agents, based on the activation regime
        match self.activation_order:
            case "Sequential":
                self.agents.do("step")
            case "Random":
                self.agents.shuffle_do("step")
            case "Simultaneous":
                self.agents.do("step")
                self.calculate_agent_edge_payoff()
                self.agents.do("advance")
            case _:
                raise ValueError(f"Unknown activation order: {self.activation_order}")

        # self.__agents_states()
        # Collect data
        self.update_state_counts()
        self.datacollector.collect(self)

    def calculate_agent_edge_payoff(self):
        """
        遍历所有超边，计算超边内代理的博弈收益并记录到数组
        """
        c = MODEL_PARAMS.get("pay_cost_c", 1)
        self.edges_nodes_payoff_array = np.zeros((len(self.space.hypergraph.incidence_dict), self.num_agents))

        for edge in self.hyper_edges.values():
            strategy_counts = {
                STRATEGY.INVESTOR.value: 0,
                STRATEGY.TRUSTWORTHY_TRUSTEE.value: 0,
                STRATEGY.UNTRUSTWORTHY_TRUSTEE.value: 0,
            }
            # 1、计算超边内博弈收益
            for node_id in edge.nodes:
                strategy_counts[self.nodes_strategy[node_id]] += 1

            num_t_u = (
                    strategy_counts[STRATEGY.TRUSTWORTHY_TRUSTEE.value]
                    + strategy_counts[STRATEGY.UNTRUSTWORTHY_TRUSTEE.value]
            )
            # Calculate payoff based on the utility matrix and strategy
            for node_id in edge.nodes:
                game_payoff = 0.0
                # beta1 = self.caculate_beta1_base_node_reputation(self.nodes_reputation[node_id])
                if num_t_u > 0:
                    match self.nodes_strategy[node_id]:
                        case STRATEGY.INVESTOR.value:
                            # TRUSTER payoff: tv * (R_T * (kT / (kT + kU)) - 1)
                            game_payoff = c * (self.r_t * (strategy_counts[STRATEGY.TRUSTWORTHY_TRUSTEE.value] / num_t_u) - 1)
                        case STRATEGY.TRUSTWORTHY_TRUSTEE.value:
                            # TRUSTWORTHY_TRUSTEE payoff: R_T * tv * (kI / (kT + kU))
                            game_payoff = c * self.r_t * (strategy_counts[STRATEGY.INVESTOR.value] / num_t_u)
                        case STRATEGY.UNTRUSTWORTHY_TRUSTEE.value:
                            game_payoff = c * self.r_u * (strategy_counts[STRATEGY.INVESTOR.value] / num_t_u)

                        case _:
                            game_payoff = 0.0
                # else:
                #     game_payoff = c
                self.edges_nodes_payoff_array[int(edge.id), node_id] = game_payoff

    def __update_ks_from_proportion(self):
        """Calculate k_I, k_T, and k_U based on the given proportion and total agents.

        Uses rounding to determine integer counts of investors, trustworthy trustees, and
        untrustworthy trustees, ensuring that the sum matches the total number of agents.
        """
        # Calculate the number of investors (k_I) and trustworthy trustees (k_T)
        self.k_I = round(self.num_agents * self.initial_proportion_I)
        self.k_T = round(self.num_agents * self.initial_proportion_T)

        # Calculate the number of untrustworthy trustees (k_U)
        self.k_U = self.num_agents - (self.k_I + self.k_T)

        # Validate the distribution of agents
        if (self.k_I + self.k_T + self.k_U) != self.num_agents:
            raise ValueError(
                "Error with the k_I, k_T, k_U distribution. Check parameters "
                "for k_I, k_T, and k_U to ensure they sum to the total number of agents."
            )

    def generate_normal_distribution(self, mu, sigma, n):
        """
        生成一个服从正态分布 (mu, sigma^2) 的数组。

        Args:
            mu (float): 正态分布的均值。
            sigma (float): 正态分布的标准差。
            n (int): 数组的长度。

        Returns:
            np.ndarray: 一个包含 n 个元素的数组，元素服从正态分布 (mu, sigma^2)。
        """
        arr = np.random.normal(mu, sigma, n)
        # 保留2位小数
        arr = np.around(arr, 2)
        logging.debug(f"Reputation array:\n {arr}")
        return arr

    def generate_uniform_distribution(self, low, high, n):
        """
        生成一个在 [low, high) 范围内均匀分布的数组。
        Args:
            low (float): 均匀分布的下界。
            high (float): 均匀分布的上界。
            n (int): 数组的长度。
        Returns:
            np.ndarray: 一个包含 n 个元素的数组，元素在 [low, high) 范围内均匀分布。
        """
        arr = np.random.uniform(low=low, high=high, size=n)
        np.random.shuffle(arr)
        arr = np.around(arr, 2)  # 保留2位小数
        logging.debug(f"Reputation array (Uniform):\n {arr}")
        return arr

    def __collect_state_fra(self, state):
        count = len([a for a in self.agents if a.get_current_state == state])
        fraction = round(count / self.num_agents, 4)
        return fraction

    def __collect_strategy_fra(self, strategy:str) -> float:
        count = np.sum(np.array(self.nodes_strategy) == strategy)
        fraction = round(count / self.num_agents, 4)
        return fraction

    def __edge_states(self):
        pass

    def __agents_states(self):
        """Print the strategies of all agents in a grid format."""
        edge_payoff = []
        edge_reputation = []
        for edge in self.hyper_edges.values():
            edge_payoff.append(edge.get_edge_payoff())
            edge_reputation.append(edge.get_edge_reputation())

        # 将每个属性的列表转换为二维矩阵
        def format_grid(grid):
            formatted_grid = []
            for i in range(self.height):
                row = grid[i * self.width: (i + 1) * self.width]
                formatted_grid.append(" ".join(map(str, row)))
            return "\n".join(formatted_grid)

        formatted_edge_payoff = format_grid(edge_payoff)
        formatted_edge_reputation = format_grid(edge_reputation)
        formatted_strategy_grid = format_grid(self.nodes_strategy)
        formatted_payoff_grid = format_grid(self.nodes_payoff)
        formatted_reputation_grid = format_grid(self.nodes_reputation)

        # 使用 logging.debug 记录每个属性的二维矩阵
        self.logger.debug(f"edge_payoff:\n{formatted_edge_payoff}")
        self.logger.debug(f"edge_reputation:\n{formatted_edge_reputation}")
        self.logger.debug(f"Strategies:\n{formatted_strategy_grid}")
        self.logger.debug(f"Payoffs:\n{formatted_payoff_grid}")
        self.logger.debug(f"node reputation:\n{formatted_reputation_grid}")


    def to_dict(self):
        return self.__dict__
