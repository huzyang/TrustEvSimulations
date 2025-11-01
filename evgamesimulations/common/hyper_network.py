#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : huzyang
# @Project : EvGameSimulations
# @File    : HyperNetwork.py
# @Software: VSCode
# @DATE    : 2024/11/19
"""A Hyper Network grid. 
Exstends from DiscreteSpace which in mesa.experimental.cell_space.discrete_space.
Similar to mesa.experimental.cell_space.network.py, but based on hypergraph."""

from random import Random
from typing import Any
import hypernetx as hnx
from hypernetx import Hypergraph
# from mesa.experimental.cell_space.cell import Cell
from mesa.discrete_space.cell import Cell
# from mesa.experimental.cell_space.discrete_space import DiscreteSpace
from mesa.discrete_space.discrete_space import DiscreteSpace

__all__ = ["HyperNetwork"]


class HyperNetwork(DiscreteSpace[Cell]):
    """A discrete space using hypergraph networked."""

    def __init__(
        self,
        hypergraph: Hypergraph,
        capacity: int | None = None,
        random: Random | None = None,
        cell_klass: type[Cell] = Cell,
    ) -> None:
        """A Hyper Networked grid.

        Args:
            hypergraph: a Hypergraph instance.
            capacity (int) : the capacity of the cell
            random (Random): a random number generator
            cell_klass (type[Cell]): The base Cell class to use in the Network

        """
        super().__init__(capacity=capacity, random=random, cell_klass=cell_klass)
        self.hypergraph = hypergraph

        for node_id in self.hypergraph.nodes:
            self._cells[node_id] = self.cell_klass(node_id, capacity, random=self.random)
        self._connect_cells()
        self.nodes_neighbors_group_dict = {}
        self._nodes_neighbors_group()

    def _connect_cells(self) -> None:
        for cell in self.all_cells:
            self._connect_single_cell(cell)

    def _connect_single_cell(self, cell: Cell):
        for node_id in self.hypergraph.neighbors(cell.coordinate):
            # print(f"connecting {cell.coordinate} to {node_id}")
            cell.connect(self._cells[node_id], node_id)

    def _nodes_neighbors_group(self):
        """
        Example:
        scenes_dictionary = {
                0: ('0', '1', '2'),
                1: ('2', '3'),
                2: ('4', '1', '5'),
                3: ('3', '6', '7', '4'),
                4: ('6', '7', '8', '9', '10', '3', '4'),
                5: ('2', '11'),
                6: ('11', '12'),
                7: ('13', '11'),
                8: ('1', '2', '14'),
                9: ('13', '15', '16'),
                10: ('11', '12', '17', '18', '19')
            }
            when cell.coordinate = '1'
            then: neighbors_group_dict = {'0': [0, 1, 2], '2': [1, 4, 5], '8': [1, 2, 14]}
            return neighbors_group_dict = {0:..., 1:{0: ('0', '1', '2'), 2: ('4', '1', '5'), 8: ('1', '2', '14')}}
        """
        for cell in self.all_cells:
            neighbors_group_dict = {}
            # "incidence_dict": Dictionary keyed by edge uids with values as the uids of nodes of each edge
            for key, value in self.hypergraph.incidence_dict.items():
                if cell.coordinate in value:
                    neighbors_group_dict[key] = value

            self.nodes_neighbors_group_dict[cell.coordinate] = neighbors_group_dict

    @property
    def node_neighbors_dict(self) -> dict | None:
        """Get nodes' neighbors dict.
        hypergraph.nodes.memberships: The memberships of a node is the set of edges incident to the node in the Hypergraph.

        Returns:
            dict | None: key is node id, value is a list of neighbors' ids.
        Example:
        scenes_dictionary = {
            0: ['0', '1', '5', '9', '12', '13', '14'],
            1: ['0', '2'],
            2: ['0', '1', '2', '3', '5', '7', '10', '11', '12', '13', '14', '15', '16'],
            3: ['1', '3', '4'],
            4: ['2', '4', '10', '17'],
            5: ['3', '6', '7', '8'],
            6: ['4'],
            7: ['5', '6'],
            8: ['6'],
            9: ['7', '8', '9', '16']
            }


        """
        node_neighbors_dict = {}
        for cell in self.all_cells:
            node_neighbors_dict[cell.coordinate] = self.hypergraph.neighbors(cell.coordinate)

        return node_neighbors_dict
