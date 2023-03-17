import gym
import matplotlib.pyplot as plt
from gym import spaces
from node2vec import Node2Vec
from gensim.models import Word2Vec

import os
import collections
import networkx as nx
import numpy as np
from fiction import pyfiction
from time import time


class QCAEnv7(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self, clocking_scheme="2DDWave", layout_width=3, layout_height=4, benchmark="trindade16", function="mux21"
    ):
        self.clocking_scheme = clocking_scheme
        self.layout_width = layout_width
        self.layout_height = layout_height
        self.layout = pyfiction.cartesian_gate_layout(
            (self.layout_width - 1, self.layout_height - 1, 1), self.clocking_scheme
        )

        self.benchmark = benchmark
        self.function = function
        (
            self.network,
            self.node_to_action,
            self.actions,
            self.DG,
        ) = self.create_action_list(self.benchmark, self.function)
        self.node_embeddings = self.create_node_embedding()
        self.observation_space = spaces.Dict(
            {
                # "occupied_tiles": spaces.MultiBinary([self.layout_width, self.layout_height]),
                # "wire_crossings": spaces.MultiBinary([self.layout_width, self.layout_height]),
                "current_node": spaces.Discrete(max(self.actions)),
                # "node_embedding": spaces.Box(low=-2.0, high=2.0, shape=(1, 20), dtype=np.float32),
                # "placed_wires": spaces.Discrete(self.layout_width * self.layout_height)
            },
        )

        self.action_space = spaces.Discrete(self.layout_width * self.layout_height * 2 * 2)

        self.current_node = 0
        self.occupied_tiles = np.zeros([self.layout_width, self.layout_height], dtype=int)
        self.occupied_tiles_except_inner_wires = np.zeros([self.layout_width, self.layout_height], dtype=int)
        self.wire_crossings = np.zeros([self.layout_width, self.layout_height], dtype=int)

        self.placement_possible = True
        self.layout_node_dict = collections.defaultdict(int)
        self.network_node_dict = collections.defaultdict(dict)
        self.placed_wires = 0
        self.max_placed_nodes = 0
        self.min_drvs = np.inf
        self.start = time()
        self.placement_times = []
        self.current_node_embedding = np.array(self.node_embeddings[str(self.actions[self.current_node])]).reshape(
            1, 20
        )
        self.last_manhattan_distance = np.inf
        self.distance_violations = 0

    @staticmethod
    def create_action_list(benchmark, function):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, f"../../benchmarks/{benchmark}/{function}.v")
        network = pyfiction.read_logic_network(path)
        network = pyfiction.fanout_substitution(network)

        DG = nx.DiGraph()

        # add nodes
        DG.add_nodes_from(network.pis())
        DG.add_nodes_from(network.pos())
        DG.add_nodes_from(network.gates())

        # add edges
        for x in range(max(network.gates()) + 1):
            for pre in network.fanins(x):
                DG.add_edge(pre, x)

        def topological_generations(G):
            indegree_map = {v: d for v, d in G.in_degree() if d > 0}
            zero_indegree = [v for v, d in G.in_degree() if d == 0]

            while zero_indegree:
                node = zero_indegree[0]
                if len(zero_indegree) > 1:
                    zero_indegree = zero_indegree[1:]
                else:
                    zero_indegree = []

                for child in G.neighbors(node):
                    indegree_map[child] -= 1
                    if indegree_map[child] == 0:
                        zero_indegree.insert(0, child)
                        del indegree_map[child]
                yield node

        def topological_sort(G):
            for generation in topological_generations(G):
                yield generation

        actions = list(topological_sort(DG))

        node_to_action = {}
        for action in actions:
            if network.is_pi(action):
                node_to_action[action] = "INPUT"
            elif network.is_po(action):
                node_to_action[action] = "OUTPUT"
            elif network.is_inv(action):
                node_to_action[action] = "INV"
            elif network.is_and(action):
                node_to_action[action] = "AND"
            elif network.is_or(action):
                node_to_action[action] = "OR"
            elif network.is_nand(action):
                node_to_action[action] = "NAND"
            elif network.is_nor(action):
                node_to_action[action] = "NOR"
            elif network.is_xor(action):
                node_to_action[action] = "XOR"
            elif network.is_xnor(action):
                node_to_action[action] = "XNOR"
            elif network.is_maj(action):
                node_to_action[action] = "MAJ"
            elif network.is_fanout(action):
                node_to_action[action] = "FAN-OUT"
            elif network.is_buf(action):
                node_to_action[action] = "BUF"
            else:
                raise Exception(f"{action}")
        return network, node_to_action, actions, DG

    def create_distance_dicts(self):
        distance_gen = nx.all_pairs_shortest_path_length(self.DG.to_undirected())
        distances = {x[0]: x[1] for x in distance_gen}
        return distances

    def create_node_embedding(self):
        if os.path.exists(os.path.join("node_embeddings", f"embeddings_{self.function}.model")):
            model = Word2Vec.load(os.path.join("node_embeddings", f"embeddings_{self.function}.model"))
        else:
            node2vec = Node2Vec(
                self.DG, dimensions=20, walk_length=50, num_walks=200, workers=4
            )  # Use temp_folder for big graphs

            # Embed nodes
            model = node2vec.fit(window=20, min_count=1, batch_words=4)
            model.save(os.path.join("node_embeddings", f"embeddings_{self.function}.model"))
        return model.wv

    def reset(self, seed=None, options=None):
        self.layout = pyfiction.cartesian_gate_layout(
            (self.layout_width - 1, self.layout_height - 1, 1), self.clocking_scheme
        )
        self.current_node = 0
        self.occupied_tiles = np.zeros([self.layout_width, self.layout_height], dtype=int)
        self.wire_crossings = np.zeros([self.layout_width, self.layout_height], dtype=int)
        self.occupied_tiles_except_inner_wires = np.zeros([self.layout_width, self.layout_height], dtype=int)
        self.placement_possible = True
        self.layout_node_dict = collections.defaultdict(int)
        self.network_node_dict = collections.defaultdict(dict)
        self.placed_wires = 0
        self.current_node_embedding = np.array(self.node_embeddings[str(self.actions[self.current_node])]).reshape(
            1, 20
        )
        self.last_manhattan_distance = np.inf
        self.distance_violations = 0

        observation = {
            # "occupied_tiles": self.occupied_tiles,
            # "wire_crossings": self.wire_crossings,
            "current_node": self.current_node,
            # "node_embedding": self.current_node_embedding,
            # "placed_wires": self.placed_wires
        }

        return observation

    def step(self, action):
        action = self.map_to_multidiscrete(action, self.layout_width, self.layout_height)
        x = action[0]
        y = action[1]
        node_or_wire = action[2]
        north_or_west = action[3]
        output_flag = False
        preceding_nodes = list(self.DG.predecessors(self.actions[self.current_node]))

        if not self.placement_possible:
            done = True
            reward = 0
        else:
            placed_node = 0
            if node_or_wire == 0:
                if self.node_to_action[self.actions[self.current_node]] == "INPUT":
                    self.layout.create_pi(f"x{self.actions[self.current_node]}", (x, y))

                elif self.node_to_action[self.actions[self.current_node]] in [
                    "AND",
                    "OR",
                    "XOR",
                ]:
                    self.place_node_with_2_inputs(preceding_nodes=preceding_nodes, x=x, y=y)

                elif self.node_to_action[self.actions[self.current_node]] in [
                    "INV",
                    "FAN-OUT",
                    "OUTPUT",
                    "BUF",
                ]:  # INVERTER
                    self.place_node_with_1_input(preceding_nodes=preceding_nodes, x=x, y=y, north_or_west=north_or_west)

                else:
                    raise Exception(f"Not a valid node: {self.node_to_action[self.actions[self.current_node]]}")

                self.update_dicts_after_node_placement(x=x, y=y)
                self.current_node += 1
                self.placed_wires = 0
                self.occupied_tiles_except_inner_wires[x][y] = 1
                placed_node = 1

            # place wire
            elif node_or_wire == 1:
                self.place_wire(x=x, y=y, north_or_west=north_or_west)

            else:
                raise ValueError(f"action ({action[2]}) must be either 0 or 1.")
            self.occupied_tiles[x][y] = 1

            reward, done = self.calculate_reward(
                preceding_nodes=preceding_nodes,
                x=x,
                y=y,
                node_or_wire=node_or_wire,
                output_flag=output_flag,
                placed_node=placed_node,
            )
        self.current_node_embedding = np.array(
            self.node_embeddings[str(self.actions[min(self.current_node, len(self.actions) - 1)])]
        ).reshape(1, 20)

        observation = {
            # "occupied_tiles": self.occupied_tiles,
            # "wire_crossings": self.wire_crossings,
            "current_node": self.current_node,
            # "node_embedding": self.current_node_embedding,
            # "placed_wires": self.placed_wires
        }

        info = {}
        return observation, reward, done, info

    def render(self):
        print(self.layout)

    def create_cell_layout(self):
        try:
            cell_layout = pyfiction.apply_qca_one_library(self.layout)
            pyfiction.write_qca_layout_svg(
                cell_layout,
                os.path.join("images", f"{self.function}_rl.svg"),
                pyfiction.write_qca_layout_svg_params(),
            )
            # print(pyfiction.gate_level_drvs(self.layout))
            # stats = pyfiction.equivalence_checking_stats()
            # eq = pyfiction.equivalence_checking(self.network, self.layout, stats)
        except:
            pass

    def plot_placement_times(self):
        nodes = range(1, len(self.placement_times) + 1)
        # plt.plot(nodes, self.placement_times)
        plt.plot(self.placement_times, nodes)
        plt.ylabel("Nodes")
        plt.xlabel("Training Time [s]")
        plt.show()

    def neighbor_indices(self, column_number, row_number):
        return self.layout.adjacent_coordinates((column_number, row_number))

    def action_masks(self):
        preceding_nodes = list(self.DG.predecessors(self.actions[self.current_node]))

        pos_pos_nodes_north = np.ones([self.layout_width, self.layout_height], dtype=int)
        pos_pos_wires_north = np.ones([self.layout_width, self.layout_height], dtype=int)
        possible_positions_nodes_north = []

        pos_pos_nodes_west = np.ones([self.layout_width, self.layout_height], dtype=int)
        pos_pos_wires_west = np.ones([self.layout_width, self.layout_height], dtype=int)
        possible_positions_nodes_west = []
        if len(preceding_nodes) == 0:
            pos_pos_nodes_north = np.zeros([self.layout_width, self.layout_height], dtype=int)
            pos_pos_nodes_west = np.zeros([self.layout_width, self.layout_height], dtype=int)
        elif len(preceding_nodes) == 1:
            heads_info = self.network_node_dict[preceding_nodes[0]]
            for head in ["HEAD1", "HEAD2"]:
                if heads_info[head] != 0:
                    loc = self.layout.get_tile(heads_info[head])
                    for layer in range(2):
                        for zone in self.layout.outgoing_clocked_zones((loc.x, loc.y, layer)):
                            if self.layout.is_empty_tile(zone):
                                if zone.x == loc.x:
                                    possible_positions_nodes_north.append(pyfiction.coordinate(zone.x, zone.y, layer))
                                if zone.y == loc.y:
                                    possible_positions_nodes_west.append(pyfiction.coordinate(zone.x, zone.y, layer))

        elif len(preceding_nodes) == 2:
            neighbors_node_1 = []
            heads_info_1 = self.network_node_dict[preceding_nodes[0]]
            head_1_locs = []

            for head in ["HEAD1", "HEAD2"]:
                if heads_info_1[head] != 0:
                    loc = self.layout.get_tile(heads_info_1[head])
                    head_1_locs.append((loc.x, loc.y, 0))
                    head_1_locs.append((loc.x, loc.y, 1))
                    for layer in range(2):
                        for zone in self.layout.outgoing_clocked_zones((loc.x, loc.y, layer)):
                            if self.layout.is_empty_tile(zone):
                                neighbors_node_1.append(pyfiction.coordinate(zone.x, zone.y, layer))

            neighbors_node_2 = []
            heads_info_2 = self.network_node_dict[preceding_nodes[1]]
            for head in ["HEAD1", "HEAD2"]:
                if heads_info_2[head] != 0:
                    loc = self.layout.get_tile(heads_info_2[head])
                    if loc not in head_1_locs:
                        for layer in range(2):
                            for zone in self.layout.outgoing_clocked_zones((loc.x, loc.y, layer)):
                                if self.layout.is_empty_tile(zone):
                                    neighbors_node_2.append(pyfiction.coordinate(zone.x, zone.y, layer))

            for neighbour_node_1 in neighbors_node_1:
                for neighbor_node_2 in neighbors_node_2:
                    if neighbour_node_1 == neighbor_node_2:
                        possible_positions_nodes_north.append(neighbour_node_1)
                        possible_positions_nodes_west.append(neighbour_node_1)

        else:
            raise Exception(f"Too many preceding nodes: {preceding_nodes}")

        if preceding_nodes:
            for preceding_node in preceding_nodes:  # self.network_node_dict.values():
                heads_info = self.network_node_dict[preceding_node]
                if heads_info["OPEN_NODE"]:
                    self.placement_possible = False
                    for head in [heads_info["HEAD1"], heads_info["HEAD2"]]:
                        if head != 0:
                            node = self.layout.get_tile(head)
                            for layer in range(2):
                                for zone in self.layout.outgoing_clocked_zones((node.x, node.y, layer)):
                                    if zone not in [(node.x, node.y, 0), (node.x, node.y, 1)]:
                                        if not self.layout.is_outgoing_signal(
                                            (node.x, node.y, 0),
                                            self.layout.make_signal(self.layout.get_node((zone.x, zone.y, 0))),
                                        ) and not self.layout.is_outgoing_signal(
                                            (node.x, node.y, 1),
                                            self.layout.make_signal(self.layout.get_node((zone.x, zone.y, 1))),
                                        ):
                                            if zone.x == node.x:
                                                pos_pos_wires_north[zone.x][zone.y] = 0
                                                self.placement_possible = True
                                            if zone.y == node.y:
                                                pos_pos_wires_west[zone.x][zone.y] = 0
                                                self.placement_possible = True

        for pos in possible_positions_nodes_north:
            pos_pos_nodes_north[pos.x][pos.y] = 0
        for pos in possible_positions_nodes_west:
            pos_pos_nodes_west[pos.x][pos.y] = 0

        if self.node_to_action[self.actions[self.current_node]] == "INPUT":
            for i in range(self.layout_width):
                for j in range(self.layout_height):
                    if self.clocking_scheme == "2DDWave":
                        if not (i == 0 or j == 0):  # in (0, self.layout_width - 1)  # in (0, self.layout_height - 1)
                            pos_pos_nodes_north[i][j] = 1
                            pos_pos_nodes_west[i][j] = 1
                    elif self.clocking_scheme == "USE":
                        if not (i in (0, self.layout_width - 1) or j in (0, self.layout_height - 1)):
                            pos_pos_nodes_north[i][j] = 1
                            pos_pos_nodes_west[i][j] = 1
                    else:
                        raise Exception(f"Unsupported clocking scheme: {self.clocking_scheme}")

        if self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
            for i in range(self.layout_width):
                for j in range(self.layout_height):
                    if not (i in (0, self.layout_width - 1) or j in (0, self.layout_height - 1)):
                        pos_pos_nodes_north[i][j] = 1
                        pos_pos_nodes_west[i][j] = 1

        for node in self.network_node_dict:
            heads_info = self.network_node_dict[node]
            if heads_info["OPEN_NODE"]:
                possible = False
                for head in [heads_info["HEAD1"], heads_info["HEAD2"]]:
                    if head != 0:
                        tile = self.layout.get_tile(head)
                        for zone in self.layout.outgoing_clocked_zones(tile):
                            if self.layout.is_empty_tile((zone.x, zone.y, 0)):
                                possible = True
                            elif not self.network.is_fanout(self.layout_node_dict[(zone.x, zone.y, 0)]) and self.network.is_buf(self.layout_node_dict[(zone.x, zone.y, 0)]) and self.layout.is_empty_tile((zone.x, zone.y, 1)):
                                possible = True

                # if not possible:
                #     self.placement_possible = False

        if self.node_to_action[self.actions[self.current_node]] in [
            "AND",
            "OR",
            "XOR",
        ]:
            predecessor_node_1_head_1 = self.network_node_dict[preceding_nodes[0]]["HEAD1"]
            predecessor_node_1_head_2 = self.network_node_dict[preceding_nodes[0]]["HEAD2"]

            predecessor_node_2_head_1 = self.network_node_dict[preceding_nodes[1]]["HEAD1"]
            predecessor_node_2_head_2 = self.network_node_dict[preceding_nodes[1]]["HEAD2"]

            if predecessor_node_1_head_1 != 0:
                predecessor_node_1_head_1_loc = self.layout.get_tile(predecessor_node_1_head_1)
                if ((predecessor_node_1_head_1_loc.x + 1) != self.layout_width) and\
                        ((predecessor_node_1_head_1_loc.y + 1) != self.layout_height):
                    if (pos_pos_wires_west[predecessor_node_1_head_1_loc.x + 1][predecessor_node_1_head_1_loc.y] == 0) and\
                            (pos_pos_wires_north[predecessor_node_1_head_1_loc.x][predecessor_node_1_head_1_loc.y + 1] == 0):
                        manhattan_distances_east = []
                        manhattan_distances_south = []
                        if predecessor_node_2_head_1 != 0:
                            manhattan_distances_east.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_1_head_1_loc.x + 1,
                                     predecessor_node_1_head_1_loc.y,
                                     predecessor_node_1_head_1_loc.z),
                                    self.layout.get_tile(predecessor_node_2_head_1),
                                )
                            )
                            manhattan_distances_south.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_1_head_1_loc.x,
                                     predecessor_node_1_head_1_loc.y + 1,
                                     predecessor_node_1_head_1_loc.z),
                                    self.layout.get_tile(predecessor_node_2_head_1),
                                )
                            )
                        if predecessor_node_2_head_2 != 0:
                            manhattan_distances_east.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_1_head_1_loc.x + 1,
                                     predecessor_node_1_head_1_loc.y,
                                     predecessor_node_1_head_1_loc.z),
                                    self.layout.get_tile(predecessor_node_2_head_2),
                                )
                            )
                            manhattan_distances_south.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_1_head_1_loc.x,
                                     predecessor_node_1_head_1_loc.y + 1,
                                     predecessor_node_1_head_1_loc.z),
                                    self.layout.get_tile(predecessor_node_2_head_2),
                                )
                            )
                        manhattan_distance_east = np.mean(manhattan_distances_east)
                        manhattan_distance_south = np.mean(manhattan_distances_south)
                        if manhattan_distance_east < manhattan_distance_south:
                            pos_pos_wires_west[predecessor_node_1_head_1_loc.x + 1][predecessor_node_1_head_1_loc.y] = 0
                            pos_pos_wires_north[predecessor_node_1_head_1_loc.x][predecessor_node_1_head_1_loc.y + 1] = 1
                        else:
                            pos_pos_wires_west[predecessor_node_1_head_1_loc.x + 1][predecessor_node_1_head_1_loc.y] = 1
                            pos_pos_wires_north[predecessor_node_1_head_1_loc.x][predecessor_node_1_head_1_loc.y + 1] = 0

            if predecessor_node_1_head_2 != 0:
                predecessor_node_1_head_2_loc = self.layout.get_tile(predecessor_node_1_head_2)
                if ((predecessor_node_1_head_2_loc.x + 1) != self.layout_width) and\
                        ((predecessor_node_1_head_2_loc.y + 1) != self.layout_height):
                    if (pos_pos_wires_west[predecessor_node_1_head_2_loc.x + 1][predecessor_node_1_head_2_loc.y] == 0) and\
                            (pos_pos_wires_north[predecessor_node_1_head_2_loc.x][predecessor_node_1_head_2_loc.y + 1] == 0):
                        manhattan_distances_east = []
                        manhattan_distances_south = []
                        if predecessor_node_2_head_1 != 0:
                            manhattan_distances_east.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_1_head_2_loc.x + 1,
                                     predecessor_node_1_head_2_loc.y,
                                     predecessor_node_1_head_2_loc.z),
                                    self.layout.get_tile(predecessor_node_2_head_1),
                                )
                            )
                            manhattan_distances_south.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_1_head_2_loc.x,
                                     predecessor_node_1_head_2_loc.y + 1,
                                     predecessor_node_1_head_2_loc.z),
                                    self.layout.get_tile(predecessor_node_2_head_1),
                                )
                            )
                        if predecessor_node_2_head_2 != 0:
                            manhattan_distances_east.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_1_head_2_loc.x + 1,
                                     predecessor_node_1_head_2_loc.y,
                                     predecessor_node_1_head_2_loc.z),
                                    self.layout.get_tile(predecessor_node_2_head_2),
                                )
                            )
                            manhattan_distances_south.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_1_head_2_loc.x,
                                     predecessor_node_1_head_2_loc.y + 1,
                                     predecessor_node_1_head_2_loc.z),
                                    self.layout.get_tile(predecessor_node_2_head_2),
                                )
                            )
                        manhattan_distance_east = np.mean(manhattan_distances_east)
                        manhattan_distance_south = np.mean(manhattan_distances_south)
                        if manhattan_distance_east < manhattan_distance_south:
                            pos_pos_wires_west[predecessor_node_1_head_2_loc.x + 1][predecessor_node_1_head_2_loc.y] = 0
                            pos_pos_wires_north[predecessor_node_1_head_2_loc.x][predecessor_node_1_head_2_loc.y + 1] = 1
                        else:
                            pos_pos_wires_west[predecessor_node_1_head_2_loc.x + 1][predecessor_node_1_head_2_loc.y] = 1
                            pos_pos_wires_north[predecessor_node_1_head_2_loc.x][predecessor_node_1_head_2_loc.y + 1] = 0

            if predecessor_node_2_head_1 != 0:
                predecessor_node_2_head_1_loc = self.layout.get_tile(predecessor_node_2_head_1)
                if ((predecessor_node_2_head_1_loc.x + 1) != self.layout_width) and\
                        ((predecessor_node_2_head_1_loc.y + 1) != self.layout_height):
                    if (pos_pos_wires_west[predecessor_node_2_head_1_loc.x + 1][predecessor_node_2_head_1_loc.y] == 0) and\
                            (pos_pos_wires_north[predecessor_node_2_head_1_loc.x][predecessor_node_2_head_1_loc.y + 1] == 0):
                        manhattan_distances_east = []
                        manhattan_distances_south = []
                        if predecessor_node_1_head_1 != 0:
                            manhattan_distances_east.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_2_head_1_loc.x + 1,
                                     predecessor_node_2_head_1_loc.y,
                                     predecessor_node_2_head_1_loc.z),
                                    self.layout.get_tile(predecessor_node_1_head_1),
                                )
                            )
                            manhattan_distances_south.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_2_head_1_loc.x,
                                     predecessor_node_2_head_1_loc.y + 1,
                                     predecessor_node_2_head_1_loc.z),
                                    self.layout.get_tile(predecessor_node_1_head_1),
                                )
                            )
                        if predecessor_node_1_head_2 != 0:
                            manhattan_distances_east.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_2_head_1_loc.x + 1,
                                     predecessor_node_2_head_1_loc.y,
                                     predecessor_node_2_head_1_loc.z),
                                    self.layout.get_tile(predecessor_node_1_head_2),
                                )
                            )
                            manhattan_distances_south.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_2_head_1_loc.x,
                                     predecessor_node_2_head_1_loc.y + 1,
                                     predecessor_node_2_head_1_loc.z),
                                    self.layout.get_tile(predecessor_node_1_head_2),
                                )
                            )
                        manhattan_distance_east = np.mean(manhattan_distances_east)
                        manhattan_distance_south = np.mean(manhattan_distances_south)
                        if manhattan_distance_east < manhattan_distance_south:
                            pos_pos_wires_west[predecessor_node_2_head_1_loc.x + 1][predecessor_node_2_head_1_loc.y] = 0
                            pos_pos_wires_north[predecessor_node_2_head_1_loc.x][predecessor_node_2_head_1_loc.y + 1] = 1
                        else:
                            pos_pos_wires_west[predecessor_node_2_head_1_loc.x + 1][predecessor_node_2_head_1_loc.y] = 1
                            pos_pos_wires_north[predecessor_node_2_head_1_loc.x][predecessor_node_2_head_1_loc.y + 1] = 0

            if predecessor_node_2_head_2 != 0:
                predecessor_node_2_head_2_loc = self.layout.get_tile(predecessor_node_2_head_2)
                if ((predecessor_node_2_head_2_loc.x + 1) != self.layout_width) and\
                        ((predecessor_node_2_head_2_loc.y + 1) != self.layout_height):
                    if (pos_pos_wires_west[predecessor_node_2_head_2_loc.x + 1][predecessor_node_2_head_2_loc.y] == 0) and\
                            (pos_pos_wires_north[predecessor_node_2_head_2_loc.x][predecessor_node_2_head_2_loc.y + 1] == 0):
                        manhattan_distances_east = []
                        manhattan_distances_south = []
                        if predecessor_node_1_head_1 != 0:
                            manhattan_distances_east.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_2_head_2_loc.x + 1,
                                     predecessor_node_2_head_2_loc.y,
                                     predecessor_node_2_head_2_loc.z),
                                    self.layout.get_tile(predecessor_node_1_head_1),
                                )
                            )
                            manhattan_distances_south.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_2_head_2_loc.x,
                                     predecessor_node_2_head_2_loc.y + 1,
                                     predecessor_node_2_head_2_loc.z),
                                    self.layout.get_tile(predecessor_node_1_head_1),
                                )
                            )
                        if predecessor_node_1_head_2 != 0:
                            manhattan_distances_east.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_2_head_2_loc.x + 1,
                                     predecessor_node_2_head_2_loc.y,
                                     predecessor_node_2_head_2_loc.z),
                                    self.layout.get_tile(predecessor_node_1_head_2),
                                )
                            )
                            manhattan_distances_south.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    (predecessor_node_2_head_2_loc.x,
                                     predecessor_node_2_head_2_loc.y + 1,
                                     predecessor_node_2_head_2_loc.z),
                                    self.layout.get_tile(predecessor_node_1_head_2),
                                )
                            )
                        manhattan_distance_east = np.mean(manhattan_distances_east)
                        manhattan_distance_south = np.mean(manhattan_distances_south)
                        if manhattan_distance_east < manhattan_distance_south:
                            pos_pos_wires_west[predecessor_node_2_head_2_loc.x + 1][predecessor_node_2_head_2_loc.y] = 0
                            pos_pos_wires_north[predecessor_node_2_head_2_loc.x][predecessor_node_2_head_2_loc.y + 1] = 1
                        else:
                            pos_pos_wires_west[predecessor_node_2_head_2_loc.x + 1][predecessor_node_2_head_2_loc.y] = 1
                            pos_pos_wires_north[predecessor_node_2_head_2_loc.x][predecessor_node_2_head_2_loc.y + 1] = 0

        mask_nodes_north = pos_pos_nodes_north.flatten(order="F") == 0
        mask_nodes_west = pos_pos_nodes_west.flatten(order="F") == 0

        mask_wires_north = pos_pos_wires_north.flatten(order="F") == 0
        mask_wires_west = pos_pos_wires_west.flatten(order="F") == 0

        mask_occupied = self.occupied_tiles.flatten(order="F") == 0
        mask_occupied_except_inner_wires = self.occupied_tiles_except_inner_wires.flatten(order="F") == 0

        mask1 = [mask_nodes_north[i] & mask_occupied[i] for i in range(len(mask_nodes_north))]
        mask2 = [mask_wires_north[i] & mask_occupied_except_inner_wires[i] for i in range(len(mask_nodes_north))]
        mask3 = [mask_nodes_west[i] & mask_occupied[i] for i in range(len(mask_nodes_west))]
        mask4 = [mask_wires_west[i] & mask_occupied_except_inner_wires[i] for i in range(len(mask_nodes_west))]

        if any(mask1) or any(mask3):
            mask2 = [False for _ in range(len(mask_nodes_north))]
            mask4 = [False for _ in range(len(mask_nodes_west))]
        mask = np.concatenate([mask1, mask2, mask3, mask4])
        if not any(mask):
            self.placement_possible = False
        return mask

    def place_node_with_2_inputs(self, preceding_nodes, x, y):
        incoming_coordinate_1, incoming_coordinate_2 = self.layout.incoming_clocked_zones((x, y))
        if incoming_coordinate_1.y != y:
            possible_north_coordinate = incoming_coordinate_1
            possible_west_coordinate = incoming_coordinate_2
        else:
            possible_north_coordinate = incoming_coordinate_2
            possible_west_coordinate = incoming_coordinate_1
        predecessor_node_1_head_1 = self.network_node_dict[preceding_nodes[0]]["HEAD1"]
        predecessor_node_1_head_2 = self.network_node_dict[preceding_nodes[0]]["HEAD2"]
        predecessor_node_2_head_1 = self.network_node_dict[preceding_nodes[1]]["HEAD1"]
        predecessor_node_2_head_2 = self.network_node_dict[preceding_nodes[1]]["HEAD2"]

        if pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 0) in [
            self.layout.get_tile(predecessor_node_1_head_1),
            self.layout.get_tile(predecessor_node_1_head_2),
            self.layout.get_tile(predecessor_node_2_head_1),
            self.layout.get_tile(predecessor_node_2_head_2),
        ] and pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 1) in [
            self.layout.get_tile(predecessor_node_1_head_1),
            self.layout.get_tile(predecessor_node_1_head_2),
            self.layout.get_tile(predecessor_node_2_head_1),
            self.layout.get_tile(predecessor_node_2_head_2),
        ]:
            multiple_node_heads_on_north_tile = True
        else:
            multiple_node_heads_on_north_tile = False

        def start_north(wst_head):
            if (
                pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 0)
                in [
                    self.layout.get_tile(predecessor_node_1_head_1),
                    self.layout.get_tile(predecessor_node_1_head_2),
                ]
                and wst_head != "NODE1"
            ):
                nrth_coordinate = pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 0)
                nrth_head = "NODE1"
            elif (
                pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 0)
                in [
                    self.layout.get_tile(predecessor_node_2_head_1),
                    self.layout.get_tile(predecessor_node_2_head_2),
                ]
                and wst_head != "NODE2"
            ):
                nrth_coordinate = pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 0)
                nrth_head = "NODE2"
            elif (
                pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 1)
                in [
                    self.layout.get_tile(predecessor_node_1_head_1),
                    self.layout.get_tile(predecessor_node_1_head_2),
                ]
                and wst_head != "NODE1"
            ):
                nrth_coordinate = pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 1)
                nrth_head = "NODE1"
            elif (
                pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 1)
                in [
                    self.layout.get_tile(predecessor_node_2_head_1),
                    self.layout.get_tile(predecessor_node_2_head_2),
                ]
                and wst_head != "NODE2"
            ):
                nrth_coordinate = pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 1)
                nrth_head = "NODE2"
            else:
                raise Exception

            nrth_node = self.layout.get_node(nrth_coordinate)
            nrth_signal = self.layout.make_signal(nrth_node)
            return nrth_node, nrth_signal, nrth_head

        def start_west(nrth_head):
            if (
                pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 0)
                in [
                    self.layout.get_tile(predecessor_node_1_head_1),
                    self.layout.get_tile(predecessor_node_1_head_2),
                ]
                and nrth_head != "NODE1"
            ):
                wst_coordinate = pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 0)
                wst_head = "NODE1"
            elif (
                pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 0)
                in [
                    self.layout.get_tile(predecessor_node_2_head_1),
                    self.layout.get_tile(predecessor_node_2_head_2),
                ]
                and nrth_head != "NODE2"
            ):
                wst_coordinate = pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 0)
                wst_head = "NODE2"
            elif (
                pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 1)
                in [
                    self.layout.get_tile(predecessor_node_1_head_1),
                    self.layout.get_tile(predecessor_node_1_head_2),
                ]
                and nrth_head != "NODE1"
            ):
                wst_coordinate = pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 1)
                wst_head = "NODE1"
            elif (
                pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 1)
                in [
                    self.layout.get_tile(predecessor_node_2_head_1),
                    self.layout.get_tile(predecessor_node_2_head_2),
                ]
                and nrth_head != "NODE2"
            ):
                wst_coordinate = pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 1)
                wst_head = "NODE2"
            else:
                raise Exception
            wst_node = self.layout.get_node(wst_coordinate)
            wst_signal = self.layout.make_signal(wst_node)
            return wst_node, wst_signal, wst_head

        if multiple_node_heads_on_north_tile:
            west_node, west_signal, west_head = start_west("")
            north_node, north_signal, north_head = start_north(west_head)
        else:
            north_node, north_signal, north_head = start_north("")
            west_node, west_signal, west_head = start_west(north_head)

        if north_head == west_head:
            raise Exception

        if self.node_to_action[self.actions[self.current_node]] == "AND":
            self.layout.create_and(
                north_signal,
                west_signal,
                (x, y),
            )
        elif self.node_to_action[self.actions[self.current_node]] == "OR":
            self.layout.create_or(
                north_signal,
                west_signal,
                (x, y),
            )
        elif self.node_to_action[self.actions[self.current_node]] == "XOR":
            self.layout.create_xor(
                north_signal,
                west_signal,
                (x, y),
            )
        else:
            raise Exception

        west_pre = self.layout_node_dict[west_node]
        north_pre = self.layout_node_dict[north_node]

        if not self.network.is_fanout(west_pre) and (
            self.layout.fanout_size(self.network_node_dict[west_pre]["HEAD1"]) == 1
        ):
            self.network_node_dict[west_pre]["HEAD1"] = 0
            self.network_node_dict[west_pre]["OPEN_NODE"] = False
        if self.network.is_fanout(west_pre):
            if (self.layout.fanout_size(self.network_node_dict[west_pre]["HEAD1"]) == 1) and (
                self.layout.fanout_size(self.network_node_dict[west_pre]["HEAD2"]) == 1
            ):
                if self.layout.is_adjacent_of(
                    self.layout.get_tile(self.network_node_dict[west_pre]["HEAD1"]),
                    (x, y, 0),
                ) or self.layout.is_adjacent_of(
                    self.layout.get_tile(self.network_node_dict[west_pre]["HEAD1"]),
                    (x, y, 1),
                ):
                    self.network_node_dict[west_pre]["HEAD1"] = 0
                else:
                    self.network_node_dict[west_pre]["HEAD2"] = 0
            elif (self.layout.fanout_size(self.network_node_dict[west_pre]["HEAD1"]) == 2) or (
                self.layout.fanout_size(self.network_node_dict[west_pre]["HEAD2"]) == 2
            ):
                self.close_node(west_pre)
            elif (self.layout.fanout_size(self.network_node_dict[west_pre]["HEAD1"]) == 1) and (
                self.layout.fanout_size(self.network_node_dict[west_pre]["HEAD2"]) == 0
            ):
                self.network_node_dict[west_pre]["HEAD1"] = 0
            elif (self.layout.fanout_size(self.network_node_dict[west_pre]["HEAD2"]) == 1) and (
                self.layout.fanout_size(self.network_node_dict[west_pre]["HEAD1"]) == 0
            ):
                self.network_node_dict[west_pre]["HEAD2"] = 0
            if (self.network_node_dict[west_pre]["HEAD1"] == 0) and (self.network_node_dict[west_pre]["HEAD2"] == 0):
                self.network_node_dict[west_pre]["OPEN_NODE"] = False

        if not self.network.is_fanout(north_pre) and (
            self.layout.fanout_size(self.network_node_dict[north_pre]["HEAD1"]) == 1
        ):
            self.network_node_dict[north_pre]["HEAD1"] = 0
            self.network_node_dict[north_pre]["OPEN_NODE"] = False
        if self.network.is_fanout(north_pre):
            if (self.layout.fanout_size(self.network_node_dict[north_pre]["HEAD1"]) == 1) and (
                self.layout.fanout_size(self.network_node_dict[north_pre]["HEAD2"]) == 1
            ):
                if self.layout.is_adjacent_of(
                    self.layout.get_tile(self.network_node_dict[north_pre]["HEAD1"]),
                    (x, y, 0),
                ) or self.layout.is_adjacent_of(
                    self.layout.get_tile(self.network_node_dict[north_pre]["HEAD1"]),
                    (x, y, 1),
                ):
                    self.network_node_dict[north_pre]["HEAD1"] = 0
                else:
                    self.network_node_dict[north_pre]["HEAD2"] = 0
            elif (self.layout.fanout_size(self.network_node_dict[north_pre]["HEAD1"]) == 2) or (
                self.layout.fanout_size(self.network_node_dict[north_pre]["HEAD2"]) == 2
            ):
                self.close_node(north_pre)
            elif (self.layout.fanout_size(self.network_node_dict[north_pre]["HEAD1"]) == 1) and (
                self.layout.fanout_size(self.network_node_dict[north_pre]["HEAD2"]) == 0
            ):
                self.network_node_dict[north_pre]["HEAD1"] = 0
            elif (self.layout.fanout_size(self.network_node_dict[north_pre]["HEAD2"]) == 1) and (
                self.layout.fanout_size(self.network_node_dict[north_pre]["HEAD1"]) == 0
            ):
                self.network_node_dict[north_pre]["HEAD2"] = 0
            if (self.network_node_dict[north_pre]["HEAD1"] == 0) and (self.network_node_dict[north_pre]["HEAD2"] == 0):
                self.network_node_dict[north_pre]["OPEN_NODE"] = False

    def place_node_with_1_input(self, preceding_nodes, x, y, north_or_west):
        if len(self.layout.incoming_clocked_zones((x, y))) == 2:
            incoming_coordinate_1, incoming_coordinate_2 = self.layout.incoming_clocked_zones((x, y))
            if incoming_coordinate_1.y != y:
                possible_north_coordinate = incoming_coordinate_1
                possible_west_coordinate = incoming_coordinate_2
            else:
                possible_north_coordinate = incoming_coordinate_2
                possible_west_coordinate = incoming_coordinate_1
        else:
            incoming_coordinate = self.layout.incoming_clocked_zones((x, y))[0]
            if incoming_coordinate.y != y:
                possible_north_coordinate = incoming_coordinate
                possible_west_coordinate = None
            else:
                possible_north_coordinate = None
                possible_west_coordinate = incoming_coordinate

        predecessor_node_head_1 = self.network_node_dict[preceding_nodes[0]]["HEAD1"]
        predecessor_node_head_2 = self.network_node_dict[preceding_nodes[0]]["HEAD2"]

        if north_or_west == 0:  # connect north
            if pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 0) in [
                self.layout.get_tile(predecessor_node_head_1),
                self.layout.get_tile(predecessor_node_head_2),
            ]:
                coordinate = pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 0)
            elif pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 1) in [
                self.layout.get_tile(predecessor_node_head_1),
                self.layout.get_tile(predecessor_node_head_2),
            ]:
                coordinate = pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 1)
            else:
                raise Exception
        elif north_or_west == 1:  # connect west
            if pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 0) in [
                self.layout.get_tile(predecessor_node_head_1),
                self.layout.get_tile(predecessor_node_head_2),
            ]:
                coordinate = pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 0)
            elif pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 1) in [
                self.layout.get_tile(predecessor_node_head_1),
                self.layout.get_tile(predecessor_node_head_2),
            ]:
                coordinate = pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 1)
            else:
                raise Exception
        else:
            raise Exception
        node = self.layout.get_node(coordinate)
        signal = self.layout.make_signal(node)

        if self.node_to_action[self.actions[self.current_node]] == "INV":
            self.layout.create_not(signal, (x, y))
        elif self.node_to_action[self.actions[self.current_node]] == "FAN-OUT":  # FAN-OUT
            self.layout.create_buf(signal, (x, y))
        elif self.node_to_action[self.actions[self.current_node]] == "OUTPUT":  # OUTPUT
            self.layout.create_po(signal, f"f{self.actions[self.current_node]}", (x, y))
        elif self.node_to_action[self.actions[self.current_node]] == "BUF":
            self.layout.create_buf(signal, (x, y))

        if not self.network.is_fanout(preceding_nodes[0]) and (self.layout.fanout_size(predecessor_node_head_1) == 1):
            self.close_node(preceding_nodes[0])
        if self.network.is_fanout(preceding_nodes[0]):
            if (self.layout.fanout_size(predecessor_node_head_1) == 1) and (
                self.layout.fanout_size(predecessor_node_head_2) == 1
            ):
                if self.layout.is_adjacent_of(
                    self.layout.get_tile(predecessor_node_head_1),
                    (x, y, 0),
                ) or self.layout.is_adjacent_of(
                    self.layout.get_tile(predecessor_node_head_1),
                    (x, y, 1),
                ):
                    self.network_node_dict[preceding_nodes[0]]["HEAD1"] = 0
                else:
                    self.network_node_dict[preceding_nodes[0]]["HEAD2"] = 0
            elif (self.layout.fanout_size(predecessor_node_head_1) == 2) or (
                self.layout.fanout_size(predecessor_node_head_2) == 2
            ):
                self.close_node(preceding_nodes[0])
            elif (self.layout.fanout_size(predecessor_node_head_1) == 1) and (
                self.layout.fanout_size(predecessor_node_head_2) == 0
            ):
                self.network_node_dict[preceding_nodes[0]]["HEAD1"] = 0
            elif (self.layout.fanout_size(predecessor_node_head_2) == 1) and (
                self.layout.fanout_size(predecessor_node_head_1) == 0
            ):
                self.network_node_dict[preceding_nodes[0]]["HEAD2"] = 0
            if (self.network_node_dict[preceding_nodes[0]]["HEAD1"] == 0) and (
                self.network_node_dict[preceding_nodes[0]]["HEAD2"] == 0
            ):
                self.network_node_dict[preceding_nodes[0]]["OPEN_NODE"] = False

    def update_dicts_after_node_placement(self, x, y):
        self.network_node_dict[self.actions[self.current_node]]["HEAD1"] = self.layout.get_node((x, y))
        if self.node_to_action[self.actions[self.current_node]] == "FAN-OUT":
            self.network_node_dict[self.actions[self.current_node]]["HEAD2"] = self.layout.get_node((x, y))
        else:
            self.network_node_dict[self.actions[self.current_node]]["HEAD2"] = 0
        self.network_node_dict[self.actions[self.current_node]]["OPEN_NODE"] = True

        if self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
            self.close_node(self.actions[self.current_node])

        self.layout_node_dict[self.layout.get_node((x, y))] = self.actions[self.current_node]

    def place_wire(self, x, y, north_or_west):
        if len(self.layout.incoming_clocked_zones((x, y))) == 2:
            incoming_coordinate_1, incoming_coordinate_2 = self.layout.incoming_clocked_zones((x, y))
            if incoming_coordinate_1.y != y:
                possible_north_coordinate = incoming_coordinate_1
                possible_west_coordinate = incoming_coordinate_2
            else:
                possible_north_coordinate = incoming_coordinate_2
                possible_west_coordinate = incoming_coordinate_1
        else:
            incoming_coordinate = self.layout.incoming_clocked_zones((x, y))[0]
            if incoming_coordinate.y != y:
                possible_north_coordinate = incoming_coordinate
                possible_west_coordinate = None
            else:
                possible_north_coordinate = None
                possible_west_coordinate = incoming_coordinate

        placed_crossing = False
        if north_or_west == 0:  # connect north
            if not self.layout.is_empty_tile(
                pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 1)
            ) and self.layout.has_no_outgoing_signal(
                pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 1)
            ):
                coordinate = pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 1)
            else:
                coordinate = pyfiction.coordinate(possible_north_coordinate.x, possible_north_coordinate.y, 0)
        elif north_or_west == 1:  # connect west
            if not self.layout.is_empty_tile(
                pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 1)
            ) and self.layout.has_no_outgoing_signal(
                pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 1)
            ):
                coordinate = pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 1)
            else:
                coordinate = pyfiction.coordinate(possible_west_coordinate.x, possible_west_coordinate.y, 0)
        else:
            raise Exception
        node = self.layout.get_node(coordinate)
        signal = self.layout.make_signal(node)
        if not self.layout.is_empty_tile((x, y)):
            if not self.layout.is_at_any_border((x, y)):  # create crossing
                self.layout.create_buf(signal, (x, y, 1))
                placed_crossing = True
            else:
                raise Exception
        else:
            self.layout.create_buf(signal, (x, y))

        z = 1 if placed_crossing else 0
        if not self.node_to_action[self.layout_node_dict[node]] == "FAN-OUT":
            self.network_node_dict[self.layout_node_dict[node]]["HEAD1"] = self.layout.get_node((x, y, z))
            self.network_node_dict[self.layout_node_dict[node]]["HEAD2"] = 0
        else:
            if self.network_node_dict[self.layout_node_dict[node]]["HEAD1"] == self.layout.get_node(coordinate):
                self.network_node_dict[self.layout_node_dict[node]]["HEAD1"] = self.layout.get_node((x, y, z))
            elif self.network_node_dict[self.layout_node_dict[node]]["HEAD2"] == self.layout.get_node(coordinate):
                self.network_node_dict[self.layout_node_dict[node]]["HEAD2"] = self.layout.get_node((x, y, z))
        self.layout_node_dict[self.layout.get_node((x, y, z))] = self.layout_node_dict[node]
        if self.layout.is_at_any_border((x, y, z)) or placed_crossing:
            self.occupied_tiles_except_inner_wires[x][y] = 1
        self.placed_wires += 1

        if placed_crossing:
            self.wire_crossings[x][y] = 1

    def calculate_reward(self, preceding_nodes, x, y, node_or_wire, output_flag, placed_node):
        reward = (
            self.layout_height * self.layout_width - pyfiction.gate_level_drvs(self.layout)[1] + self.current_node
            if self.current_node == len(self.actions)
            else placed_node
        )
        if self.current_node == len(self.actions):
            self.create_cell_layout()
        if placed_node:
            reward *= 1 - ((x + y) / (self.layout_width * self.layout_height))

        if (
            self.placed_wires >= 10 + 2 * self.current_node # self.layout_width + self.layout_height
            and self.node_to_action[self.actions[self.current_node]] != "OUTPUT"
        ):
            output_flag = True
        reward = 0 if output_flag else reward
        done = True if self.current_node == len(self.actions) or output_flag else False
        if self.current_node > self.max_placed_nodes:
            print(f"New best placement: {self.current_node}/{len(self.actions)} ({time() - self.start:.2f}s)")
            print(self.layout)
            self.max_placed_nodes = self.current_node
            self.placement_times.append(time() - self.start)
            if self.current_node == len(self.actions):
                print(f"Found solution after {time() - self.start:.2f}s")
        if self.current_node == len(self.actions):
            self.current_node = 0
            drvs = pyfiction.gate_level_drvs(self.layout)[1]
            if drvs < self.min_drvs:
                print(f"Found improved solution with {drvs} drvs.")
                self.min_drvs = drvs

        if self.node_to_action[self.actions[self.current_node]] in [
            "AND",
            "OR",
            "XOR",
        ]:
            if node_or_wire == 1:
                predecessor_node_1_head_1 = self.network_node_dict[preceding_nodes[0]]["HEAD1"]
                predecessor_node_1_head_2 = self.network_node_dict[preceding_nodes[0]]["HEAD2"]
                predecessor_node_2_head_1 = self.network_node_dict[preceding_nodes[1]]["HEAD1"]
                predecessor_node_2_head_2 = self.network_node_dict[preceding_nodes[1]]["HEAD2"]
                manhattan_distances = []
                for node_1 in [
                    predecessor_node_1_head_1,
                    predecessor_node_1_head_2,
                ]:
                    for node_2 in [
                        predecessor_node_2_head_1,
                        predecessor_node_2_head_2,
                    ]:
                        if node_1 != 0 and node_2 != 0:
                            manhattan_distances.append(
                                pyfiction.manhattan_distance(
                                    self.layout,
                                    self.layout.get_tile(node_1),
                                    self.layout.get_tile(node_2),
                                )
                            )
                if manhattan_distances:
                    reward = (1 - min(manhattan_distances) / (self.layout_width + self.layout_height)) / (
                        self.layout_width + self.layout_height
                    )
                    # if max(manhattan_distances) > self.last_manhattan_distance:
                    #     self.distance_violations += 1

                    #     if self.distance_violations >= 5:
                    #         done = True
                    # else:
                    #     self.distance_violations = 0
                    self.last_manhattan_distance = max(manhattan_distances)
            else:
                self.distance_violations = 0
                self.last_manhattan_distance = np.inf

        if node_or_wire == 1 and self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
            predecessor_node_head_1 = self.network_node_dict[preceding_nodes[0]]["HEAD1"]
            predecessor_node_head_2 = self.network_node_dict[preceding_nodes[0]]["HEAD2"]

            manhattan_distances = []
            if predecessor_node_head_1 != 0:
                manhattan_distances.append(self.layout_width - (self.layout.get_tile(predecessor_node_head_1).x + 1))
                manhattan_distances.append(self.layout_height - (self.layout.get_tile(predecessor_node_head_1).y + 1))
            if predecessor_node_head_2 != 0:
                manhattan_distances.append(self.layout_width - (self.layout.get_tile(predecessor_node_head_2).x + 1))
                manhattan_distances.append(self.layout_height - (self.layout.get_tile(predecessor_node_head_2).y + 1))
            # if manhattan_distances:
            #     for dis in manhattan_distances:
            #         reward += (1 - dis / (self.layout_width + self.layout_height)) / (
            #             (self.layout_width + self.layout_height) * len(manhattan_distances)
            #         )

        return reward, done

    def close_node(self, node):
        self.network_node_dict[node]["HEAD1"] = 0
        self.network_node_dict[node]["HEAD2"] = 0
        self.network_node_dict[node]["OPEN_NODE"] = False

    def close(self):
        pass

    @staticmethod
    def map_to_multidiscrete(action, layout_width, layout_height):
        if action < layout_width * layout_height * 2:
            d = 0
        else:
            d = 1
            action -= layout_width * layout_height * 2
        if action < layout_width * layout_height:
            c = 0
        else:
            c = 1
            action -= layout_width * layout_height
        b = int(action / layout_width)
        a = action % layout_width
        return [a, b, c, d]

    @staticmethod
    def map_to_discrete(a, b, c, d, layout_width, layout_height):
        action = 0
        action += a
        action += b * layout_width
        action += c * layout_width * layout_height
        action += d * layout_width * layout_height * 2
        return action
