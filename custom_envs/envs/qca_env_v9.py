import gym
import matplotlib.pyplot as plt
from gym import spaces
import math

import os
import collections
import networkx as nx
import numpy as np
from fiction import pyfiction
from time import time, sleep


class QCAEnv9(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        clocking_scheme="2DDWave",
        layout_width=3,
        layout_height=4,
        benchmark="trindade16",
        function="mux21",
        verbose=1,
    ):
        self.last_pos = None
        self.clocking_scheme = clocking_scheme
        self.layout_width = layout_width
        self.layout_height = layout_height
        self.layout = pyfiction.cartesian_obstruction_layout(
            pyfiction.cartesian_gate_layout(
                (self.layout_width - 1, self.layout_height - 1, 1),
                self.clocking_scheme,
            )
        )
        # hex_height = self.to_hex((self.layout_width - 1, self.layout_height - 1))[1]
        # hex_width = self.to_hex((self.layout_width - 1, 0))[0]
        # self.hex_layout = pyfiction.hexagonal_gate_layout(
        #             (hex_width, hex_height, 1),
        #             "ROW",
        #         )

        self.benchmark = benchmark
        self.function = function
        (
            self.network,
            self.node_to_action,
            self.actions,
            self.DG,
        ) = self.create_action_list(self.benchmark, self.function)
        self.observation_space = spaces.Dict(
            {
                "current_node": spaces.Discrete(max(self.actions)),
            },
        )

        self.action_space = spaces.Discrete(self.layout_width * self.layout_height)

        self.current_node = 0

        self.placement_possible = True
        self.node_dict = collections.defaultdict(int)
        self.node_dict_hex = collections.defaultdict(int)
        self.max_placed_nodes = 0
        self.current_tries = 0
        self.max_tries = 0
        self.min_drvs = np.inf
        self.start = time()
        self.placement_times = []
        self.occupied_tiles = np.zeros([self.layout_width, self.layout_height], dtype=int)
        self.gates = np.zeros([self.layout_width, self.layout_height], dtype=int)
        self.verbose = verbose
        self.layout_mask = 8

    def to_hex(self, old_coord):
        if isinstance(old_coord, tuple):
            old_x, old_y = old_coord
        else:
            old_x = old_coord.x
            old_y = old_coord.y
        y = old_x + old_y
        x = old_x + math.ceil(math.floor(self.layout_height / 2) - y / 2)
        return x, y

    @staticmethod
    def create_action_list(benchmark, function):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, f"../../benchmarks/{benchmark}/{function}.v")
        network = pyfiction.read_logic_network(path)
        params = pyfiction.fanout_substitution_params()
        params.strategy = pyfiction.substitution_strategy.DEPTH
        network = pyfiction.fanout_substitution(network, params)

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

    def reset(self, seed=None, options=None):
        self.layout = pyfiction.cartesian_obstruction_layout(
            pyfiction.cartesian_gate_layout(
                (self.layout_width - 1, self.layout_height - 1, 1),
                self.clocking_scheme,
            )
        )
        # hex_height = self.to_hex((self.layout_width - 1, self.layout_height - 1))[1]
        # hex_width = self.to_hex((self.layout_width - 1, 0))[0]
        # self.hex_layout = pyfiction.hexagonal_gate_layout(
        #     (hex_width, hex_height, 1),
        #     "ROW",
        # )
        self.current_node = 0
        self.current_tries = 0
        self.placement_possible = True
        self.node_dict = collections.defaultdict(int)
        self.node_dict_hex = collections.defaultdict(int)
        self.occupied_tiles = np.zeros([self.layout_width, self.layout_height], dtype=int)

        observation = {
            "current_node": self.current_node,
        }

        self.last_pos = None
        self.current_tries = 0
        self.max_tries = 0
        self.gates = np.zeros([self.layout_width, self.layout_height], dtype=int)
        self.layout_mask = 8

        return observation

    def step(self, action):
        action = self.map_to_multidiscrete(action, self.layout_width)
        x = action[0]
        y = action[1]
        # _hex, y_hex = self.to_hex((x, y))

        preceding_nodes = list(self.DG.predecessors(self.actions[self.current_node]))
        obstruct_coordinates = False

        if not self.placement_possible or not self.layout.is_empty_tile((x, y)):
            done = True
            reward = 0
        else:
            placed_node = 0
            if self.node_to_action[self.actions[self.current_node]] == "INPUT":
                self.layout.create_pi(f"x{self.actions[self.current_node]}", (x, y))
                # self.hex_layout.create_pi(f"x{self.actions[self.current_node]}", (x_hex, y_hex))
                placed_node = 1
            elif self.node_to_action[self.actions[self.current_node]] in [
                "AND",
                "OR",
                "XOR",
            ]:
                if self.current_tries == 0:
                    self.max_tries = sum(self.action_masks())

                layout_node_1 = self.node_dict[preceding_nodes[0]]
                layout_tile_1 = self.layout.get_tile(layout_node_1)
                signal_1 = self.layout.make_signal(layout_node_1)

                layout_node_2 = self.node_dict[preceding_nodes[1]]
                layout_tile_2 = self.layout.get_tile(layout_node_2)
                signal_2 = self.layout.make_signal(layout_node_2)

                # hex
                # layout_node_1_hex = self.node_dict_hex[preceding_nodes[0]]
                # layout_tile_1_hex = self.hex_layout.get_tile(layout_node_1_hex)
                # signal_1_hex = self.hex_layout.make_signal(layout_node_1_hex)

                # layout_node_2_hex = self.node_dict_hex[preceding_nodes[1]]
                # layout_tile_2_hex = self.hex_layout.get_tile(layout_node_2_hex)
                # signal_2_hex = self.hex_layout.make_signal(layout_node_2_hex)

                if self.current_tries == 0:
                    self.place_node_with_2_inputs(x=x, y=y, signal_1=signal_1, signal_2=signal_2)
                    self.layout.move_node(self.layout.get_node((x, y)), (x, y), [])

                    # hex
                    # self.place_node_with_2_inputs_hex(x=x_hex, y=y_hex, signal_1=signal_1_hex, signal_2=signal_2_hex)
                    # self.hex_layout.move_node(self.hex_layout.get_node((x_hex, y_hex)), (x_hex, y_hex), [])
                else:
                    self.layout.move_node(self.layout.get_node(self.last_pos), (x, y), [])

                    # hex
                    # self.hex_layout.move_node(self.hex_layout.get_node((x_hex, y_hex)), (x_hex, y_hex), [])
                self.last_pos = (x, y)

                params = pyfiction.a_star_params()
                params.crossings = True
                path_node_1 = pyfiction.a_star(self.layout, layout_tile_1, (x, y), params)
                if len(path_node_1) != 0:
                    for el in path_node_1:
                        self.layout.obstruct_coordinate(el)
                    path_node_2 = pyfiction.a_star(self.layout, layout_tile_2, (x, y), params)
                    if len(path_node_2) != 0:
                        for el in path_node_2:
                            self.layout.obstruct_coordinate(el)
                        placed_node = 1
                        self.current_tries = 0
                        pyfiction.route_path(self.layout, path_node_1)
                        pyfiction.route_path(self.layout, path_node_2)
                        for el in path_node_2:
                            self.occupied_tiles[el.x][el.y] = 1
                        for el in path_node_1:
                            self.occupied_tiles[el.x][el.y] = 1

                        # hex
                        # hex_path_1 = [self.to_hex(coord) for coord in path_node_1]
                        # hex_path_2 = [self.to_hex(coord) for coord in path_node_2]
                        # pyfiction.route_path(self.hex_layout, hex_path_1)
                        # pyfiction.route_path(self.hex_layout, hex_path_2)
                    else:
                        self.current_tries += 1
                        for el in path_node_1:
                            self.layout.clear_obstructed_coordinate(el)
                else:
                    self.current_tries += 1

                if self.current_tries == self.max_tries:
                    self.placement_possible = False

            # elif self.node_to_action[self.actions[self.current_node]] in [
            #     "INV",
            #     "FAN-OUT",
            #     "BUF",
            # ]:
            #     layout_node = self.node_dict[preceding_nodes[0]]
            #     signal = self.layout.make_signal(layout_node)

            #     self.place_node_with_1_input(x=x, y=y, signal=signal)
            #     placed_node = 1

                # hex
                # layout_node_hex = self.node_dict_hex[preceding_nodes[0]]
                # signal_hex = self.hex_layout.make_signal(layout_node_hex)

                # self.place_node_with_1_input_hex(x=x_hex, y=y_hex, signal=signal_hex)

            elif self.node_to_action[self.actions[self.current_node]] in [
                "INV",
                "FAN-OUT",
                "BUF",
                "OUTPUT",
            ]:
                if self.current_tries == 0:
                    self.max_tries = sum(self.action_masks())

                layout_node = self.node_dict[preceding_nodes[0]]
                layout_tile = self.layout.get_tile(layout_node)
                signal = self.layout.make_signal(layout_node)

                # hex
                # layout_node_hex = self.node_dict_hex[preceding_nodes[0]]
                # layout_tile_hex = self.hex_layout.get_tile(layout_node_hex)
                # signal_hex = self.hex_layout.make_signal(layout_node_hex)

                if self.current_tries == 0:
                    self.place_node_with_1_input(x, y, signal)
                    self.layout.move_node(self.layout.get_node((x, y)), (x, y), [])

                    # hex
                    # self.place_node_with_1_input_hex(x_hex, y_hex, signal_hex)
                    # self.hex_layout.move_node(self.hex_layout.get_node((x_hex, y_hex)), (x_hex, y_hex), [])
                else:
                    self.layout.move_node(self.layout.get_node(self.last_pos), (x, y), [])
                self.last_pos = (x, y)

                params = pyfiction.a_star_params()
                params.crossings = True
                path = pyfiction.a_star(self.layout, layout_tile, (x, y), params)

                if len(path) == 0:
                    self.current_tries += 1
                else:
                    pyfiction.route_path(self.layout, path)
                    placed_node = 1
                    self.current_tries = 0
                    for fanin in self.layout.fanins((x, y)):
                        while fanin != layout_tile:
                            self.layout.obstruct_coordinate(fanin)
                            self.occupied_tiles[fanin.x][fanin.y] = 1
                            fanin = self.layout.fanins(fanin)[0]

                    # hex
                    # hex_path = [self.to_hex(coord) for coord in path]
                    # pyfiction.route_path(self.hex_layout, hex_path)

                if self.current_tries == self.max_tries:
                    self.placement_possible = False
            else:
                raise Exception(f"Not a valid node: {self.node_to_action[self.actions[self.current_node]]}")

            self.node_dict[self.actions[self.current_node]] = self.layout.get_node((x, y))

            # hex
            # self.node_dict_hex[self.actions[self.current_node]] = self.hex_layout.get_node((x_hex, y_hex))
            if placed_node:
                self.current_node += 1
                self.occupied_tiles[x][y] = 1
                self.gates[x][y] = 1
                self.layout.obstruct_coordinate((x, y, 0))
                self.layout.obstruct_coordinate((x, y, 1))
            if obstruct_coordinates:
                for coordinate in list(zip(*np.where(self.gates == 1))):
                    if not self.layout.is_obstructed_coordinate(coordinate + (1,)):
                        self.layout.obstruct_coordinate(coordinate + (1,))
                    if not self.layout.is_obstructed_coordinate(coordinate + (0,)):
                        self.layout.obstruct_coordinate(coordinate + (0,))

            reward, done = self.calculate_reward(
                x=x,
                y=y,
                placed_node=placed_node,
            )

        observation = {
            "current_node": self.current_node,
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
                os.path.join("images", f"{self.function}_{self.clocking_scheme}_rl.svg"),
                pyfiction.write_qca_layout_svg_params(),
            )
        except:
            print("Could not create cell layout.")
            pass
        # try:
        #    pyfiction.write_dot_layout(self.hex_layout, "parity_hex.dot")
        # except:
        #     pass

    def plot_placement_times(self):
        nodes = range(1, len(self.placement_times) + 1)
        plt.plot(self.placement_times, nodes)
        plt.ylabel("Nodes")
        plt.xlabel("Training Time [s]")
        plt.show()

    def place_node_with_1_input(self, x, y, signal):
        if self.node_to_action[self.actions[self.current_node]] == "INV":
            self.layout.create_not(signal, (x, y))
        elif self.node_to_action[self.actions[self.current_node]] == "FAN-OUT":
            self.layout.create_buf(signal, (x, y))
        elif self.node_to_action[self.actions[self.current_node]] == "BUF":
            self.layout.create_buf(signal, (x, y))
        elif self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
            self.layout.create_po(signal, f"f{self.actions[self.current_node]}", (x, y))

    # def place_node_with_1_input_hex(self, x, y, signal):
    #     if self.node_to_action[self.actions[self.current_node]] == "INV":
    #         self.hex_layout.create_not(signal, (x, y))
    #     elif self.node_to_action[self.actions[self.current_node]] == "FAN-OUT":
    #         self.hex_layout.create_buf(signal, (x, y))
    #     elif self.node_to_action[self.actions[self.current_node]] == "BUF":
    #         self.hex_layout.create_buf(signal, (x, y))
    #     elif self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
    #         self.hex_layout.create_po(signal, f"f{self.actions[self.current_node]}", (x, y))

    def place_node_with_2_inputs(self, x, y, signal_1, signal_2):
        if self.node_to_action[self.actions[self.current_node]] == "AND":
            self.layout.create_and(
                signal_1,
                signal_2,
                (x, y),
            )
        elif self.node_to_action[self.actions[self.current_node]] == "OR":
            self.layout.create_or(
                signal_1,
                signal_2,
                (x, y),
            )

        elif self.node_to_action[self.actions[self.current_node]] == "XOR":
            self.layout.create_xor(
                signal_1,
                signal_2,
                (x, y),
            )
        else:
            raise Exception

    # def place_node_with_2_inputs_hex(self, x, y, signal_1, signal_2):
    #     if self.node_to_action[self.actions[self.current_node]] == "AND":
    #         self.hex_layout.create_and(
    #             signal_1,
    #             signal_2,
    #             (x, y),
    #         )
    #     elif self.node_to_action[self.actions[self.current_node]] == "OR":
    #         self.hex_layout.create_or(
    #             signal_1,
    #             signal_2,
    #             (x, y),
    #         )
    #
    #     elif self.node_to_action[self.actions[self.current_node]] == "XOR":
    #         self.hex_layout.create_xor(
    #             signal_1,
    #             signal_2,
    #             (x, y),
    #         )
    #     else:
    #         raise Exception

    def action_masks(self):
        preceding_nodes = list(self.DG.predecessors(self.actions[self.current_node]))
        possible_positions_nodes = np.ones([self.layout_width, self.layout_height], dtype=int)

        self.layout_mask = int(4 + ((self.current_node * (max(self.layout_width, self.layout_height) - 4))/len(self.actions))) + 1
        if self.node_to_action[self.actions[self.current_node]] not in ["INPUT", "OUTPUT"] and len(preceding_nodes) != 1:
            if not self.node_to_action[self.actions[self.current_node]] == "OUTPUT" and self.clocking_scheme == "2DDWave":
                possible_positions_nodes[:self.layout_mask, :self.layout_mask] = 0

        if self.node_to_action[self.actions[self.current_node]] == "INPUT":
            if self.clocking_scheme == "2DDWave":
                possible_positions_nodes[0, :self.layout_mask] = 0
                possible_positions_nodes[:self.layout_mask, 0] = 0
            elif self.clocking_scheme in ("USE", "RES", "CFE"):
                possible_positions_nodes[0, :] = 0
                possible_positions_nodes[self.layout_width - 1, :] = 0
                possible_positions_nodes[:, 0] = 0
                possible_positions_nodes[:, self.layout_height - 1] = 0
            else:
                raise Exception(f"Unsupported clocking scheme: {self.clocking_scheme}")

        elif self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
            if self.clocking_scheme == "2DDWave":
                node = self.node_dict[preceding_nodes[0]]
                loc = self.layout.get_tile(node)
                possible_positions_nodes[self.layout_width - 1, loc.y - 1:self.layout_mask] = 0
                possible_positions_nodes[loc.x - 1:self.layout_mask, self.layout_height - 1] = 0
            elif self.clocking_scheme in ("USE", "RES", "CFE"):
                possible_positions_nodes[0][:] = 0
                possible_positions_nodes[self.layout_width - 1, 0] = 0
                possible_positions_nodes[:, 0] = 0
                possible_positions_nodes[:, self.layout_height - 1] = 0
            else:
                raise Exception(f"Unsupported clocking scheme: {self.clocking_scheme}")

        elif len(preceding_nodes) == 1 and self.node_to_action[self.actions[self.current_node]] != "OUTPUT":
            node = self.node_dict[preceding_nodes[0]]
            loc = self.layout.get_tile(node)
            for zone in self.layout.outgoing_clocked_zones(loc):
                if self.clocking_scheme == "2DDWave":
                    if self.layout.is_empty_tile((zone.x, zone.y, 0)) and zone.x < self.layout_mask and zone.y < self.layout_mask:
                        possible_positions_nodes[zone.x][zone.y] = 0
                else:
                    if self.layout.is_empty_tile(
                            (zone.x, zone.y, 0)) and zone.x in range(0, self.layout_width) and zone.y in range(0, self.layout_height):
                        possible_positions_nodes[zone.x][zone.y] = 0
                    for second_zone in self.layout.outgoing_clocked_zones((zone.x, zone.y, 0)):
                        if self.layout.is_empty_tile(
                                (second_zone.x, second_zone.y, 0)) and second_zone.x in range(0, self.layout_width) and\
                                second_zone.y in range(0, self.layout_height):
                            possible_positions_nodes[second_zone.x][second_zone.y] = 0

        elif len(preceding_nodes) == 2:
            if self.clocking_scheme == "2DDWave":
                node_1 = self.node_dict[preceding_nodes[0]]
                loc_1 = self.layout.get_tile(node_1)
                node_2 = self.node_dict[preceding_nodes[1]]
                loc_2 = self.layout.get_tile(node_2)
                min_x = max(loc_1.x, loc_2.x)
                min_y = max(loc_1.y, loc_2.y)
                if loc_1.x == loc_2.x:
                    min_x += 1
                if loc_1.y == loc_2.y:
                    min_y += 1
                possible_positions_nodes[:min_x, :] = 1
                possible_positions_nodes[:, :min_y] = 1
            else:
                possible_positions_nodes = np.zeros([self.layout_width, self.layout_height], dtype=int)

        for node in self.node_dict:
            if not self.layout.is_po_tile(self.layout.get_tile(self.node_dict[node])):
                if (self.layout.fanout_size(self.node_dict[node]) == 0) or \
                        (self.layout.fanout_size(self.node_dict[node]) == 1 and self.network.is_fanout(node)):
                    possible = False
                    tile = self.layout.get_tile(self.node_dict[node])
                    for zone in self.layout.outgoing_clocked_zones(tile):
                        if self.layout.is_empty_tile((zone.x, zone.y, 0)) and zone.x != self.layout_width and zone.y != self.layout_height:
                            possible = True
                        elif self.layout.is_empty_tile((zone.x, zone.y, 1)) and zone.x != self.layout_width and zone.y != self.layout_height:
                            if self.layout.get_node((zone.x, zone.y, 0)) not in self.node_dict.values():
                                possible = True
                    params = pyfiction.a_star_params()
                    params.crossings = True

                    width = self.layout_width + 1
                    height = self.layout_height + 1
                    if self.clocking_scheme == "RES":
                        if ((self.layout_width + 1) % 4) == 1:
                            width += 1
                        elif ((self.layout_height + 1) % 4) == 2:
                            height += 1
                    self.layout.resize((width - 1, height - 1, 1))
                    if self.clocking_scheme in ("USE", "RES"):
                        goals = []
                        if (width % 2 == 0) and (height % 2 == 0):
                            goals.append((0, width - 1))
                        elif (width % 2 == 1) and (height % 2 == 1):
                            goals.append((width - 1, 0))
                        elif (width % 2 == 0) and (height % 2 == 1):
                            goals.append((width - 1, height - 1))
                        elif (width % 2 == 1) and (height % 2 == 0):
                            goals.append((width - 1, 0))
                            goals.append((0, height - 1))
                        else:
                            raise Exception
                        for goal in goals:
                            overall = False
                            if len(pyfiction.a_star(self.layout, tile, goal, params)) == 0:
                                possible = False
                            else:
                                overall = True
                            if overall:
                                possible = True
                    elif self.clocking_scheme == "2DDWave":
                        if len(pyfiction.a_star(self.layout, tile, (min(self.layout_width, self.layout_mask),
                                                                    min(self.layout_height, self.layout_mask)),
                                                params)) == 0:
                            possible = False
                    self.layout.resize((self.layout_width - 1, self.layout_height - 1, 1))

                    if not possible:
                        self.placement_possible = False
        mask_occupied = self.occupied_tiles.flatten(order="F") == 0
        mask = possible_positions_nodes.flatten(order="F") == 0
        if not any(mask):
            self.placement_possible = False
        return [mask[i] & mask_occupied[i] for i in range(len(mask))]

    def calculate_reward(self, x, y, placed_node):
        reward = (
            10000
            if self.current_node == len(self.actions)
            else placed_node
        )
        if placed_node:
            if self.clocking_scheme == "2DDWave":
                reward *= 1 - ((x + y) / (self.layout_mask * self.layout_mask))

        done = True if self.current_node == len(self.actions) or not self.placement_possible else False
        if self.current_node > self.max_placed_nodes:
            print(f"New best placement: {self.current_node}/{len(self.actions)} ({time() - self.start:.2f}s)")
            if self.verbose == 1:
                print(self.layout)
            self.max_placed_nodes = self.current_node
            self.placement_times.append(time() - self.start)
            if self.current_node == len(self.actions):
                print(f"Found solution after {time() - self.start:.2f}s")
        if self.current_node == len(self.actions):
            drvs = pyfiction.gate_level_drvs(self.layout)[1]
            if drvs < self.min_drvs:
                print(f"Found improved solution with {drvs} drvs.")
                self.create_cell_layout()
                self.min_drvs = drvs

        return reward, done

    @staticmethod
    def map_to_multidiscrete(action, layout_width):
        b = int(action / layout_width)
        a = action % layout_width
        return [a, b]

    @staticmethod
    def map_to_discrete(a, b, layout_width):
        action = 0
        action += a
        action += b * layout_width
        return action
