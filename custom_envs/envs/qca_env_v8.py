import gym
import matplotlib.pyplot as plt
from gym import spaces
# from node2vec import Node2Vec
# from gensim.models import Word2Vec

import os
import collections
import networkx as nx
import numpy as np
from fiction import pyfiction
from time import time


class QCAEnv8(gym.Env):
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
                (self.layout_width, self.layout_height, 1),
                self.clocking_scheme,
            )
        )

        self.benchmark = benchmark
        self.function = function
        (
            self.network,
            self.node_to_action,
            self.actions,
            self.DG,
        ) = self.create_action_list(self.benchmark, self.function)
        # self.node_embeddings = self.create_node_embedding()
        self.observation_space = spaces.Dict(
            {
                "current_node": spaces.Discrete(max(self.actions)),
                # "node_embedding": spaces.Box(low=-2.0, high=2.0, shape=(1, 20), dtype=np.float32),
                # "occupied_tiles": spaces.MultiBinary([self.layout_width, self.layout_height]),
            },
        )
        # self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(1, 20), dtype=np.float32)

        self.action_space = spaces.Discrete(self.layout_width * self.layout_height)

        self.current_node = 0

        self.placement_possible = True
        self.node_dict = collections.defaultdict(int)
        self.max_placed_nodes = 0
        self.current_tries = 0
        self.max_tries = 0
        self.min_drvs = np.inf
        self.start = time()
        self.placement_times = []
        # self.current_node_embedding = np.array(self.node_embeddings[str(self.actions[self.current_node])]).reshape(
        #     1, 20
        #  )
        self.params = pyfiction.color_routing_params()
        self.params.crossings = True
        self.params.path_limit = 50
        self.params.engine = pyfiction.graph_coloring_engine.MCS
        self.occupied_tiles = np.zeros([self.layout_width, self.layout_height], dtype=int)
        self.gates = np.zeros([self.layout_width, self.layout_height], dtype=int)
        self.verbose = verbose
        self.mask_time = 0
        self.step_time = 0
        self.color_route_time = 0
        self.steps = 0
        self.color_routing_steps = 0

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

    # def create_node_embedding(self):
    #     if os.path.exists(os.path.join("node_embeddings", f"embeddings_{self.function}.model")):
    #         model = Word2Vec.load(os.path.join("node_embeddings", f"embeddings_{self.function}.model"))
    #    else:
    #         node2vec = Node2Vec(
    #             self.DG, dimensions=20, walk_length=50, num_walks=200, workers=4
    #         )  # Use temp_folder for big graphs
    #
    #         # Embed nodes
    #         model = node2vec.fit(window=1, min_count=1, batch_words=4)
    #         model.save(os.path.join("node_embeddings", f"embeddings_{self.function}.model"))
    #     return model.wv

    def reset(self, seed=None, options=None):
        self.layout = pyfiction.cartesian_obstruction_layout(
            pyfiction.cartesian_gate_layout(
                (self.layout_width, self.layout_height, 1),
                self.clocking_scheme,
            )
        )
        self.current_node = 0
        self.current_tries = 0
        self.placement_possible = True
        self.node_dict = collections.defaultdict(int)
        # self.current_node_embedding = np.array(self.node_embeddings[str(self.actions[self.current_node])]).reshape(
        #     1, 20
        #  )
        self.occupied_tiles = np.zeros([self.layout_width, self.layout_height], dtype=int)

        observation = {
            "current_node": self.current_node,
            # "node_embedding": self.current_node_embedding,
            # "occupied_tiles": self.occupied_tiles
        }

        self.last_pos = None
        self.current_tries = 0
        self.max_tries = 0
        self.gates = np.zeros([self.layout_width, self.layout_height], dtype=int)

        return observation

    def step(self, action):
        self.steps += 1
        start = time()
        diff = 0
        action = self.map_to_multidiscrete(action, self.layout_width)
        x = action[0]
        y = action[1]

        preceding_nodes = list(self.DG.predecessors(self.actions[self.current_node]))

        if not self.placement_possible or not self.layout.is_empty_tile((x, y)):
            done = True
            reward = 0
        else:
            placed_node = 0
            if self.node_to_action[self.actions[self.current_node]] == "INPUT":
                self.layout.create_pi(f"x{self.actions[self.current_node]}", (x, y))
                placed_node = 1
            elif self.node_to_action[self.actions[self.current_node]] in [
                "AND",
                "OR",
                "XOR",
            ]:
                if self.current_tries == 0:
                    start_diff = time()
                    self.max_tries = sum(self.action_masks())
                    diff = time() - start_diff

                layout_node_1 = self.node_dict[preceding_nodes[0]]
                layout_tile_1 = self.layout.get_tile(layout_node_1)
                signal_1 = self.layout.make_signal(layout_node_1)

                layout_node_2 = self.node_dict[preceding_nodes[1]]
                layout_tile_2 = self.layout.get_tile(layout_node_2)
                signal_2 = self.layout.make_signal(layout_node_2)

                if self.current_tries == 0:
                    self.place_node_with_2_inputs(x=x, y=y, signal_1=signal_1, signal_2=signal_2)
                    self.layout.move_node(self.layout.get_node((x, y)), (x, y), [])
                else:
                    self.layout.move_node(self.layout.get_node(self.last_pos), (x, y), [])
                self.last_pos = (x, y)
                start_color = time()
                self.color_routing_steps += 1
                if not pyfiction.color_routing(
                    self.layout,
                    [(layout_tile_1, (x, y)), (layout_tile_2, (x, y))],
                    params=self.params,
                ):
                    self.current_tries += 1

                else:
                    placed_node = 1
                    self.current_tries = 0

                    for fanin in self.layout.fanins((x, y)):
                        while fanin not in (layout_tile_1, layout_tile_2):
                            self.layout.obstruct_coordinate(fanin)
                            if fanin.x != self.layout_width and fanin.y != self.layout_height:
                                self.occupied_tiles[fanin.x][fanin.y] = 1
                            fanin = self.layout.fanins(fanin)[0]
                self.color_route_time += (time() - start_color)
                if self.current_tries == self.max_tries:
                    self.placement_possible = False

            elif self.node_to_action[self.actions[self.current_node]] in [
                "INV",
                "FAN-OUT",
                "BUF",
            ]:
                layout_node = self.node_dict[preceding_nodes[0]]
                signal = self.layout.make_signal(layout_node)

                self.place_node_with_1_input(x=x, y=y, signal=signal)
                placed_node = 1

            elif self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
                if self.current_tries == 0:
                    start_diff = time()
                    self.max_tries = sum(self.action_masks())
                    diff = time() - start_diff

                layout_node = self.node_dict[preceding_nodes[0]]
                layout_tile = self.layout.get_tile(layout_node)
                signal = self.layout.make_signal(layout_node)

                if self.current_tries == 0:
                    self.place_node_with_1_input(x, y, signal)
                    self.layout.move_node(self.layout.get_node((x, y)), (x, y), [])
                else:
                    self.layout.move_node(self.layout.get_node(self.last_pos), (x, y), [])
                self.last_pos = (x, y)
                start_color = time()
                self.color_routing_steps += 1

                self.layout.move_node(self.layout.get_node((x, y)), (x, y), [])
                if not pyfiction.color_routing(self.layout, [(layout_tile, (x, y))], params=self.params):
                    self.current_tries += 1
                else:
                    placed_node = 1
                    self.current_tries = 0
                    for fanin in self.layout.fanins((x, y)):
                        while fanin != layout_tile:
                            self.layout.obstruct_coordinate(fanin)
                            if fanin.x != self.layout_width and fanin.y != self.layout_height:
                                self.occupied_tiles[fanin.x][fanin.y] = 1
                            fanin = self.layout.fanins(fanin)[0]
                self.color_route_time += (time() - start_color)
                if self.current_tries == self.max_tries:
                    self.placement_possible = False
            else:
                raise Exception(f"Not a valid node: {self.node_to_action[self.actions[self.current_node]]}")

            self.node_dict[self.actions[self.current_node]] = self.layout.get_node((x, y))
            if placed_node:
                self.current_node += 1
                self.occupied_tiles[x][y] = 1
                self.gates[x][y] = 1
                self.layout.obstruct_coordinate((x, y, 0))
                self.layout.obstruct_coordinate((x, y, 1))
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
        # self.current_node_embedding = np.array(
        #     self.node_embeddings[str(self.actions[min(self.current_node, len(self.actions) - 1)])]
        # ).reshape(1, 20)

        observation = {
            "current_node": self.current_node,
            # "node_embedding": self.current_node_embedding,
            # "occupied_tiles": self.occupied_tiles
        }

        info = {}
        self.step_time += (time() - start - diff)
        self.mask_time += diff
        return observation, reward, done, info

    def render(self):
        self.layout.resize((self.layout_width - 1, self.layout_height - 1, 1))
        print(self.layout)
        self.layout.resize((self.layout_width, self.layout_height, 1))

    def create_cell_layout(self):
        try:
            self.layout.resize((self.layout_width - 1, self.layout_height - 1, 1))
            print(self.layout)
            cell_layout = pyfiction.apply_qca_one_library(self.layout)
            pyfiction.write_qca_layout_svg(
                cell_layout,
                os.path.join("images", f"{self.function}_rl.svg"),
                pyfiction.write_qca_layout_svg_params(),
            )
            self.layout.resize((self.layout_width, self.layout_height, 1))
        except:
            print("Could not create cell layout.")
            pass

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

    def action_masks(self):
        start = time()
        preceding_nodes = list(self.DG.predecessors(self.actions[self.current_node]))
        possible_positions_nodes = np.ones([self.layout_width, self.layout_height], dtype=int)

        layout_mask = int(4 + ((self.current_node * (max(self.layout_width, self.layout_height) - 4))/len(self.actions))) + 1
        if self.node_to_action[self.actions[self.current_node]] not in ["INPUT", "OUTPUT"] and len(preceding_nodes) != 1:
            if not self.node_to_action[self.actions[self.current_node]] == "OUTPUT" and self.clocking_scheme == "2DDWave":
                possible_positions_nodes[:layout_mask, :layout_mask] = 0

        if self.node_to_action[self.actions[self.current_node]] == "INPUT":
            if self.clocking_scheme == "2DDWave":
                possible_positions_nodes[0, :layout_mask] = 0
                possible_positions_nodes[:layout_mask, 0] = 0
            elif self.clocking_scheme == "USE":
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
                possible_positions_nodes[self.layout_width - 1, loc.y:layout_mask] = 0
                possible_positions_nodes[loc.x:layout_mask, self.layout_height - 1] = 0
            elif self.clocking_scheme == "USE":
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
                    if self.layout.is_empty_tile((zone.x, zone.y, 0)) and zone.x < layout_mask and zone.y < layout_mask:
                        possible_positions_nodes[zone.x][zone.y] = 0
                else:
                    if self.layout.is_empty_tile(
                            (zone.x, zone.y, 0)) and zone.x in range(0, self.layout_width) and zone.y in range(0, self.layout_height):
                        possible_positions_nodes[zone.x][zone.y] = 0

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
                    if len(pyfiction.a_star(self.layout, tile, (layout_mask, layout_mask), params)) == 0 and self.clocking_scheme == "2DDWave":
                        possible = False
                    if not possible:
                        self.placement_possible = False
        mask_occupied = self.occupied_tiles.flatten(order="F") == 0
        mask = possible_positions_nodes.flatten(order="F") == 0
        if not any(mask):
            self.placement_possible = False
        self.mask_time += (time() - start)
        return [mask[i] & mask_occupied[i] for i in range(len(mask))]

    def calculate_reward(self, x, y, placed_node):
        reward = (
            10000
            if self.current_node == len(self.actions)
            else placed_node
        )
        if placed_node and self.clocking_scheme == "2DDWave":
            reward *= 1 - ((x + y) / (self.layout_width * self.layout_height))

        done = True if self.current_node == len(self.actions) or not self.placement_possible else False
        if self.current_node > self.max_placed_nodes:
            print(f"New best placement: {self.current_node}/{len(self.actions)} ({time() - self.start:.2f}s)")
            print(f"Mask Time: {self.mask_time:.2f}s")
            print(f"Step Time: {self.step_time - self.color_route_time:.2f}s")
            print(f"Color Routing Time: {self.color_route_time:.2f}s")
            print(f"RL Time: {time() - self.start - self.mask_time - self.step_time:.2f}s")
            print(f"Total Steps: {self.steps}")
            print(f"Total color routing calls: {self.color_routing_steps}")
            if self.verbose == 1:
                self.layout.resize((self.layout_width - 1, self.layout_height - 1, 1))
                print(self.layout)
                self.layout.resize((self.layout_width, self.layout_height, 1))
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
