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


class QCAEnv8(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        clocking_scheme="2DDWave",
        layout_width=7,
        layout_height=7,
        benchmark="fontes18",
        function="clpl",
    ):
        self.clocking_scheme = clocking_scheme
        self.layout_width = layout_width
        self.layout_height = layout_height
        self.layout = pyfiction.cartesian_gate_layout(
            (self.layout_width - 1, self.layout_height - 1, 1),
            self.clocking_scheme,
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
                "current_node": spaces.Discrete(max(self.actions)),
                "node_embedding": spaces.Box(low=-2.0, high=2.0, shape=(1, 20), dtype=np.float32),
                "occupied_tiles": spaces.MultiBinary([self.layout_width, self.layout_height]),
            },
        )

        self.action_space = spaces.Discrete(self.layout_width * self.layout_height)

        self.current_node = 0

        self.placement_possible = True
        self.node_dict = collections.defaultdict(int)
        self.max_placed_nodes = 0
        self.current_tries = 0
        self.min_drvs = np.inf
        self.start = time()
        self.placement_times = []
        self.current_node_embedding = np.array(self.node_embeddings[str(self.actions[self.current_node])]).reshape(
            1, 20
        )
        self.params = pyfiction.color_routing_params()
        self.params.crossings = True
        self.params.path_limit = 1
        # self.params.engine = pyfiction.graph_coloring_engine.MCS
        self.occupied_tiles = np.zeros([self.layout_width, self.layout_height], dtype=int)

    @staticmethod
    def create_action_list(benchmark, function):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, f"../../benchmarks/{benchmark}/{function}.v")
        network = pyfiction.read_logic_network(path)
        params = pyfiction.fanout_substitution_params()
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
            (self.layout_width - 1, self.layout_height - 1, 1),
            self.clocking_scheme,
        )
        self.current_node = 0
        self.current_tries = 0
        self.placement_possible = True
        self.node_dict = collections.defaultdict(int)
        self.current_node_embedding = np.array(self.node_embeddings[str(self.actions[self.current_node])]).reshape(
            1, 20
        )
        self.occupied_tiles = np.zeros([self.layout_width, self.layout_height], dtype=int)

        observation = {
            "current_node": self.current_node,
            "node_embedding": self.current_node_embedding,
            "occupied_tiles": self.occupied_tiles
        }

        self.params.path_limit = 1
        return observation

    def step(self, action):
        action = self.map_to_multidiscrete(action, self.layout_width)
        x = action[0]
        y = action[1]

        preceding_nodes = list(self.DG.predecessors(self.actions[self.current_node]))

        if not self.placement_possible:
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
                max_tries = 5  # sum(self.action_masks())
                layout_tile_1, layout_tile_2 = self.place_node_with_2_inputs(x=x, y=y, preceding_nodes=preceding_nodes)

                if self.node_to_action[self.actions[self.current_node]] == "OR":
                    print(layout_tile_1, layout_tile_2)
                    print(self.layout)
                if not pyfiction.color_routing(
                    self.layout,
                    [(layout_tile_1, (x, y)), (layout_tile_2, (x, y))],
                    params=self.params,
                ):
                    self.layout.clear_tile((x, y))
                    self.current_tries += 1

                else:
                    placed_node = 1
                    self.current_tries = 0

                if self.current_tries == max_tries:
                    self.placement_possible = False

            elif self.node_to_action[self.actions[self.current_node]] in [
                "INV",
                "FAN-OUT",
                "BUF",
            ]:
                self.place_node_with_1_input(x=x, y=y, preceding_nodes=preceding_nodes)
                placed_node = 1

            elif self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
                max_tries = 5 #sum(self.action_masks())
                layout_tile = self.place_node_with_1_input(x, y, preceding_nodes)

                if not pyfiction.color_routing(self.layout, [(layout_tile, (x, y))], params=self.params):
                    self.layout.clear_tile((x, y))
                    self.current_tries += 1
                else:
                    placed_node = 1
                    self.current_tries = 0

                if self.current_tries == max_tries:
                    self.placement_possible = False
            else:
                raise Exception(f"Not a valid node: {self.node_to_action[self.actions[self.current_node]]}")

            self.node_dict[self.actions[self.current_node]] = self.layout.get_node((x, y))
            if placed_node:
                self.current_node += 1
                self.occupied_tiles[x][y] = 1
                self.params.path_limit = self.current_node

            reward, done = self.calculate_reward(
                x=x,
                y=y,
                placed_node=placed_node,
            )
        self.current_node_embedding = np.array(
            self.node_embeddings[str(self.actions[min(self.current_node, len(self.actions) - 1)])]
        ).reshape(1, 20)

        observation = {
            "current_node": self.current_node,
            "node_embedding": self.current_node_embedding,
            "occupied_tiles": self.occupied_tiles
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
        except:
            pass

    def plot_placement_times(self):
        nodes = range(1, len(self.placement_times) + 1)
        plt.plot(self.placement_times, nodes)
        plt.ylabel("Nodes")
        plt.xlabel("Training Time [s]")
        plt.show()

    def place_node_with_1_input(self, x, y, preceding_nodes):
        layout_node = self.node_dict[preceding_nodes[0]]
        layout_tile = self.layout.get_tile(layout_node)
        signal = self.layout.make_signal(layout_node)
        if self.node_to_action[self.actions[self.current_node]] == "INV":
            self.layout.create_not(signal, (x, y))
        elif self.node_to_action[self.actions[self.current_node]] == "FAN-OUT":
            self.layout.create_buf(signal, (x, y))
        elif self.node_to_action[self.actions[self.current_node]] == "BUF":
            self.layout.create_buf(signal, (x, y))
        elif self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
            self.layout.create_po(signal, f"f{self.actions[self.current_node]}", (x, y))
        return layout_tile

    def place_node_with_2_inputs(self, x, y, preceding_nodes):
        layout_node_1 = self.node_dict[preceding_nodes[0]]
        layout_tile_1 = self.layout.get_tile(layout_node_1)
        signal_1 = self.layout.make_signal(layout_node_1)
        layout_node_2 = self.node_dict[preceding_nodes[1]]
        layout_tile_2 = self.layout.get_tile(layout_node_2)
        signal_2 = self.layout.make_signal(layout_node_2)

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

        return layout_tile_1, layout_tile_2

    def action_masks(self):
        preceding_nodes = list(self.DG.predecessors(self.actions[self.current_node]))
        possible_positions_nodes = np.zeros([self.layout_width, self.layout_height], dtype=int)

        if len(preceding_nodes) == 1 and self.node_to_action[self.actions[self.current_node]] != "OUTPUT":
            possible_positions_nodes = np.ones([self.layout_width, self.layout_height], dtype=int)
            node = self.node_dict[preceding_nodes[0]]
            loc = self.layout.get_tile(node)
            for zone in self.layout.outgoing_clocked_zones(loc):
                if self.layout.is_empty_tile((zone.x, zone.y, 0)):
                    possible_positions_nodes[zone.x][zone.y] = 0

        elif len(preceding_nodes) == 2:
            node_1 = self.node_dict[preceding_nodes[0]]
            loc_1 = self.layout.get_tile(node_1)
            node_2 = self.node_dict[preceding_nodes[1]]
            loc_2 = self.layout.get_tile(node_2)
            min_x = max(loc_1.x, loc_2.x)
            min_y = max(loc_1.y, loc_2.y)
            for i in range(self.layout_width):
                for j in range(self.layout_height):
                    if (i < min_x) or (j < min_y):
                        possible_positions_nodes[i][j] = 1

        for i in range(self.layout_width):
            for j in range(self.layout_height):
                if self.node_to_action[self.actions[self.current_node]] == "INPUT":
                    if self.clocking_scheme == "2DDWave":
                        if not (i == 0 or j == 0):
                            possible_positions_nodes[i][j] = 1
                    elif self.clocking_scheme == "USE":
                        if not (i in (0, self.layout_width - 1) or j in (0, self.layout_height - 1)):
                            possible_positions_nodes[i][j] = 1
                    else:
                        raise Exception(f"Unsupported clocking scheme: {self.clocking_scheme}")
                if self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
                    if self.clocking_scheme == "2DDWave":
                        if not (i == self.layout_width or j == self.layout_height):
                            possible_positions_nodes[i][j] = 1
                    elif self.clocking_scheme == "USE":
                        if not (i in (0, self.layout_width - 1) or j in (0, self.layout_height - 1)):
                            possible_positions_nodes[i][j] = 1
                    else:
                        raise Exception(f"Unsupported clocking scheme: {self.clocking_scheme}")

                if not self.layout.is_empty_tile((i, j)):
                    possible_positions_nodes[i][j] = 1
        mask = possible_positions_nodes.flatten(order="F") == 0

        if not any(mask):
            self.placement_possible = False
        return mask

    def calculate_reward(self, x, y, placed_node):
        reward = (
            self.layout_height * self.layout_width - pyfiction.gate_level_drvs(self.layout)[1] + self.current_node
            if self.current_node == len(self.actions)
            else placed_node
        )
        if self.current_node == len(self.actions):
            self.create_cell_layout()
        if placed_node:
            reward *= 1 - ((x + y) / (self.layout_width * self.layout_height))

        done = True if self.current_node == len(self.actions) or not self.placement_possible else False
        if self.current_node > self.max_placed_nodes:
            print(f"New best placement: {self.current_node}/{len(self.actions)} ({time() - self.start:.2f}s)")
            self.max_placed_nodes = self.current_node
            self.placement_times.append(time() - self.start)
            if self.current_node == len(self.actions):
                print(f"Found solution after {time() - self.start:.2f}s")
        if self.current_node == len(self.actions):
            drvs = pyfiction.gate_level_drvs(self.layout)[1]
            if drvs < self.min_drvs:
                print(f"Found improved solution with {drvs} drvs.")
                self.min_drvs = drvs

        return reward, done

    def close(self):
        pass

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

    def get_a_star_path(self, x, y, preceding_node):
        node = self.node_dict[preceding_node]
        params = pyfiction.a_star_params()
        params.crossings = True
        path = pyfiction.a_star(self.layout, self.layout.get_tile(node), (x, y), params)
        return path

    def connect_gates(self, preceding_node, path):
        last_coordinate = self.layout.get_tile(self.node_dict[preceding_node])
        for wire in path:
            self.place_wire(pre=last_coordinate, suc=wire)
            last_coordinate = wire
        return last_coordinate