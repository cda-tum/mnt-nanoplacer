import collections
import os
from pathlib import Path
from time import time

import gymnasium as gym
import numpy as np

from mnt import pyfiction
from mnt.nanoplacer.placement_envs.utils import create_action_list, map_to_multidiscrete


class NanoPlacementEnv(gym.Env):
    """Environment used by the RL agent to place gates on the layout and route them via A*.
    Subclass of the gym Environment and reimplements all base functions."""

    def __init__(
        self,
        clocking_scheme: str = "2DDWave",
        technology: str = "QCA",
        layout_width: int = 3,
        layout_height: int = 4,
        benchmark: str = "trindade16",
        function: str = "mux21",
        verbose: int = 1,
        optimize: bool = True,
    ):
        """Constructor."""

        self.last_pos = None
        self.technology = technology
        self.clocking_scheme = (
            "2DDWave"
            if (self.technology == "SiDB" or clocking_scheme.upper() == "2DDWAVE")
            else clocking_scheme.upper()
        )

        self.layout_width = layout_width
        self.layout_height = layout_height

        self.layout = pyfiction.cartesian_obstruction_layout(
            pyfiction.cartesian_gate_layout(
                (self.layout_width - 1, self.layout_height - 1, 1),
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
            self.pi_names,
            self.po_names,
        ) = create_action_list(self.benchmark, self.function)
        self.observation_space = gym.spaces.Discrete(max(self.actions))

        self.action_space = gym.spaces.Discrete(self.layout_width * self.layout_height)

        self.current_node = 0
        self.current_pi = 0
        self.current_po = 0

        self.placement_possible = True
        self.node_dict = collections.defaultdict(int)
        self.max_placed_nodes = 0
        self.current_tries = 0
        self.max_tries = 0
        self.start = time()
        self.placement_times = []
        self.occupied_tiles = np.zeros([self.layout_width, self.layout_height], dtype=int)
        self.gates = np.zeros([self.layout_width, self.layout_height], dtype=int)
        self.verbose = verbose
        self.layout_mask_width = 4
        self.layout_mask_height = 4
        self.optimize = optimize if self.clocking_scheme.upper() == "2DDWAVE" else False

    def reset(self, seed: int = None, options: dict = None) -> tuple[int, dict]:  # noqa: ARG002
        """Creates a new empty layout and resets all placement variables.

        :param seed:       Sets random seed (not implemented)
        :param options:    Additional options (not implemented)

        :return:           Current observation (node to be placed next)"""
        self.layout = pyfiction.cartesian_obstruction_layout(
            pyfiction.cartesian_gate_layout(
                (self.layout_width - 1, self.layout_height - 1, 1),
                self.clocking_scheme,
            )
        )

        self.current_node = 0
        self.current_pi = 0
        self.current_po = 0
        self.current_tries = 0
        self.placement_possible = True
        self.node_dict = collections.defaultdict(int)
        self.occupied_tiles = np.zeros([self.layout_width, self.layout_height], dtype=int)

        observation = self.current_node

        self.last_pos = None
        self.current_tries = 0
        self.max_tries = 0
        self.gates = np.zeros([self.layout_width, self.layout_height], dtype=int)
        self.layout_mask_width = 4
        self.layout_mask_height = 4

        return observation, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """Taking a step in the environment includes:
            - placing the gate
            - try to route it with it(s) predecessor(s)
            - calculate reward
            - update observation

        :param action:    Discrete action output by the policy network

        :return           observation, reward, done, info
        """
        action = map_to_multidiscrete(action, self.layout_width)
        x = action[0]
        y = action[1]

        preceding_nodes = list(self.DG.predecessors(self.actions[self.current_node]))

        if not self.placement_possible or not self.layout.is_empty_tile((x, y)):
            done = True
            reward = 0
        else:
            placed_node = 0
            if self.node_to_action[self.actions[self.current_node]] == "INPUT":
                self.layout.create_pi(f"{self.pi_names[self.current_pi]}", (x, y))
                placed_node = 1
                self.current_pi += 1
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

                if self.current_tries == 0:
                    self.place_node_with_2_inputs(x=x, y=y, signal_1=signal_1, signal_2=signal_2)
                    self.layout.move_node(self.layout.get_node((x, y)), (x, y), [])
                else:
                    self.layout.move_node(self.layout.get_node(self.last_pos), (x, y), [])

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

                    else:
                        self.current_tries += 1
                        for el in path_node_1:
                            self.layout.clear_obstructed_coordinate(el)
                else:
                    self.current_tries += 1

                if self.current_tries == self.max_tries:
                    self.placement_possible = False

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

                if self.current_tries == 0:
                    self.place_node_with_1_input(x, y, signal)
                    self.layout.move_node(self.layout.get_node((x, y)), (x, y), [])

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
                    if self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
                        self.current_po += 1
                    self.current_tries = 0
                    for fanins in self.layout.fanins((x, y)):
                        fanin = fanins
                        while fanin != layout_tile:
                            self.layout.obstruct_coordinate(fanin)
                            self.occupied_tiles[fanin.x][fanin.y] = 1
                            fanin = self.layout.fanins(fanin)[0]

                if self.current_tries == self.max_tries:
                    self.placement_possible = False
            else:
                error_message = f"Not a valid node: {self.node_to_action[self.actions[self.current_node]]}"
                raise Exception(error_message)

            self.node_dict[self.actions[self.current_node]] = self.layout.get_node((x, y))

            if placed_node:
                self.current_node += 1
                self.occupied_tiles[x][y] = 1
                self.gates[x][y] = 1
                self.layout.obstruct_coordinate((x, y, 0))
                self.layout.obstruct_coordinate((x, y, 1))

            reward, done = self.calculate_reward(
                x=x,
                y=y,
                placed_node=placed_node,
            )

        observation = self.current_node

        info = {}
        return observation, reward, done, False, info

    def save_layout(self):
        """Creates cell layout and saves it as .svg for QCA and .dot for SiDB.
        If technology is set to gate-level, it will be saved as an .fgl file."""
        if not Path.exists(Path("layouts")):
            Path.mkdir(Path("layouts"), parents=True)

        if self.technology.lower() == "qca":
            try:
                cell_layout = pyfiction.apply_qca_one_library(self.layout)
                params = pyfiction.write_qca_layout_svg_params()
                params.simple = len(self.actions) > 200
                pyfiction.write_qca_layout_svg(
                    cell_layout,
                    os.path.join("layouts", f"{self.function}_{self.clocking_scheme}_qca.svg"),
                    params,
                )
            finally:
                pass
        elif self.technology.lower() == "sidb":
            try:
                hex_layout = pyfiction.hexagonalization(self.layout)
                pyfiction.write_dot_layout(hex_layout, os.path.join("layouts", f"{self.function}_ROW_sidb.dot"))
            finally:
                pass
        elif self.technology.lower() == "gate-level":
            pyfiction.write_fgl_layout(
                self.layout,
                os.path.join(
                    "layouts",
                    f"{self.function}_ONE_{self.clocking_scheme}_NanoPlaceR_{'Un' if not self.optimize else ''}Opt_UnOrd.fgl",
                ),
            )
        else:
            error_message = f"Not a supported technology: {self.technology}"
            raise Exception(error_message)

    def place_node_with_1_input(self, x: int, y: int, signal: int):
        """Place gate with a single input on a Cartesian grid."""
        if self.node_to_action[self.actions[self.current_node]] == "INV":
            self.layout.create_not(signal, (x, y))
        elif self.node_to_action[self.actions[self.current_node]] in ("FAN-OUT", "BUF"):
            self.layout.create_buf(signal, (x, y))
        elif self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
            self.layout.create_po(
                signal,
                f"{self.po_names[self.current_po]}",
                (x, y),
            )
        else:
            raise Exception

    def place_node_with_2_inputs(self, x: int, y: int, signal_1: int, signal_2: int):
        """Place gate with two inputs on a Cartesian grid."""
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

    def action_masks(self) -> list[np.matrix]:
        """Calculate action mask based on current partial placement.
        Additionally, checks termination criteria to stop current placement.

        :return:    Action masks"""
        preceding_nodes = list(self.DG.predecessors(self.actions[self.current_node]))
        possible_positions_nodes = np.ones([self.layout_width, self.layout_height], dtype=int)

        self.layout_mask_width = int(8 + ((self.current_node * (self.layout_width - 8)) / len(self.actions))) + 1
        self.layout_mask_height = int(8 + ((self.current_node * (self.layout_height - 8)) / len(self.actions))) + 1
        if (
            self.node_to_action[self.actions[self.current_node]] not in ["INPUT", "OUTPUT"]
            and len(preceding_nodes) != 1
        ) and (
            self.node_to_action[self.actions[self.current_node]] != "OUTPUT"
            and self.clocking_scheme.upper() == "2DDWAVE"
        ):
            possible_positions_nodes[: self.layout_mask_width, : self.layout_mask_height] = 0

        if self.node_to_action[self.actions[self.current_node]] == "INPUT":
            if self.clocking_scheme.upper() == "2DDWAVE":
                possible_positions_nodes[0, : self.layout_mask_height] = 0
                possible_positions_nodes[: self.layout_mask_width, 0] = 0
            elif self.clocking_scheme.upper() in ("USE", "RES", "ESR"):
                possible_positions_nodes[0, :] = 0
                possible_positions_nodes[self.layout_width - 1, :] = 0
                possible_positions_nodes[:, 0] = 0
                possible_positions_nodes[:, self.layout_height - 1] = 0
            else:
                error_message = f"Unsupported clocking scheme: {self.clocking_scheme}"
                raise Exception(error_message)

        elif self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
            if self.clocking_scheme.upper() == "2DDWAVE":
                node = self.node_dict[preceding_nodes[0]]
                loc = self.layout.get_tile(node)
                possible_positions_nodes[self.layout_width - 1, loc.y - 1 : self.layout_mask_height] = 0
                possible_positions_nodes[loc.x - 1 : self.layout_mask_width, self.layout_height - 1] = 0
            elif self.clocking_scheme.upper() in ("USE", "RES", "ESR"):
                possible_positions_nodes[0][:] = 0
                possible_positions_nodes[self.layout_width - 1, 0] = 0
                possible_positions_nodes[:, 0] = 0
                possible_positions_nodes[:, self.layout_height - 1] = 0
            else:
                error_message = f"Unsupported clocking scheme: {self.clocking_scheme}"
                raise Exception(error_message)

        elif len(preceding_nodes) == 1 and self.node_to_action[self.actions[self.current_node]] != "OUTPUT":
            node = self.node_dict[preceding_nodes[0]]
            loc = self.layout.get_tile(node)
            for zone in self.layout.outgoing_clocked_zones(loc):
                if self.clocking_scheme.upper() == "2DDWAVE":
                    if (
                        self.layout.is_empty_tile((zone.x, zone.y, 0))
                        and zone.x < self.layout_mask_width
                        and zone.y < self.layout_mask_height
                    ):
                        possible_positions_nodes[zone.x][zone.y] = 0
                else:
                    if (
                        self.layout.is_empty_tile((zone.x, zone.y, 0))
                        and zone.x in range(0, self.layout_width)
                        and zone.y in range(0, self.layout_height)
                    ):
                        possible_positions_nodes[zone.x][zone.y] = 0
                    for second_zone in self.layout.outgoing_clocked_zones((zone.x, zone.y, 0)):
                        if (
                            self.layout.is_empty_tile((second_zone.x, second_zone.y, 0))
                            and second_zone.x in range(0, self.layout_width)
                            and second_zone.y in range(0, self.layout_height)
                        ):
                            possible_positions_nodes[second_zone.x][second_zone.y] = 0

        elif len(preceding_nodes) == 2:
            if self.clocking_scheme.upper() == "2DDWAVE":
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
            if (
                not self.layout.is_po_tile(self.layout.get_tile(self.node_dict[node]))
                and (self.layout.fanout_size(self.node_dict[node]) == 0)
                or (self.layout.fanout_size(self.node_dict[node]) == 1 and self.network.is_fanout(node))
            ):
                possible = False
                tile = self.layout.get_tile(self.node_dict[node])
                for zone in self.layout.outgoing_clocked_zones(tile):
                    if (
                        self.layout.is_empty_tile((zone.x, zone.y, 0))
                        and zone.x != self.layout_width
                        and zone.y != self.layout_height
                    ) or (
                        (
                            self.layout.is_empty_tile((zone.x, zone.y, 1))
                            and zone.x != self.layout_width
                            and zone.y != self.layout_height
                        )
                        and self.layout.get_node((zone.x, zone.y, 0)) not in self.node_dict.values()
                    ):
                        possible = True
                params = pyfiction.a_star_params()
                params.crossings = True

                width = self.layout_width + 1
                height = self.layout_height + 1
                if self.clocking_scheme.upper() in ("RES", "ESR"):
                    if ((self.layout_width + 1) % 4) == 1:
                        width += 1
                    elif ((self.layout_height + 1) % 4) == 2:
                        height += 1
                self.layout.resize((width - 1, height - 1, 1))
                if self.clocking_scheme.upper() in ("USE", "RES", "ESR"):
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
                elif (
                    self.clocking_scheme.upper() == "2DDWAVE"
                    and possible
                    and (
                        len(
                            pyfiction.a_star(
                                self.layout,
                                tile,
                                (
                                    min(
                                        self.layout_width,
                                        self.layout_mask_width,
                                    ),
                                    min(
                                        self.layout_height,
                                        self.layout_mask_height,
                                    ),
                                ),
                                params,
                            )
                        )
                        == 0
                    )
                ):
                    possible = False
                self.layout.resize((self.layout_width - 1, self.layout_height - 1, 1))

                if not possible:
                    self.placement_possible = False
        mask_occupied = self.occupied_tiles.flatten(order="F") == 0
        mask = possible_positions_nodes.flatten(order="F") == 0
        if not any(mask):
            self.placement_possible = False
            return [True] * len(mask)
        return [mask[i] & mask_occupied[i] for i in range(len(mask))]

    def calculate_reward(self, x: int, y: int, placed_node: bool) -> tuple[float, bool]:
        """Calculate reward based on whether a node was placed or not.
        If a node was placed, reward is scaled by the location on the layout if the 2DDWave clocking scheme is used.

        :param x:              X-coordinate of the placed gate
        :param y:              Y-coordinate of the placed gate
        :param placed_node:    Indicates whether a gate was placed or not

        :return:               Reward and termination indicator
        """
        reward = 10000 if self.current_node == len(self.actions) else placed_node
        if placed_node and self.clocking_scheme.upper() == "2DDWAVE":
            reward *= 1 - ((x + y) / (self.layout_mask_width * self.layout_mask_height))

        done = bool(self.current_node == len(self.actions) or not self.placement_possible)
        if self.current_node > self.max_placed_nodes:
            print(f"New best placement: {self.current_node}/{len(self.actions)} ({time() - self.start:.2f}s)")
            if self.verbose == 1:
                print(self.layout)
            self.max_placed_nodes = self.current_node
            self.placement_times.append(time() - self.start)
            if self.current_node == len(self.actions):
                print(f"Found solution after {time() - self.start:.2f}s")
                if self.optimize:
                    print(f"Dimension before optimization: {self.layout.x() + 1} x {self.layout.y() + 1}")
                    pyfiction.post_layout_optimization(self.layout)
                    print(self.layout)
                    print(f"Dimension after optimization: {self.layout.x() + 1} x {self.layout.y() + 1}")
                self.save_layout()
                stats = pyfiction.equivalence_checking_stats()
                eq = pyfiction.equivalence_checking(self.layout, self.network, stats)
                print(f"Equivalent: {eq}")

        return reward, done

    def render(self, mode="human"):
        """Render current placement (not implemented).

        :param mode:    Render mode
        """
