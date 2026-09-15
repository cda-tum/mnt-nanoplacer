from __future__ import annotations

from pathlib import Path
from time import time
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from mnt import pyfiction
from mnt.nanoplacer.placement_envs.utils import create_action_list, layout_dimensions, map_to_multidiscrete

if TYPE_CHECKING:
    from collections.abc import Callable


class NanoPlacementEnv(gym.Env):
    """Environment used by the RL agent to place gates on the layout and route them via A*.
    Subclass of the Gymnasium environment and reimplements all base functions."""

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
        *,
        routing_fallback: bool = False,
        on_best: Callable[[NanoPlacementEnv], None] | None = None,
    ) -> None:
        """Constructor."""
        super().__init__()

        if technology.lower() not in {"qca", "sidb", "gate-level"}:
            msg = f"Not a supported technology: {technology}"
            raise ValueError(msg)
        if layout_width <= 0 or layout_height <= 0:
            msg = "Layout dimensions must be positive"
            raise ValueError(msg)

        self.last_pos = None
        self.technology = technology
        self.clocking_scheme = (
            "2DDWave"
            if (self.technology.lower() == "sidb" or clocking_scheme.upper() == "2DDWAVE")
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
        self._preceding_nodes = tuple(tuple(self.DG.predecessors(node)) for node in self.actions)
        self.observation_space = gym.spaces.Discrete(len(self.actions) + 1)

        self.action_space = gym.spaces.Discrete(self.layout_width * self.layout_height)

        self.current_node = 0
        self.current_pi = 0
        self.current_po = 0

        self.placement_possible = True
        self.node_dict: dict[int, int] = {}
        self.max_placed_nodes = 0
        self.current_tries = 0
        self.max_tries = 0
        self.tried_positions: set[tuple[int, int]] = set()
        self._action_mask_count: tuple[int, int, int, int] | None = None
        self.start = time()
        self.placement_times = []
        self.occupied_tiles = np.zeros([self.layout_width, self.layout_height], dtype=int)
        self.verbose = verbose
        self.layout_mask_width = 4
        self.layout_mask_height = 4
        self.optimize = optimize if self.clocking_scheme.upper() == "2DDWAVE" else False
        self.routing_fallback = routing_fallback
        self.on_best = on_best
        self.equivalent: str | None = None
        self.verified_solution = False
        self.best_metrics: dict[str, int] = {}
        self.first_solution_time: float | None = None
        self._candidate_verified = False
        dimensions = layout_dimensions.get(self.clocking_scheme, {}).get(benchmark, {}).get(function)
        self.reported_dimensions = list(dimensions) if dimensions else None
        self.target_reproduced = False
        self.target_equivalent: str | None = None

    def reset(self, seed: int | None = None, options: dict[str, object] | None = None) -> tuple[int, dict[str, object]]:  # noqa: ARG002
        """Creates a new empty layout and resets all placement variables.

        :param seed:       Sets the environment's random seed
        :param options:    Additional options (not implemented)

        :return:           Current observation (node to be placed next)"""
        super().reset(seed=seed)
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
        self.node_dict = {}
        self.occupied_tiles = np.zeros([self.layout_width, self.layout_height], dtype=int)

        observation = self.current_node

        self.last_pos = None
        self.max_tries = 0
        self.tried_positions.clear()
        self._action_mask_count = None
        self.layout_mask_width = 4
        self.layout_mask_height = 4
        self._candidate_verified = False

        return observation, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, object]]:
        """Taking a step in the environment includes:
            - placing the gate
            - try to route it with it(s) predecessor(s)
            - calculate reward
            - update observation

        :param action:    Discrete action output by the policy network

        :return:          Observation, reward, termination, truncation, and info
        """
        if not self.action_space.contains(action):
            msg = f"Action {action} is outside the action space"
            raise ValueError(msg)

        x, y = map_to_multidiscrete(action, self.layout_width)

        preceding_nodes = self._preceding_nodes[self.current_node]
        cached_count = self._action_mask_count
        self._action_mask_count = None
        mask_count = (
            cached_count[3]
            if cached_count is not None and cached_count[:3] == (id(self.layout), self.current_node, self.current_tries)
            else None
        )

        if not self.placement_possible or not self.layout.is_empty_tile((x, y)):
            done = True
            reward = 0.0
        else:
            placed_node = False
            if self.node_to_action[self.actions[self.current_node]] == "INPUT":
                self.layout.create_pi(self.pi_names[self.current_pi], (x, y))
                placed_node = True
                self.current_pi += 1
            elif self.node_to_action[self.actions[self.current_node]] in [
                "AND",
                "OR",
                "XOR",
            ]:
                if self.current_tries == 0:
                    self.max_tries = mask_count if mask_count is not None else sum(self.action_masks())
                    self._action_mask_count = None
                self.tried_positions.add((x, y))

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

                path_node_1, path_node_2 = self._two_input_paths(layout_tile_1, layout_tile_2, (x, y))
                if self.routing_fallback and path_node_1 and not path_node_2:
                    path_node_2, path_node_1 = self._two_input_paths(layout_tile_2, layout_tile_1, (x, y))
                if path_node_1 and path_node_2:
                    placed_node = True
                    self.current_tries = 0
                    pyfiction.route_path(self.layout, path_node_1)
                    pyfiction.route_path(self.layout, path_node_2)
                    for el in path_node_2:
                        self.occupied_tiles[el.x][el.y] = 1
                    for el in path_node_1:
                        self.occupied_tiles[el.x][el.y] = 1
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
                    self.max_tries = mask_count if mask_count is not None else sum(self.action_masks())
                    self._action_mask_count = None
                self.tried_positions.add((x, y))

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
                    placed_node = True
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
                msg = f"Not a valid node: {self.node_to_action[self.actions[self.current_node]]}"
                raise ValueError(msg)

            self.node_dict[self.actions[self.current_node]] = self.layout.get_node((x, y))

            if placed_node:
                self.current_node += 1
                self.tried_positions.clear()
                self.occupied_tiles[x][y] = 1
                self.layout.obstruct_coordinate((x, y, 0))
                self.layout.obstruct_coordinate((x, y, 1))

            reward, done = self.calculate_reward(
                x=x,
                y=y,
                placed_node=placed_node,
            )

        observation = self.current_node

        info = (
            {
                "complete_candidate": self.current_node == len(self.actions),
                "verified": self._candidate_verified,
                "routing_failed": not self.placement_possible and self.current_node < len(self.actions),
                "target_reproduced": self.target_reproduced,
                "target_equivalent": self.target_equivalent,
            }
            if done
            else {}
        )
        return observation, reward, done, False, info

    def _two_input_paths(self, source_1, source_2, target):
        """Try one routing order, undoing temporary obstructions on failure."""
        params = pyfiction.a_star_params()
        params.crossings = True
        path_1 = pyfiction.a_star(self.layout, source_1, target, params)
        if not path_1:
            return path_1, []
        # Keep legacy behavior by default. Retrying must preserve existing marks,
        # including occupied source/target tiles, before trying the other order.
        temporary = (
            [coordinate for coordinate in path_1 if not self.layout.is_obstructed_coordinate(coordinate)]
            if self.routing_fallback
            else path_1
        )
        for coordinate in temporary:
            self.layout.obstruct_coordinate(coordinate)
        path_2 = []
        try:
            path_2 = pyfiction.a_star(self.layout, source_2, target, params)
        finally:
            if not path_2:
                for coordinate in temporary:
                    self.layout.clear_obstructed_coordinate(coordinate)
        if path_2:
            if self.routing_fallback:
                for coordinate in path_1:
                    self.layout.obstruct_coordinate(coordinate)
            for coordinate in path_2:
                self.layout.obstruct_coordinate(coordinate)
        return path_1, path_2

    def save_layout(self) -> None:
        """Creates cell layout and saves it as .svg for QCA and .dot for SiDB.
        If technology is set to gate-level, it will be saved as an .fgl file."""
        output_dir = Path("layouts")
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.technology.lower() == "qca":
            cell_layout = pyfiction.apply_qca_one_library(self.layout)
            params = pyfiction.write_qca_layout_svg_params()
            params.simple = len(self.actions) > 200
            path = output_dir / f"{self.function}_{self.clocking_scheme}_qca.svg"
            pyfiction.write_qca_layout_svg(cell_layout, str(path), params)
        elif self.technology.lower() == "sidb":
            hex_layout = pyfiction.hexagonalization(self.layout)
            path = output_dir / f"{self.function}_ROW_sidb.dot"
            pyfiction.write_dot_layout(hex_layout, str(path))
        elif self.technology.lower() == "gate-level":
            path = output_dir / (
                f"{self.function}_ONE_{self.clocking_scheme}_NanoPlaceR_"
                f"{'Un' if not self.optimize else ''}Opt_UnOrd_area.fgl"
            )
            pyfiction.write_fgl_layout(
                self.layout,
                str(path),
            )
        else:
            msg = f"Not a supported technology: {self.technology}"
            raise ValueError(msg)

    def place_node_with_1_input(self, x: int, y: int, signal: int) -> None:
        """Place gate with a single input on a Cartesian grid."""
        if self.node_to_action[self.actions[self.current_node]] == "INV":
            self.layout.create_not(signal, (x, y))
        elif self.node_to_action[self.actions[self.current_node]] in ("FAN-OUT", "BUF"):
            self.layout.create_buf(signal, (x, y))
        elif self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
            self.layout.create_po(
                signal,
                self.po_names[self.current_po],
                (x, y),
            )
        else:
            msg = "Current node does not have exactly one input"
            raise ValueError(msg)

    def place_node_with_2_inputs(self, x: int, y: int, signal_1: int, signal_2: int) -> None:
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
            msg = "Current node does not have exactly two inputs"
            raise ValueError(msg)

    def action_masks(self) -> list[bool]:
        """Calculate action mask based on current partial placement.
        Additionally, checks termination criteria to stop current placement.

        :return:    Action masks"""
        self._action_mask_count = None
        if self.current_node >= len(self.actions):
            return [True] * self.action_space.n

        preceding_nodes = self._preceding_nodes[self.current_node]
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
                msg = f"Unsupported clocking scheme: {self.clocking_scheme}"
                raise ValueError(msg)

        elif self.node_to_action[self.actions[self.current_node]] == "OUTPUT":
            if self.clocking_scheme.upper() == "2DDWAVE":
                node = self.node_dict[preceding_nodes[0]]
                loc = self.layout.get_tile(node)
                possible_positions_nodes[self.layout_width - 1, max(0, loc.y - 1) : self.layout_mask_height] = 0
                possible_positions_nodes[max(0, loc.x - 1) : self.layout_mask_width, self.layout_height - 1] = 0
            elif self.clocking_scheme.upper() in ("USE", "RES", "ESR"):
                possible_positions_nodes[0, :] = 0
                possible_positions_nodes[self.layout_width - 1, :] = 0
                possible_positions_nodes[:, 0] = 0
                possible_positions_nodes[:, self.layout_height - 1] = 0
            else:
                msg = f"Unsupported clocking scheme: {self.clocking_scheme}"
                raise ValueError(msg)

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
                        and 0 <= zone.x < self.layout_width
                        and 0 <= zone.y < self.layout_height
                    ):
                        possible_positions_nodes[zone.x][zone.y] = 0
                    for second_zone in self.layout.outgoing_clocked_zones((zone.x, zone.y, 0)):
                        if (
                            self.layout.is_empty_tile((second_zone.x, second_zone.y, 0))
                            and 0 <= second_zone.x < self.layout_width
                            and 0 <= second_zone.y < self.layout_height
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
            if self.current_tries and node == self.actions[self.current_node]:
                # This unrouted placeholder will move on the next attempt; it is not a placed gate.
                continue
            if (
                not self.layout.is_po_tile(self.layout.get_tile(self.node_dict[node]))
                and self.layout.fanout_size(self.node_dict[node]) == 0
            ) or (self.layout.fanout_size(self.node_dict[node]) == 1 and self.network.is_fanout(node)):
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
                        goals.append((0, height - 1))
                    elif (width % 2 == 1) and (height % 2 == 1):
                        goals.append((width - 1, 0))
                    elif (width % 2 == 0) and (height % 2 == 1):
                        goals.append((width - 1, height - 1))
                    elif (width % 2 == 1) and (height % 2 == 0):
                        goals.append((width - 1, 0))
                        goals.append((0, height - 1))
                    else:
                        msg = "Unable to determine a routing goal"
                        raise ValueError(msg)
                    possible = any(pyfiction.a_star(self.layout, tile, goal, params) for goal in goals)
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
        mask = (possible_positions_nodes == 0) & (self.occupied_tiles == 0)
        for tried_position in self.tried_positions:
            mask[tried_position] = False
        mask = mask.flatten(order="F")
        if not mask.any():
            self.placement_possible = False
            return [True] * len(mask)
        if self.current_tries == 0:
            # Only hand the count to the next step; feasibility is recomputed on every mask request.
            self._action_mask_count = (
                id(self.layout),
                self.current_node,
                self.current_tries,
                int(np.count_nonzero(mask)),
            )
        return mask.tolist()

    def calculate_reward(self, x: int, y: int, placed_node: bool) -> tuple[float, bool]:
        """Calculate reward based on whether a node was placed or not.
        If a node was placed, reward is scaled by the location on the layout if the 2DDWave clocking scheme is used.

        :param x:              X-coordinate of the placed gate
        :param y:              Y-coordinate of the placed gate
        :param placed_node:    Indicates whether a gate was placed or not

        :return:               Reward and termination indicator
        """
        reward = 10000.0 if self.current_node == len(self.actions) else float(placed_node)
        if placed_node and self.clocking_scheme.upper() == "2DDWAVE":
            reward *= 1 - ((x + y) / (self.layout_mask_width * self.layout_mask_height))

        complete = self.current_node == len(self.actions)
        done = bool(complete or not self.placement_possible)
        improved = self.current_node > self.max_placed_nodes
        metrics = {}
        equivalent = None
        if complete:
            initial_width, initial_height = self.layout.x() + 1, self.layout.y() + 1
            target_equivalent = None
            if not self.target_reproduced and self.reported_dimensions == [initial_width, initial_height]:
                # A smaller optimized result alone does not prove the reported starting grid.
                stats = pyfiction.equivalence_checking_stats()
                target_equivalent = pyfiction.equivalence_checking(self.layout, self.network, stats).name
                if target_equivalent in {"STRONG", "WEAK"}:
                    output = Path("layouts")
                    output.mkdir(parents=True, exist_ok=True)
                    target = output / f"{self.function}_{self.clocking_scheme}_reported_target.fgl"
                    temporary = target.with_suffix(".tmp.fgl")
                    pyfiction.write_fgl_layout(self.layout, str(temporary))
                    temporary.replace(target)
                    self.target_reproduced = True
                    self.target_equivalent = target_equivalent
            if self.optimize:
                pyfiction.post_layout_optimization(self.layout)
            if target_equivalent is not None and not self.optimize:
                equivalent = target_equivalent
            else:
                stats = pyfiction.equivalence_checking_stats()
                equivalent = pyfiction.equivalence_checking(self.layout, self.network, stats).name
            self._candidate_verified = equivalent in {"STRONG", "WEAK"}
            metrics = {
                "width": self.layout.x() + 1,
                "height": self.layout.y() + 1,
                "area": (self.layout.x() + 1) * (self.layout.y() + 1),
                "wires": self.layout.num_wires(),
                "crossings": self.layout.num_crossings(),
                "initial_width": initial_width,
                "initial_height": initial_height,
                "initial_area": initial_width * initial_height,
            }
            if self._candidate_verified:
                if self.first_solution_time is None:
                    self.first_solution_time = time() - self.start
                improved = not self.verified_solution or tuple(
                    metrics[key] for key in ("area", "wires", "crossings")
                ) < tuple(self.best_metrics[key] for key in ("area", "wires", "crossings"))
            else:
                improved = improved and not self.verified_solution
            if self.verbose:
                print(f"Complete candidate after {time() - self.start:.2f}s; equivalence: {equivalent}")

        if improved:
            if self.verbose:
                print(f"New best placement: {self.current_node}/{len(self.actions)} ({time() - self.start:.2f}s)")
            if self.verbose == 1:
                print(self.layout)
            self.max_placed_nodes = self.current_node
            self.placement_times.append(time() - self.start)
            self.equivalent = equivalent
            self.best_metrics = metrics
            if complete and self._candidate_verified:
                self.verified_solution = True
                self.save_layout()
            if self.on_best is not None:
                # VecEnv resets terminal layouts immediately after step(), so snapshot here.
                self.on_best(self)

        return float(reward), done

    def render(self) -> None:
        """Render current placement (not implemented)."""
