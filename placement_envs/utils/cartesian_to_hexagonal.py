import math
from typing import Union
from fiction import pyfiction


def to_hex(old_coord: Union[tuple, pyfiction.coordinate], height: int) -> tuple[int, int, int]:
    """Transform Cartesian coordinate to the corresponding coordinate on the hexagonal grid.

    :param old_coord:   coordinate on the Cartesian grid
    :param height:      layout height

    :return:            coordinate on the hexagonal grid"""
    if isinstance(old_coord, tuple):
        old_x, old_y, old_z = old_coord
    else:
        old_x = old_coord.x
        old_y = old_coord.y
        old_z = old_coord.z
    y = old_x + old_y
    x = old_x + math.ceil(math.floor(height / 2) - y / 2)
    z = old_z
    return x, y, z


def cartesian_to_hexagonal(
    layout: pyfiction.cartesian_gate_layout,
    layout_width: int,
    layout_height: int,
    hex_layout: pyfiction.hexagonal_gate_layout,
) -> pyfiction.hexagonal_gate_layout:
    """Transform a Cartesian to a hexagonal layout by remapping each gate/wire.

    :param layout:          Cartesian layout
    :param layout_width     width of the Cartesian layout
    :param layout_height    height of the Cartesian layout
    :param hex_layout       hexagonal layout to map the Cartesian layout to

    :return:                hexagonal layout
    """

    for k in range(layout_width + layout_height - 1):
        for x in range(k + 1):
            y = k - x
            if y < layout_height and x < layout_width:
                for z in range(2):
                    node = (x, y, z)
                    hex = to_hex((x, y, z), layout_height)
                    if layout.is_pi_tile(node):
                        hex_layout.create_pi(layout.get_input_name(layout.get_node(node)), hex)
                    elif layout.is_po_tile(node):
                        hex_layout.create_po(
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[0], layout_height))),
                            layout.get_output_name(layout.get_node(node)),
                            hex,
                        )
                    elif layout.is_wire(layout.get_node(node)):
                        hex_layout.create_buf(
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[0], layout_height))),
                            hex,
                        )
                    elif layout.is_inv(layout.get_node(node)):
                        hex_layout.create_not(
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[0], layout_height))),
                            hex,
                        )
                    elif layout.is_and(layout.get_node(node)):
                        hex_layout.create_and(
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[0], layout_height))),
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[1], layout_height))),
                            hex,
                        )
                    elif layout.is_nand(layout.get_node(node)):
                        hex_layout.create_nand(
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[0], layout_height))),
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[1], layout_height))),
                            hex,
                        )
                    elif layout.is_or(layout.get_node(node)):
                        hex_layout.create_or(
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[0], layout_height))),
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[1], layout_height))),
                            hex,
                        )
                    elif layout.is_nor(layout.get_node(node)):
                        hex_layout.create_nor(
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[0], layout_height))),
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[1], layout_height))),
                            hex,
                        )
                    elif layout.is_xor(layout.get_node(node)):
                        hex_layout.create_xor(
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[0], layout_height))),
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[1], layout_height))),
                            hex,
                        )
                    elif layout.is_xnor(layout.get_node(node)):
                        hex_layout.create_xnor(
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[0], layout_height))),
                            hex_layout.make_signal(hex_layout.get_node(to_hex(layout.fanins(node)[1], layout_height))),
                            hex,
                        )
    return hex_layout
