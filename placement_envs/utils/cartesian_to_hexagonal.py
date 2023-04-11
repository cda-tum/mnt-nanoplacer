import math
from typing import Union
from fiction import pyfiction


def to_hex(old_coord: Union[tuple, pyfiction.coordinate], height: int) -> tuple[int, int, int]:
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


def cartesian_to_hexagonal(layout, layout_width, layout_height, hex_layout):
    print(layout)
    print(layout_width)
    print(layout_height)
    print(hex_layout)
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
                        print("here")
                        print(hex)
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
