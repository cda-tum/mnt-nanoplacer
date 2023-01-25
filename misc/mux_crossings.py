import numpy as np

from fiction import pyfiction


if __name__ == "__main__":
    # Create empty layout
    layout = pyfiction.cartesian_gate_layout((2, 3, 1), "2DDWave")

    # Create 2:1 MUX

    # Inputs
    layout.create_pi("x1", (0, 0, 0))
    layout.create_pi("x2", (0, 1, 0))
    layout.create_pi("x3", (0, 2, 0))

    # Wires
    layout.create_buf(layout.make_signal(layout.get_node((0, 0, 0))), (1, 0, 0))

    layout.create_buf(layout.make_signal(layout.get_node((0, 1, 0))), (1, 1, 0))
    layout.create_buf(layout.make_signal(layout.get_node((1, 0, 0))), (1, 1, 1))

    # NOT
    layout.create_not(layout.make_signal(layout.get_node((1, 0, 0))), (2, 0))

    # AND
    layout.create_and(
        layout.make_signal(layout.get_node((2, 0, 0))), layout.make_signal(layout.get_node((1, 1, 0))), (2, 1)
    )

    # AND
    layout.create_and(
        layout.make_signal(layout.get_node((1, 1, 1))), layout.make_signal(layout.get_node((0, 2, 0))), (1, 2)
    )

    # OR
    layout.create_or(
        layout.make_signal(layout.get_node((2, 1, 0))), layout.make_signal(layout.get_node((1, 2, 0))), (2, 2)
    )

    a = pyfiction.a_star_distance(layout, (2, 1), (2, 2))
    # Outputs
    layout.create_po(layout.make_signal(layout.get_node((2, 2, 0))), "f1", (2, 3))

    print(layout.is_outgoing_signal((1, 2), layout.make_signal(layout.get_node((2, 2)))))
    print(layout)
    print(pyfiction.gate_level_drvs(layout))

    cell_layout = pyfiction.apply_qca_one_library(layout)
    print(cell_layout)
    pyfiction.write_qca_layout_svg(cell_layout, "mux_crossing.svg", pyfiction.write_qca_layout_svg_params())
