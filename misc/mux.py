from fiction import pyfiction


if __name__ == "__main__":
    # Create empty layout
    layout = pyfiction.cartesian_gate_layout((3, 5, 1), "2DDWave")
    layout = pyfiction.cartesian_obstruction_layout(layout)

    # Create 2:1 MUX

    # Inputs
    layout.create_pi("x1", (0, 3))
    layout.create_pi("x2", (0, 0))
    layout.create_pi("x3", (2, 0))

    # Wires
    layout.create_buf(layout.make_signal(layout.get_node((0, 0, 0))), (0, 1))
    layout.create_buf(layout.make_signal(layout.get_node((0, 1, 0))), (1, 1))

    # NOT
    layout.create_not(layout.make_signal(layout.get_node((0, 1, 0))), (0, 2))

    # Wires
    layout.create_buf(layout.make_signal(layout.get_node((0, 2, 0))), (1, 2))
    # AND
    layout.create_and(
        layout.make_signal(layout.get_node((0, 3, 0))), layout.make_signal(layout.get_node((1, 2, 0))), (1, 3)
    )

    # AND
    layout.create_and(
        layout.make_signal(layout.get_node((1, 1, 0))), layout.make_signal(layout.get_node((2, 0, 0))), (2, 1)
    )

    # Wires
    layout.create_buf(layout.make_signal(layout.get_node((2, 1, 0))), (2, 2))

    # OR
    layout.create_or(
        layout.make_signal(layout.get_node((1, 3, 0))), layout.make_signal(layout.get_node((2, 2, 0))), (2, 3)
    )

    # Outputs
    layout.create_po(layout.make_signal(layout.get_node((2, 3, 0))), "f1", (2, 4))

    # for gate in set(layout.gates() + layout.wires()):
    #     layout.obstruct_coordinate(gate)

    params = pyfiction.a_star_params()
    params.crossings = True
    source = (0, 0)
    target = (3, 5)
    print(f"Source coordinate: {source}")
    print(f"Target coordinate: {target}")
    path = pyfiction.a_star(layout, source, target, params)
    print(pyfiction.a_star_distance(layout, source, target))
    print(f"Shortest path: {path}")
    print(layout)
    layout.resize((2, 4))
    print(layout)

    # cell_layout = pyfiction.apply_qca_one_library(layout)
    # pyfiction.write_qca_layout_svg(cell_layout, "mux2.svg", pyfiction.write_qca_layout_svg_params())
