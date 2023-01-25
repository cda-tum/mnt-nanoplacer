from fiction import pyfiction


if __name__ == "__main__":
    # Create empty layout
    layout = pyfiction.cartesian_gate_layout((3, 3, 0))

    # Possible actions

    # Inputs
    x1 = layout.create_pi("x1", (0, 0))
    x2 = layout.create_pi("x2", (0, 1))
    x3 = layout.create_pi("x3", (0, 2))
    x4 = layout.create_pi("x4", (0, 3))

    # WIRES
    w1 = layout.create_buf(x1, (1, 0))
    w2 = layout.create_buf(x3, (1, 2))

    # Gates
    # AND/NAND
    a1 = layout.create_and(x2, w1, (1, 1))
    a2 = layout.create_and(x4, w2, (1, 3))

    w3 = layout.create_buf(a1, (2, 1))
    w4 = layout.create_buf(w3, (2, 2))

    # OR/NOR/XOR/XNOR
    a3 = layout.create_or(w4, a2, (2, 3))
    # a3 = layout.create_nor(w4, a2, (2, 3))
    # a3 = layout.create_xor(w4, a2, (2, 3))
    # a3 = layout.create_or(w4, a2, (2, 3))
    # MAJ
    # layout.create_maj()
    # NOT
    # layout.create_not()

    # Outputs
    f1 = layout.create_po(w4, "f1", (3, 2))
    f2 = layout.create_po(a3, "f2", (3, 3))

    # Clock numbers
    layout.assign_clock_number((0, 0), 1)
    layout.assign_clock_number((0, 1), 2)
    layout.assign_clock_number((0, 2), 3)
    layout.assign_clock_number((0, 3), 4)

    layout.assign_clock_number((1, 0), 2)
    layout.assign_clock_number((1, 1), 3)
    layout.assign_clock_number((1, 2), 4)
    layout.assign_clock_number((1, 3), 1)

    layout.assign_clock_number((2, 0), 3)
    layout.assign_clock_number((2, 1), 4)
    layout.assign_clock_number((2, 2), 1)
    layout.assign_clock_number((2, 3), 2)

    layout.assign_clock_number((3, 0), 4)
    layout.assign_clock_number((3, 1), 1)
    layout.assign_clock_number((3, 2), 2)
    layout.assign_clock_number((3, 3), 3)
    print(layout)
    print(pyfiction.gate_level_drvs(layout))

    cell_layout = pyfiction.apply_qca_one_library(layout)
    print(cell_layout)
    pyfiction.write_qca_layout_svg(cell_layout, "test2.svg", pyfiction.write_qca_layout_svg_params())
