from fiction import pyfiction
import numpy as np


if __name__ == "__main__":
	layout = pyfiction.cartesian_obstruction_layout(pyfiction.cartesian_gate_layout((3, 3, 1), "2DDWave"))
	params = pyfiction.color_routing_params()
	params.crossings = True
	params.path_limit = 50
	params.engine = pyfiction.graph_coloring_engine.MCS
	gates = np.zeros([4, 4], dtype=int)  # used for storing coordinates of gates

	x_0_0 = layout.create_pi("x1", (0, 0))
	layout.obstruct_coordinate((0, 0, 0))
	layout.obstruct_coordinate((0, 0, 1))
	gates[0][0] = 1

	x_1_0 = layout.create_buf(x_0_0, (1, 0))
	layout.obstruct_coordinate((1, 0, 0))
	layout.obstruct_coordinate((1, 0, 1))
	gates[1][0] = 1

	x_1_1 = layout.create_not(x_1_0, (1, 1))
	layout.obstruct_coordinate((1, 1, 0))
	layout.obstruct_coordinate((1, 1, 1))
	gates[1][1] = 1

	x_1_2 = layout.create_pi("x2", (1, 2))
	layout.obstruct_coordinate((1, 2, 0))
	layout.obstruct_coordinate((1, 2, 1))
	gates[1][2] = 1

	x_2_2 = layout.create_buf(x_1_2, (2, 2))
	layout.obstruct_coordinate((2, 2, 0))
	layout.obstruct_coordinate((2, 2, 1))
	gates[2][2] = 1

	x_3_2 = layout.create_and(x_2_2, x_1_0, (3, 2))
	layout.move_node(layout.get_node((3, 2)), (3, 2), [])
	if pyfiction.color_routing(layout, [((2, 2), (3, 2)), ((1, 0), (3, 2))], params):
		print("success")
	# obstruct wires
	for fanin in layout.fanins((3, 2)):
		while fanin not in ((2, 2), (1, 0)):
			layout.obstruct_coordinate(fanin)
			fanin = layout.fanins(fanin)[0]
	layout.obstruct_coordinate((3, 2, 0))
	layout.obstruct_coordinate((3, 2, 1))
	gates[3][2] = 1
	# make sure every gate is obstructed
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))
	print(layout)

	# Error occurs here
	x_2_3 = layout.create_and(x_2_2, x_1_1, (2, 3))
	layout.move_node(layout.get_node((2, 3)), (2, 3), [])
	if pyfiction.color_routing(layout, [((2, 2), (2, 3)), ((1, 1), (2, 3))], params):
		print("success")
	print(layout)
