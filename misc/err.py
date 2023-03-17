from fiction import pyfiction
import numpy as np


if __name__ == "__main__":
	layout = pyfiction.cartesian_obstruction_layout(pyfiction.cartesian_gate_layout((11, 12, 1), "2DDWave"))
	params = pyfiction.color_routing_params()
	params.crossings = True
	params.path_limit = 50
	params.engine = pyfiction.graph_coloring_engine.MCS
	gates = np.zeros([12, 13], dtype=int)

	x_0_0 = layout.create_pi("x1", (0, 0))
	layout.obstruct_coordinate((0, 0, 0))
	layout.obstruct_coordinate((0, 0, 1))
	gates[0][0] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_0_1 = layout.create_buf(x_0_0, (0, 1))
	layout.obstruct_coordinate((0, 1, 0))
	layout.obstruct_coordinate((0, 1, 1))
	gates[0][1] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_0_2 = layout.create_not(x_0_1, (0, 2))
	layout.obstruct_coordinate((0, 2, 0))
	layout.obstruct_coordinate((0, 2, 1))
	gates[0][2] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_4_0 = layout.create_pi("x2", (4, 0))
	layout.obstruct_coordinate((4, 0, 0))
	layout.obstruct_coordinate((4, 0, 1))
	gates[4][0] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_4_1 = layout.create_buf(x_4_0, (4, 1))
	layout.obstruct_coordinate((4, 1, 0))
	layout.obstruct_coordinate((4, 1, 1))
	gates[4][1] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_4_5 = layout.create_and(x_0_2, x_4_1, (4, 5))
	layout.move_node(layout.get_node((4, 5)), (4, 5), [])
	if pyfiction.color_routing(layout, [((0, 2), (4, 5)), ((4, 1), (4, 5))], params):
		print("success")
	for fanin in layout.fanins((4, 5)):
		while fanin not in ((0, 2), (4, 1)):
			layout.obstruct_coordinate(fanin)
			fanin = layout.fanins(fanin)[0]
	layout.obstruct_coordinate((4, 5, 0))
	layout.obstruct_coordinate((4, 5, 1))
	gates[4][5] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_5_5 = layout.create_not(x_4_5, (5, 5))
	layout.obstruct_coordinate((5, 5, 0))
	layout.obstruct_coordinate((5, 5, 1))
	gates[5][5] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_5_1 = layout.create_not(x_4_1, (5, 1))
	layout.obstruct_coordinate((5, 1, 0))
	layout.obstruct_coordinate((5, 1, 1))
	gates[5][1] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_5_3 = layout.create_and(x_0_1, x_5_1, (5, 3))
	layout.move_node(layout.get_node((5, 3)), (5, 3), [])
	if pyfiction.color_routing(layout, [((0, 1), (5, 3)), ((5, 1), (5, 3))], params):
		print("success")
	for fanin in layout.fanins((5, 3)):
		while fanin not in ((0, 1), (5, 1)):
			layout.obstruct_coordinate(fanin)
			fanin = layout.fanins(fanin)[0]
	layout.obstruct_coordinate((5, 3, 0))
	layout.obstruct_coordinate((5, 3, 1))
	gates[5][3] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_6_3 = layout.create_not(x_5_3, (6, 3))
	layout.obstruct_coordinate((6, 3, 0))
	layout.obstruct_coordinate((6, 3, 1))
	gates[6][3] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_6_5 = layout.create_and(x_6_3, x_5_5, (6, 5))
	layout.move_node(layout.get_node((6, 5)), (6, 5), [])
	if pyfiction.color_routing(layout, [((6, 3), (6, 5)), ((5, 5), (6, 5))], params):
		print("success")
	for fanin in layout.fanins((6, 5)):
		while fanin not in ((6, 3), (5, 5)):
			layout.obstruct_coordinate(fanin)
			fanin = layout.fanins(fanin)[0]
	layout.obstruct_coordinate((6, 5, 0))
	layout.obstruct_coordinate((6, 5, 1))
	gates[6][5] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_7_5 = layout.create_buf(x_6_5, (7, 5))
	layout.obstruct_coordinate((7, 5, 0))
	layout.obstruct_coordinate((7, 5, 1))
	gates[7][5] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_7_6 = layout.create_not(x_7_5, (7, 6))
	layout.obstruct_coordinate((7, 6, 0))
	layout.obstruct_coordinate((7, 6, 1))
	gates[7][6] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_0_4 = layout.create_pi("x3", (0, 4))
	layout.obstruct_coordinate((0, 4, 0))
	layout.obstruct_coordinate((0, 4, 1))
	gates[0][4] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_0_5 = layout.create_buf(x_0_4, (0, 5))
	layout.obstruct_coordinate((0, 5, 0))
	layout.obstruct_coordinate((0, 5, 1))
	gates[0][5] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_0_6 = layout.create_not(x_0_5, (0, 6))
	layout.obstruct_coordinate((0, 6, 0))
	layout.obstruct_coordinate((0, 6, 1))
	gates[0][6] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_0_7 = layout.create_pi("x4", (0, 7))
	layout.obstruct_coordinate((0, 7, 0))
	layout.obstruct_coordinate((0, 7, 1))
	gates[0][7] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_0_8 = layout.create_buf(x_0_7, (0, 8))
	layout.obstruct_coordinate((0, 8, 0))
	layout.obstruct_coordinate((0, 8, 1))
	gates[0][8] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_4_8 = layout.create_and(x_0_6, x_0_8, (4, 8))
	layout.move_node(layout.get_node((4, 8)), (4, 8), [])
	if pyfiction.color_routing(layout, [((0, 6), (4, 8)), ((0, 8), (4, 8))], params):
		print("success")
	for fanin in layout.fanins((4, 8)):
		while fanin not in ((0, 6), (0, 8)):
			layout.obstruct_coordinate(fanin)
			fanin = layout.fanins(fanin)[0]
	layout.obstruct_coordinate((4, 8, 0))
	layout.obstruct_coordinate((4, 8, 1))
	gates[4][8] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_4_9 = layout.create_not(x_4_8, (4, 9))
	layout.obstruct_coordinate((4, 9, 0))
	layout.obstruct_coordinate((4, 9, 1))
	gates[4][9] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_0_9 = layout.create_not(x_0_8, (0, 9))
	layout.obstruct_coordinate((0, 9, 0))
	layout.obstruct_coordinate((0, 9, 1))
	gates[0][9] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_3_10 = layout.create_and(x_0_5, x_0_9, (3, 10))
	layout.move_node(layout.get_node((3, 10)), (3, 10), [])
	if pyfiction.color_routing(layout, [((0, 5), (3, 10)), ((0, 9), (3, 10))], params):
		print("success")
	for fanin in layout.fanins((3, 10)):
		while fanin not in ((0, 5), (0, 9)):
			layout.obstruct_coordinate(fanin)
			fanin = layout.fanins(fanin)[0]
	layout.obstruct_coordinate((3, 10, 0))
	layout.obstruct_coordinate((3, 10, 1))
	gates[3][10] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_4_10 = layout.create_not(x_3_10, (4, 10))
	layout.obstruct_coordinate((4, 10, 0))
	layout.obstruct_coordinate((4, 10, 1))
	gates[4][10] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_7_11 = layout.create_and(x_4_10, x_4_9, (7, 11))
	layout.move_node(layout.get_node((7, 11)), (7, 11), [])
	if pyfiction.color_routing(layout, [((4, 10), (7, 11)), ((4, 9), (7, 11))], params):
		print("success")
	for fanin in layout.fanins((7, 11)):
		while fanin not in ((4, 10), (4, 9)):
			layout.obstruct_coordinate(fanin)
			fanin = layout.fanins(fanin)[0]
	layout.obstruct_coordinate((7, 11, 0))
	layout.obstruct_coordinate((7, 11, 1))
	gates[7][11] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_8_11 = layout.create_buf(x_7_11, (8, 11))
	layout.obstruct_coordinate((8, 11, 0))
	layout.obstruct_coordinate((8, 11, 1))
	gates[8][11] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_9_11 = layout.create_not(x_8_11, (9, 11))
	layout.obstruct_coordinate((9, 11, 0))
	layout.obstruct_coordinate((9, 11, 1))
	gates[9][11] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_11_11 = layout.create_and(x_9_11, x_7_5, (11, 11))
	layout.move_node(layout.get_node((11, 11)), (11, 11), [])
	if pyfiction.color_routing(layout, [((9, 11), (11, 11)), ((7, 5), (11, 11))], params):
		print("success")
	for fanin in layout.fanins((11, 11)):
		while fanin not in ((9, 11), (7, 5)):
			layout.obstruct_coordinate(fanin)
			fanin = layout.fanins(fanin)[0]
	layout.obstruct_coordinate((11, 11, 0))
	layout.obstruct_coordinate((11, 11, 1))
	gates[11][11] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	x_11_12 = layout.create_not(x_11_11, (11, 12))
	layout.obstruct_coordinate((11, 12, 0))
	layout.obstruct_coordinate((11, 12, 1))
	gates[11][12] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))

	print(layout)
	# Error occurs here
	x_8_12 = layout.create_and(x_8_11, x_7_6, (8, 12))
	layout.move_node(layout.get_node((8, 12)), (8, 12), [])
	if pyfiction.color_routing(layout, [((8, 11), (8, 12)), ((7, 6), (8, 12))], params):
		print("success")
	for fanin in layout.fanins((8, 12)):
		while fanin not in ((8, 11), (7, 6)):
			layout.obstruct_coordinate(fanin)
			fanin = layout.fanins(fanin)[0]
	layout.obstruct_coordinate((8, 12, 0))
	layout.obstruct_coordinate((8, 12, 1))
	gates[8][12] = 1
	for coordinate in list(zip(*np.where(gates == 1))):
		if not layout.is_obstructed_coordinate(coordinate + (1,)):
			layout.obstruct_coordinate(coordinate + (1,))
		if not layout.is_obstructed_coordinate(coordinate + (0,)):
			layout.obstruct_coordinate(coordinate + (0,))
	print(layout)
