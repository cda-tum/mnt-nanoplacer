from fiction import pyfiction


if __name__ == "__main__":
	layout = pyfiction.cartesian_obstruction_layout(pyfiction.cartesian_gate_layout((3, 6, 1), "2DDWave"))
	params = pyfiction.color_routing_params()
	params.crossings = True
	params.path_limit = 50
	params.engine = pyfiction.graph_coloring_engine.MCS

	x1 = layout.create_pi("x1", (0, 0))
	layout.obstruct_coordinate((0, 0, 0))
	layout.obstruct_coordinate((0, 0, 1))
	print(f"(0, 0, 0) obstructed: {layout.is_obstructed_coordinate((0, 0, 0))}")
	print(f"(0, 0, 1) obstructed: {layout.is_obstructed_coordinate((0, 0, 1))}")

	w1 = layout.create_buf(x1, (1, 0))
	layout.obstruct_coordinate((1, 0, 0))
	layout.obstruct_coordinate((1, 0, 1))

	x2 = layout.create_pi("x2", (0, 1))
	layout.obstruct_coordinate((0, 1, 0))
	layout.obstruct_coordinate((0, 1, 1))

	w2 = layout.create_buf(x2, (1, 1))
	layout.obstruct_coordinate((1, 1, 0))
	layout.obstruct_coordinate((1, 1, 1))

	and_gate = layout.create_and(w1, w2, (2, 1))
	layout.move_node(layout.get_node((2, 1)), (2, 1), [])
	if pyfiction.color_routing(layout, [((1, 0), (2, 1)), ((1, 1), (2, 1))], params):
		print("success")

	print(f"(0, 0, 0) obstructed: {layout.is_obstructed_coordinate((0, 0, 0))}")
	print(f"(0, 0, 1) obstructed: {layout.is_obstructed_coordinate((0, 0, 1))}")
	layout.obstruct_coordinate((2, 1, 0))
	layout.obstruct_coordinate((2, 1, 1))

	not_gate = layout.create_not(and_gate, (2, 2))
	layout.obstruct_coordinate((2, 2, 0))
	layout.obstruct_coordinate((2, 2, 1))

	or_gate = layout.create_or(w1, w2, (1, 2))
	layout.move_node(layout.get_node((1, 2)), (1, 2), [])
	if pyfiction.color_routing(layout, [((1, 0), (1, 2)), ((1, 1), (1, 2))], params):
		print("success")

	print(layout)
