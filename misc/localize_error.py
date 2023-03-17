from fiction import pyfiction
import numpy as np


if __name__ == "__main__":
	layout = pyfiction.cartesian_obstruction_layout(pyfiction.cartesian_gate_layout((11, 12, 1), "2DDWave"))
	params = pyfiction.color_routing_params()
	params.crossings = True
	params.path_limit = 50
	params.engine = pyfiction.graph_coloring_engine.MCS
	gates = np.zeros([12, 13], dtype=int)

	x = layout.create_pi("x1", (0, 0))
	layout.obstruct_coordinate((0, 0, 0))
	layout.obstruct_coordinate((0, 0, 1))

	para = pyfiction.a_star_params()
	para.crossings = True
	print(pyfiction.a_star(layout, (0, 0), (11, 12), para))