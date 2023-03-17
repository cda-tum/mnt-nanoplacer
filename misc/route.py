from fiction import pyfiction

if __name__ == "__main__":
	lyt = pyfiction.cartesian_gate_layout((10, 10, 1), "2DDWave")

	x1 = lyt.create_pi("x1", (0, 0))
	f1 = lyt.create_po(x1, "f1", (10, 10))

	path = pyfiction.a_star(lyt, (0, 0), (10, 10))

	pyfiction.route_path(lyt, path)
	print(lyt)
	print(pyfiction.gate_level_drvs(lyt, print_report=True))
