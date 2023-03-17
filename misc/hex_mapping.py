import math

w = 3
h = 3
test_coords = [(x, y) for y in range(h) for x in range(w)]


def to_hex(old_coords):
	new_coords = []
	for coord in old_coords:
		old_x, old_y = coord
		y = old_x + old_y
		x = old_x + math.floor(math.floor(h / 2) - y / 2)
		new_coords.append((x, y))
	return new_coords


if __name__ == "__main__":
	hex_coords = to_hex(test_coords)
	print(hex_coords)
