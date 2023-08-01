from time import time

from fiction import pyfiction

params = pyfiction.a_star_params()
params.crossings = True
drv_params = pyfiction.gate_level_drv_params()


def optimize_post_placement(layout):
    width = layout.x()
    height = layout.y()
    gates = []
    start = time()
    for x in range(width + 1):
        for y in range(height + 1):
            if (
                layout.is_inv(layout.get_node((x, y)))
                or layout.is_and(layout.get_node((x, y)))
                or layout.is_xor(layout.get_node((x, y)))
                or layout.is_fanout(layout.get_node((x, y)))
                or layout.is_or(layout.get_node((x, y)))
                or layout.is_pi_tile((x, y))
            ):
                gates.append((x, y))
                layout.obstruct_coordinate((x, y, 1))
    gates = sorted(gates, key=lambda x: (sum(x), x[1]))
    current_step = 0
    moved = 1
    cur = time()
    for h in range(20):
        if moved != 0:
            moved = 0
        else:
            break
        for gate_id in range(len(gates)):
            moved_gate, new_pos = move_gate(gates[gate_id], layout, width, height)
            if moved_gate:
                moved += 1
                gates[gate_id] = new_pos

            current_step += 1

        print(f"Step: {h + 1}, Moved Gates: {moved}")
        gates = sorted(gates, key=lambda x: (sum(x), x[1]))

    delete_wires(layout, width, height)
    optimize_output(layout)
    fix_dead_nodes(layout, gates)
    print(f"{time() - cur}s")
    min_coord, max_coord = layout.bounding_box_2d()
    layout.resize((max_coord.x, max_coord.y, 1))
    print(layout)
    print(time() - start)

    size_before = (width + 1) * (height + 1)
    size_after = (max_coord.x + 1) * (max_coord.y + 1)
    improv = (size_before - size_after) / size_before * 100
    print(pyfiction.gate_level_drvs(layout, drv_params, True))
    print(f"Before: {size_before}, After: {size_after}, Improv: -{improv:.2f}%")
    print(max_coord.x + 1, max_coord.y + 1)


def optimize_output(lyt):
    pos = lyt.pos()
    paths = []
    for po in pos:
        route = []
        for fin in lyt.fanins(po):
            fanin = fin
            route.insert(0, po)
            route.insert(0, fanin)

            while (
                lyt.is_wire_tile(fanin)
                and lyt.fanout_size(lyt.get_node(fanin)) != 2
                and lyt.fanin_size(lyt.get_node(fanin)) != 0
            ):
                fanin = lyt.fanins(fanin)[0]
                route.insert(0, fanin)
        paths.append(route)
    min_x = max([route[1].x for route in paths])
    min_y = max([route[1].y for route in paths])
    updates = []
    for route in paths:
        dangling = None
        new_pos = None
        moved = False
        for tile in route:
            if tile.x < min_x and tile.y < min_y:
                dangling = tile
            elif not lyt.is_po_tile(tile) and lyt.is_wire_tile(tile) and lyt.fanout_size(lyt.get_node(tile)) != 2:
                lyt.clear_tile(tile)
                if not new_pos:
                    new_pos = tile
            elif dangling:
                if new_pos:
                    updates.append((tile, (new_pos.x, new_pos.y, 0), dangling))
                moved = True
        if not moved:
            lyt.move_node(lyt.get_node(route[-1]), route[1], [lyt.make_signal(lyt.get_node(route[0]))])
    for update in updates:
        tile, new_pos, dangling = update
        lyt.move_node(lyt.get_node(tile), new_pos, [lyt.make_signal(lyt.get_node(dangling))])


def fix_dead_nodes(lyt, gt):
    empty_pos = None
    for coord in lyt.coordinates():
        if lyt.is_empty_tile(coord):
            empty_pos = coord
            break
    if empty_pos:
        for gate in gt:
            if lyt.is_dead(lyt.get_node(gate)):
                ins = lyt.fanins(gate)
                lyt.move_node(lyt.get_node(gate), empty_pos)
                lyt.clear_tile(gate)
                lyt.move_node(lyt.get_node(empty_pos), gate, [lyt.make_signal(lyt.get_node(inss)) for inss in ins])
    else:
        raise Exception


def check_wires(lyt, tc):
    moved_tiles = []
    for tile in tc:
        if lyt.is_empty_tile((tile.x, tile.y, 0)) and lyt.is_wire_tile((tile.x, tile.y, 1)):
            fint = lyt.fanins((tile.x, tile.y, 1))[0]
            foutt = lyt.fanouts((tile.x, tile.y, 1))[0]

            lyt.move_node(lyt.get_node((tile.x, tile.y, 1)), (tile.x, tile.y, 0), [lyt.make_signal(lyt.get_node(fint))])

            if lyt.fanins(foutt) and (lyt.fanins(foutt)[0] not in tc or lyt.fanins(foutt)[0] in moved_tiles):
                lyt.move_node(
                    lyt.get_node(foutt),
                    foutt,
                    [
                        lyt.make_signal(lyt.get_node((tile.x, tile.y, 0))),
                        lyt.make_signal(lyt.get_node(lyt.fanins(foutt)[0])),
                    ],
                )
            else:
                lyt.move_node(lyt.get_node(foutt), foutt, [lyt.make_signal(lyt.get_node((tile.x, tile.y, 0)))])
            lyt.obstruct_coordinate((tile.x, tile.y, 0))
            lyt.clear_obstructed_coordinate((tile.x, tile.y, 1))
            moved_tiles.append(tile)
    return moved_tiles


def get_fanin_and_fanouts(lyt, op):
    fanin1 = None
    fanin2 = None
    fanins = []
    fanout1 = None
    fanout2 = None
    fanouts = []
    to_clear = []
    route1 = []
    route2 = []
    route3 = []
    route4 = []

    for num, fin in enumerate(set(lyt.fanins(op))):
        fanin = fin
        if num == 0:
            route1.insert(0, op)
            route1.insert(0, fanin)
        else:
            route2.insert(0, op)
            route2.insert(0, fanin)
        while (
            lyt.is_wire_tile(fanin)
            and lyt.fanout_size(lyt.get_node(fanin)) != 2
            and lyt.fanin_size(lyt.get_node(fanin)) != 0
        ):
            to_clear.append(fanin)
            fanin = lyt.fanins(fanin)[0]
            if num == 0:
                route1.insert(0, fanin)
            else:
                route2.insert(0, fanin)
        if num == 0:
            fanin1 = fanin
        else:
            fanin2 = fanin

    for num, fout in enumerate(set(lyt.fanouts(op))):
        fanout = fout
        if num == 0:
            route3.append(op)
            route3.append(fanout)
        else:
            route4.append(op)
            route4.append(fanout)
        while lyt.is_wire_tile(fanout) and lyt.fanout_size(lyt.get_node(fanout)) not in (0, 2):
            to_clear.append(fanout)
            fanout = lyt.fanouts(fanout)[0]
            if num == 0:
                route3.append(fanout)
            else:
                route4.append(fanout)
        if num == 0:
            fanout1 = fanout
        else:
            fanout2 = fanout

    if fanin1:
        fanins.append(fanin1)
    if fanin2:
        fanins.append(fanin2)

    if fanout1:
        fanouts.append(fanout1)
    if fanout2:
        fanouts.append(fanout2)

    return fanins, fanouts, to_clear, route1, route2, route3, route4


def move_gate(old_pos, lyt, width, height):
    new_pos = (0, 0)
    fanins, fanouts, to_clear, r1, r2, r3, r4 = get_fanin_and_fanouts(lyt, old_pos)
    min_x = max([fanin.x for fanin in fanins]) if fanins else 0
    min_y = max([fanin.y for fanin in fanins]) if fanins else 0
    max_x = old_pos[0]
    max_y = old_pos[1]
    max_diagonal = max_x + max_y
    for fanin in fanins:
        if fanin in lyt.fanins(old_pos):
            return False, new_pos

    for tile in to_clear:
        lyt.clear_tile(tile)
        lyt.clear_obstructed_coordinate(tile)

    for fanout in fanouts:
        fins = [lyt.make_signal(lyt.get_node(fout)) for fout in lyt.fanins(fanout) if fout != old_pos]

        lyt.move_node(
            lyt.get_node(fanout),
            fanout,
            fins,
        )
    lyt.move_node(lyt.get_node(old_pos), old_pos, [])
    check_wires(lyt, to_clear)
    success = False
    current_pos = old_pos
    optimized = False

    for k in range(width + height + 1):
        for x in range(k + 1):
            y = k - x

            if success:
                break

            if (
                height >= y >= min_y
                and width >= x >= min_x
                and (x + y) <= max_diagonal
                and (not lyt.is_pi_tile(current_pos) or (lyt.is_pi_tile(current_pos) and (x == 0 or y == 0)))
            ):
                new_pos = (x, y)

                if lyt.is_empty_tile(new_pos) and lyt.is_empty_tile((*new_pos, 1)):
                    lyt.move_node(lyt.get_node(current_pos), new_pos, [])
                    lyt.obstruct_coordinate(new_pos)
                    lyt.obstruct_coordinate((*new_pos, 1))
                    lyt.clear_obstructed_coordinate(current_pos)
                    lyt.clear_obstructed_coordinate((*current_pos, 1))
                    if len(fanins) > 0:
                        path1 = pyfiction.a_star(lyt, fanins[0], new_pos, params)
                        for tile in path1:
                            lyt.obstruct_coordinate(tile)
                    else:
                        path1 = True

                    if len(fanins) == 2:
                        path2 = pyfiction.a_star(lyt, fanins[1], new_pos, params)
                        for tile in path2:
                            lyt.obstruct_coordinate(tile)
                    else:
                        path2 = True
                    if fanouts:
                        path3 = pyfiction.a_star(lyt, new_pos, fanouts[0], params)
                        for tile in path3:
                            lyt.obstruct_coordinate(tile)
                    else:
                        path3 = True

                    if len(fanouts) == 2:
                        path4 = pyfiction.a_star(lyt, new_pos, fanouts[1], params)
                        for tile in path4:
                            lyt.obstruct_coordinate(tile)
                    else:
                        path4 = True

                    if path1 and path2 and path3 and path4:
                        for path in [path1, path2, path3, path4]:
                            if type(path) == list:
                                pyfiction.route_path(lyt, path)
                                for tile in path:
                                    lyt.obstruct_coordinate(tile)

                        success = True
                        if new_pos != old_pos:
                            optimized = True

                        if len(fanins) == 2:
                            lyt.move_node(
                                lyt.get_node(new_pos),
                                new_pos,
                                [
                                    lyt.make_signal(lyt.get_node(path1[-2])),
                                    lyt.make_signal(lyt.get_node(path2[-2])),
                                ],
                            )
                        elif len(fanins) == 1:
                            lyt.move_node(lyt.get_node(new_pos), new_pos, [lyt.make_signal(lyt.get_node(path1[-2]))])
                        else:
                            pass

                        for fanout in fanouts:
                            lyt.move_node(
                                lyt.get_node(fanout),
                                fanout,
                                [lyt.make_signal(lyt.get_node(fout)) for fout in lyt.fanins(fanout)],
                            )

                    else:
                        for path in [path1, path2, path3, path4]:
                            if type(path) == list:
                                for tile in path:
                                    lyt.clear_obstructed_coordinate(tile)
                    current_pos = new_pos
        else:
            continue
        break
    if not success:
        lyt.move_node(lyt.get_node(current_pos), old_pos, [])

        for r in [r1, r2, r3, r4]:
            if r:
                pyfiction.route_path(lyt, r)
        for tile in r1 + r2 + r3 + r4:
            lyt.obstruct_coordinate(tile)
        lyt.clear_obstructed_coordinate(current_pos)
        lyt.clear_obstructed_coordinate((*current_pos, 1))
        lyt.obstruct_coordinate(old_pos)
        lyt.obstruct_coordinate((*old_pos, 1))

        lyt.move_node(
            lyt.get_node(old_pos),
            old_pos,
            [lyt.make_signal(lyt.get_node(fanin)) for fanin in lyt.fanins(old_pos)],
        )

        for fanout in fanouts:
            lyt.move_node(
                lyt.get_node(fanout),
                fanout,
                [lyt.make_signal(lyt.get_node(fout)) for fout in lyt.fanins(fanout)],
            )
    return optimized, new_pos


def delete_wires(lyt, width, height):
    for y in reversed(range(height + 1)):
        found_row = True
        for x in reversed(range(width + 1)):
            if (
                lyt.is_wire_tile((x, y))
                and lyt.fanin_size(lyt.get_node((x, y))) == 1
                and lyt.fanout_size(lyt.get_node((x, y))) == 1
                and lyt.has_northern_incoming_signal((x, y))
                and lyt.has_southern_outgoing_signal((x, y))
            ) or lyt.is_empty_tile((x, y)):
                pass
            else:
                found_row = False
        if found_row:
            print(f"Row {y} can be deleted")
            delete_row(lyt, y, width, height)

    for x in reversed(range(width + 1)):
        found_column = True
        for y in reversed(range(height + 1)):
            if (
                lyt.is_wire_tile((x, y))
                and lyt.fanin_size(lyt.get_node((x, y))) == 1
                and lyt.fanout_size(lyt.get_node((x, y))) == 1
                and lyt.has_western_incoming_signal((x, y))
                and lyt.has_eastern_outgoing_signal((x, y))
            ) or lyt.is_empty_tile((x, y)):
                pass
            else:
                found_column = False
        if found_column:
            print(f"Column {x} can be deleted")
            delete_column(lyt, x, width, height)


def delete_row(lyt, row_idx, width, height):
    fanins = {}
    for y in range(row_idx, height + 1):
        fanins[y] = {}
        for x in range(width + 1):
            fanins[y][x] = {}
            for z in range(2):
                if y == row_idx and lyt.fanins((x, y, z)):
                    fanin_row = lyt.fanins((x, y, z))[0]
                    fanin_next_row = lyt.fanins((x, y + 1, z))
                    fanins[y][x][z] = []
                    for fanin in fanin_next_row:
                        if fanin.y == y:
                            fanins[y][x][z].append(lyt.coord(fanin_row.x, fanin_row.y + 1, fanin_row.z))
                        else:
                            fanins[y][x][z].append(fanin)
                else:
                    fanins[y][x][z] = lyt.fanins((x, y + 1, z))
        for x in range(width + 1):
            for z in range(2):
                old_pos = (x, y, z)
                if not lyt.is_empty_tile(old_pos):
                    if y == row_idx:
                        lyt.clear_tile(old_pos)
                    else:
                        new_pos = (x, y - 1, z)
                        lyt.move_node(
                            lyt.get_node(old_pos),
                            new_pos,
                            [
                                lyt.make_signal(lyt.get_node((fanin.x, fanin.y - 1, fanin.z)))
                                for fanin in fanins[y - 1][x][z]
                            ],
                        )


def delete_column(lyt, column_idx, width, height):
    fanins = {}
    for x in range(column_idx, width + 1):
        fanins[x] = {}
        for y in range(height + 1):
            fanins[x][y] = {}
            for z in range(2):
                if x == column_idx and lyt.fanins((x, y, z)):
                    fanin_column = lyt.fanins((x, y, z))[0]
                    fanin_next_column = lyt.fanins((x + 1, y, z))
                    fanins[x][y][z] = []
                    for fanin in fanin_next_column:
                        if fanin.x == x:
                            fanins[x][y][z].append(lyt.coord(fanin_column.x + 1, fanin_column.y, fanin_column.z))
                        else:
                            fanins[x][y][z].append(fanin)
                else:
                    fanins[x][y][z] = lyt.fanins((x + 1, y, z))
        for y in range(height + 1):
            for z in range(2):
                old_pos = (x, y, z)
                if not lyt.is_empty_tile(old_pos):
                    if x == column_idx:
                        lyt.clear_tile(old_pos)
                    else:
                        new_pos = (x - 1, y, z)
                        lyt.move_node(
                            lyt.get_node(old_pos),
                            new_pos,
                            [
                                lyt.make_signal(lyt.get_node((fanin.x - 1, fanin.y, fanin.z)))
                                for fanin in fanins[x - 1][y][z]
                            ],
                        )
