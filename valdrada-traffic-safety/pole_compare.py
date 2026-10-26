"""The camera inside a tree, and the pose the pole search moves it to (Broadway & W 96th).

    python pole_compare.py [--out /f/out/pole_compare]

check3.py placed four poles at Broadway & W 96th by fixed geometry (20 m out, 7 m up, pitch -15)
and the one on bearing 225 landed inside a street tree. Here the same fixed pose is scored with
the pole search's three numbers (vegetation, closer than 5 m, road), the 48 search candidates are
scored at this junction, and the best clear pose on the same bearing (the same corner) is kept,
or the best clear pose overall if that corner has none, or the best-ranked pose if none is clear.
Both poses then capture RGB + instance labels on the same simulation step. Writes
<out>/fixed_*.png|json, <out>/search_*.png|json, <out>/pole_search.json and <out>/pole_compare.json.
"""

import argparse
import json
import math
import os
import random

import numpy as np
from check3 import (
    DT,
    ROAD_USERS,
    Location,
    Rig,
    Rotation,
    Transform,
    candidates,
    go_to,
    save,
    spawn_vehicles,
    valdrada,
)

FIXED_BEARING = 225  # the bottom-left pole of check3's Broadway capture (k = 2)


def fixed_pose(centre):
    r = math.radians(FIXED_BEARING)
    loc = centre + Location(20 * math.cos(r), 20 * math.sin(r), 7)
    return {
        "bearing": FIXED_BEARING,
        "dist": 20.0,
        "height": 7.0,
        "pitch": -15.0,
        "tf": Transform(loc, Rotation(pitch=-15, yaw=FIXED_BEARING + 180)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/f/out/pole_compare")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(5)
    client = valdrada.Client()
    client.set_timeout(300.0)
    world = client.get_world()
    world.apply_settings(valdrada.WorldSettings(synchronous_mode=True, fixed_delta_seconds=DT))
    m = world.get_map()
    lib = world.get_blueprint_library()
    classes = {c["id"]: c["name"] for c in world.get_semantic_classes()}
    veg = [i for i, n in classes.items() if n == "vegetation"]
    road = [i for i, n in classes.items() if n in ("road", "roadline", "road_line", "crosswalk")]

    junction, load_s, _ = go_to(world, m, "broadway_w96")
    print(f"loaded in {load_s:.1f} s: {junction}", flush=True)
    actors = spawn_vehicles(world, m, lib, junction.location, 20, rng)
    rig = None
    rigs = []
    try:
        for _ in range(int(6.0 / DT)):
            world.tick()

        fixed = fixed_pose(junction.location)
        cands = [fixed] + candidates(junction.location)
        rig = Rig(world, lib, cands[0]["tf"], ["semantic", "depth"])
        for c in cands:
            rig.move(c["tf"])
            rig.drain()
            rig.tick()
            _, data = rig.tick()  # the first frame after a move may predate it
            sem = data["semantic"].to_numpy()
            dep = data["depth"].to_numpy()
            c["vegetation"] = float(np.isin(sem, veg).mean())
            c["near"] = float((dep < 5.0).mean())
            c["road"] = float(np.isin(sem, road).mean())
            c["clear"] = c["vegetation"] < 0.10 and c["near"] < 0.05 and c["road"] > 0.20
        rig.destroy()
        rig = None

        search = cands[1:]
        strip = lambda c: {k: v for k, v in c.items() if k != "tf"}  # noqa: E731
        with open(os.path.join(a.out, "pole_search.json"), "w") as f:
            json.dump([strip(c) for c in cands], f, indent=1)
        for c in cands:
            print(
                f"bearing {c['bearing']:3d} dist {c['dist']:4.0f} h {c['height']:.0f}: "
                f"veg {c['vegetation']:.2f} near {c['near']:.2f} road {c['road']:.2f} "
                f"{'CLEAR' if c['clear'] else ''}",
                flush=True,
            )
        clear = [c for c in search if c["clear"]]
        clear.sort(key=lambda c: (c["vegetation"] + c["near"]) - 0.2 * c["road"])
        same_corner = [c for c in clear if c["bearing"] == FIXED_BEARING]
        # no pose may pass at a tree-lined junction: then show the one the search ranks first
        ranked = sorted(search, key=lambda c: (c["vegetation"] + c["near"]) - 0.2 * c["road"])
        best = (same_corner or clear or ranked)[0]
        print(
            f"clear {len(clear)}/{len(search)}; picked bearing {best['bearing']} "
            f"dist {best['dist']:.0f} h {best['height']:.0f}",
            flush=True,
        )

        # both poses capture on the same step, so the traffic is the same
        rigs = [Rig(world, lib, p["tf"], ["rgb", "instance"]) for p in (fixed, best)]
        for _ in range(2):
            world.tick()
        for rg in rigs:
            rg.drain()
        frame = world.tick()
        users = {}
        for name, rg in zip(("fixed", "search"), rigs, strict=True):
            data = {k: rg.q[k].get() for k in rg.kinds}
            assert all(d.frame == frame for d in data.values()), "rig out of sync"
            save(data, a.out, name)
            users[name] = len(data["instance"].filter(classes=ROAD_USERS, min_area=100))
    finally:
        if rig:
            rig.destroy()
        for rg in rigs:
            rg.destroy()
        for x in reversed(actors):
            x.destroy()

    summary = {
        "site": "broadway_w96",
        "load_s": load_s,
        "frame": frame,
        "fixed": {**strip(fixed), "road_users": users["fixed"]},
        "search_pick": {
            **strip(best),
            "road_users": users["search"],
            "same_bearing": best["bearing"] == FIXED_BEARING,
        },
        "search_clear": f"{len(clear)}/{len(search)}",
        "search": [strip(c) for c in search],
    }
    with open(os.path.join(a.out, "pole_compare.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps({k: summary[k] for k in ("fixed", "search_pick", "search_clear")}, indent=1))
    print("ok", flush=True)


if __name__ == "__main__":
    main()
