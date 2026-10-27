"""P0 check 3 (+ the post-0 clip): clear-view pole placement, conditions, a second intersection.

    python check3.py [--out /f/out/check3] [--clip-frames 100]

1. Pole search at W 120th & Amsterdam: candidate poses around the junction, one semantic + depth
   frame each; a pose is clear when little of the frame is vegetation or anything closer than
   5 m, and enough of it is road.
2. Conditions on the best pose: day, golden, dusk, night, rain, rain at night (RGB + instance).
3. The post-0 clip: --clip-frames captures at 10 fps from the best pose: RGB, semantic, instance
   (+ labels), depth.
4. A second intersection (Broadway & W 96th): how long the tiles take to stream in, then one
   4-pose capture.
Writes PNG/JSON per frame and a summary.json.
"""

import argparse
import json
import math
import os
import queue
import random
import sys
import time

import numpy as np

sys.path.insert(0, "/f/src/PythonAPI")
import valdrada  # noqa: E402
from valdrada import Location, Rotation, Transform, WeatherParameters  # noqa: E402

DT = 0.05
SITES = {"amsterdam_w120": (40.80955, -73.95905), "broadway_w96": (40.79370, -73.97220)}
ROAD_USERS = ("car", "bus", "truck", "bicycle", "pedestrian")
CONDITIONS = [
    ("day", 0.0),
    ("golden", 0.0),
    ("dusk", 0.0),
    ("night", 0.0),
    ("day", 0.8),
    ("night", 0.8),
]


def candidates(centre):
    """Poles on 8 bearings, 3 distances, 2 heights, looking at the junction centre."""
    out = []
    for bearing in range(0, 360, 45):
        a = math.radians(bearing)
        for dist in (14.0, 20.0, 26.0):
            for height in (6.0, 8.0):
                loc = centre + Location(dist * math.cos(a), dist * math.sin(a), height)
                pitch = -math.degrees(math.atan2(height, dist)) + 6  # aim a little past the centre
                out.append(
                    {
                        "bearing": bearing,
                        "dist": dist,
                        "height": height,
                        "tf": Transform(loc, Rotation(pitch=pitch, yaw=bearing + 180)),
                    }
                )
    return out


class Rig:
    """Cameras of the given kinds at one pose, all capturing every `every` steps; moved together."""

    KIND = {
        "rgb": "sensor.camera.rgb",
        "semantic": "sensor.camera.semantic_segmentation",
        "instance": "sensor.camera.instance_segmentation",
        "depth": "sensor.camera.depth",
    }

    def __init__(self, world, lib, tf, kinds, every=1, amodal=8):
        self.world, self.kinds, self.cams, self.q = world, kinds, {}, {}
        for k in kinds:
            bp = lib.find(self.KIND[k])
            bp.set_attribute("fov", 70)
            bp.set_attribute("sensor_tick", every * DT if every > 1 else 0)
            if k == "instance":
                bp.set_attribute("amodal", amodal)
            cam = world.spawn_actor(bp, tf)
            self.q[k] = queue.Queue()
            cam.listen(self.q[k].put)
            self.cams[k] = cam

    def move(self, tf):
        for c in self.cams.values():
            c.set_transform(tf)

    def tick(self):
        """One step; the rig's data if it captured on it, else None."""
        frame = self.world.tick()
        if self.q[self.kinds[0]].empty():
            return frame, None
        data = {k: self.q[k].get() for k in self.kinds}
        assert all(d.frame == frame for d in data.values()), "rig out of sync"
        return frame, data

    def drain(self):
        for q in self.q.values():
            while not q.empty():
                q.get()

    def destroy(self):
        for c in self.cams.values():
            c.destroy()


def spawn_vehicles(world, m, lib, centre, n, rng):
    bps = [
        b for b in lib.filter("vehicle.*") if b.id not in ("vehicle.firetruck", "vehicle.ambulance")
    ]
    sps = m.get_spawn_points(center=centre, radius=120, spacing=16)
    rng.shuffle(sps)
    out = []
    for sp in sps[:n]:
        v = world.try_spawn_actor(rng.choice(bps), sp)
        if v is not None:
            v.set_autopilot(True)
            out.append(v)
    return out


def save(data, out, name):
    os.makedirs(out, exist_ok=True)
    for k, d in data.items():
        if k == "depth":
            d.save_to_disk(os.path.join(out, f"{name}_depth.png"))
        else:
            d.save_to_disk(os.path.join(out, f"{name}_{k}.png"))


def go_to(world, m, site):
    here = m.geolocation_to_location(*SITES[site])
    world.get_spectator().set_transform(Transform(here + Location(0, 0, 30)))
    t0 = time.perf_counter()
    info = world.wait_until_loaded()
    return m.get_junctions(center=here, radius=60)[0], time.perf_counter() - t0, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/f/out/check3")
    ap.add_argument("--clip-frames", type=int, default=100)
    a = ap.parse_args()
    rng = random.Random(5)
    client = valdrada.Client()
    client.set_timeout(300.0)
    world = client.get_world()
    world.apply_settings(valdrada.WorldSettings(synchronous_mode=True, fixed_delta_seconds=DT))
    m = world.get_map()
    lib = world.get_blueprint_library()
    classes = {c["id"]: c["name"] for c in world.get_semantic_classes()}
    print("semantic classes:", sorted(classes.values()), flush=True)
    veg = [i for i, n in classes.items() if n == "vegetation"]
    road = [i for i, n in classes.items() if n in ("road", "roadline", "road_line", "crosswalk")]
    summary = {"gpu": client.get_server_info().get("gpu"), "classes": classes}

    junction, load_s, info = go_to(world, m, "amsterdam_w120")
    summary["load_first"] = {"seconds": load_s, "info": info}
    actors = spawn_vehicles(world, m, lib, junction.location, 20, rng)
    rig = None
    try:
        for _ in range(int(6.0 / DT)):
            world.tick()

        # 1. pole search
        cands = candidates(junction.location)
        rig = Rig(world, lib, cands[0]["tf"], ["semantic", "depth"])
        for c in cands:
            rig.move(c["tf"])
            rig.drain()
            _, data = rig.tick()
            _, data = rig.tick()  # the first frame after a move may predate it
            sem = data["semantic"].to_numpy()
            dep = data["depth"].to_numpy()
            c["vegetation"] = float(np.isin(sem, veg).mean())
            c["near"] = float((dep < 5.0).mean())
            c["road"] = float(np.isin(sem, road).mean())
            c["clear"] = c["vegetation"] < 0.10 and c["near"] < 0.05 and c["road"] > 0.20
        rig.destroy()
        rig = None
        table = [{k: v for k, v in c.items() if k != "tf"} for c in cands]
        summary["pole_search"] = table
        for row in table:
            print(
                f"bearing {row['bearing']:3d} dist {row['dist']:4.0f} h {row['height']:.0f}: "
                f"veg {row['vegetation']:.2f} "
                f"near {row['near']:.2f} road {row['road']:.2f} {'CLEAR' if row['clear'] else ''}",
                flush=True,
            )
        clear = [c for c in cands if c["clear"]]
        clear.sort(key=lambda c: (c["vegetation"] + c["near"]) - 0.2 * c["road"])
        picked, bearings = [], set()
        for c in clear:  # one pose per bearing, best first
            if c["bearing"] not in bearings:
                picked.append(c)
                bearings.add(c["bearing"])
        summary["picked"] = [{k: v for k, v in c.items() if k != "tf"} for c in picked[:4]]
        if not picked:
            raise SystemExit("no clear pose found")
        best = picked[0]["tf"]

        # 2. conditions
        rig = Rig(world, lib, best, ["rgb", "instance"])
        cond = []
        for tod, rain in CONDITIONS:
            world.set_weather(WeatherParameters(tod, rain))
            for _ in range(20):
                world.tick()
            rig.drain()
            frame, data = rig.tick()
            name = f"{tod}_rain{rain:.1f}"
            save(data, os.path.join(a.out, "conditions"), name)
            users = data["instance"].filter(classes=ROAD_USERS, min_area=100)
            mean_rgb = float(data["rgb"].to_numpy()[..., :3].mean())
            cond.append(
                {
                    "condition": name,
                    "frame": frame,
                    "road_users": len(users),
                    "mean_rgb": mean_rgb,
                    "weather_back": repr(world.get_weather()),
                }
            )
            print(
                f"{name}: {len(users)} road users, mean RGB {mean_rgb:.1f}, {world.get_weather()}",
                flush=True,
            )
        summary["conditions"] = cond
        world.set_weather(WeatherParameters("day", 0.0))
        rig.destroy()
        rig = None

        # 3. the post-0 clip, 10 fps
        rig = Rig(world, lib, best, ["rgb", "semantic", "instance", "depth"], every=2)
        rig.drain()
        got, t0 = 0, time.perf_counter()
        while got < a.clip_frames:
            frame, data = rig.tick()
            if data is None:
                continue
            save(data, os.path.join(a.out, "clip"), f"{got:04d}")
            got += 1
        summary["clip"] = {"frames": got, "fps": 1 / (2 * DT), "wall_s": time.perf_counter() - t0}
        print(f"clip: {got} frames in {summary['clip']['wall_s']:.0f} s", flush=True)
        rig.destroy()
        rig = None
    finally:
        if rig:
            rig.destroy()
        for x in reversed(actors):
            x.destroy()

    # 4. a second intersection
    junction, load_s, info = go_to(world, m, "broadway_w96")
    summary["load_second"] = {"seconds": load_s, "info": info, "junction": repr(junction)}
    print(f"second intersection loaded in {load_s:.1f} s: {junction}", flush=True)
    actors = spawn_vehicles(world, m, lib, junction.location, 20, rng)
    rigs = []
    try:
        for _ in range(int(6.0 / DT)):
            world.tick()
        for k in range(4):
            b = 45 + 90 * k
            r = math.radians(b)
            tf = Transform(
                junction.location + Location(20 * math.cos(r), 20 * math.sin(r), 7),
                Rotation(pitch=-15, yaw=b + 180),
            )
            rigs.append(Rig(world, lib, tf, ["rgb", "instance"]))
        for _ in range(2):
            world.tick()
        for rg in rigs:
            rg.drain()
        frame = world.tick()
        for k, rg in enumerate(rigs):
            data = {kk: rg.q[kk].get() for kk in rg.kinds}
            save(data, os.path.join(a.out, "second"), f"pole{k}")
    finally:
        for rg in rigs:
            rg.destroy()
        for x in reversed(actors):
            x.destroy()
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print("ok", flush=True)


if __name__ == "__main__":
    main()
