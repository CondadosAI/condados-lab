"""P0 check 5: seconds per labelled frame with P pole cameras capturing on the same tick.

    python budget.py [--poses 1,2,4] [--captures 12] [--every 20] [--kinds rgb,instance]
                     [--out /f/out/budget]

Each pose gets one camera per kind, all with sensor_tick = every * DT, so they render on the same
step and the idle steps in between cost only the world update. Prints, per P: wall seconds per
capture, per labelled frame, and where the tick milliseconds went (World.last_tick timing), split
into idle and capture steps.
"""

import argparse
import json
import math
import os
import queue
import random
import sys
import time

sys.path.insert(0, "/f/src/PythonAPI")
import valdrada  # noqa: E402
from valdrada import Location, Rotation, Transform  # noqa: E402

AMSTERDAM_120 = (40.80955, -73.95905)
DT = 0.05
KIND = {
    "rgb": "sensor.camera.rgb",
    "instance": "sensor.camera.instance_segmentation",
    "semantic": "sensor.camera.semantic_segmentation",
    "depth": "sensor.camera.depth",
}
ROAD_USERS = ("car", "bus", "truck", "bicycle", "pedestrian")


def pole_poses(centre, n, dist=20.0, height=7.0, pitch=-18.0):
    """n poles on the junction's diagonals, each looking at its centre."""
    out = []
    for k in range(n):
        a = math.radians(225 + 90 * k)
        loc = centre + Location(dist * math.cos(a), dist * math.sin(a), height)
        out.append(Transform(loc, Rotation(pitch=pitch, yaw=math.degrees(a) + 180)))
    return out


def spawn_traffic(world, m, lib, junction, n_vehicles, rng):
    actors = []
    bps = [
        b for b in lib.filter("vehicle.*") if b.id not in ("vehicle.firetruck", "vehicle.ambulance")
    ]
    sps = m.get_spawn_points(center=junction.location, radius=120, spacing=16)
    rng.shuffle(sps)
    for sp in sps[:n_vehicles]:
        v = world.try_spawn_actor(rng.choice(bps), sp)
        if v is not None:
            v.set_autopilot(True)
            actors.append(v)
    return actors


def timing_ms(tick):
    t = (tick or {}).get("timing") or {}
    return t


def run(world, lib, poses, kinds, captures, every, out):
    sensors, queues = [], {}
    for p, pose in enumerate(poses):
        for kind in kinds:
            bp = lib.find(KIND[kind])
            bp.set_attribute("fov", 70)
            bp.set_attribute("sensor_tick", every * DT)
            if kind == "instance":
                bp.set_attribute("amodal", 8)
            cam = world.spawn_actor(bp, pose)
            q = queue.Queue()
            cam.listen(q.put)
            queues[(p, kind)] = q
            sensors.append(cam)
    os.makedirs(out, exist_ok=True)
    idle_s, cap_s, save_s, labels, timings = [], [], [], 0, {"idle": [], "capture": []}
    got = 0
    try:
        for _ in range(every):  # warm-up: shaders, first readbacks
            world.tick()
            for q in queues.values():
                while not q.empty():
                    q.get()
        t_all = time.perf_counter()
        while got < captures:
            t0 = time.perf_counter()
            frame = world.tick()
            dt = time.perf_counter() - t0
            due = not queues[(0, kinds[0])].empty()
            (cap_s if due else idle_s).append(dt)
            timings["capture" if due else "idle"].append(timing_ms(world.last_tick))
            if not due:
                continue
            t1 = time.perf_counter()
            for p in range(len(poses)):
                data = {k: queues[(p, k)].get() for k in kinds}
                assert all(d.frame == frame for d in data.values())
                for k, d in data.items():
                    d.save_to_disk(os.path.join(out, f"p{p}_{frame:06d}_{k}.png"))
                if "instance" in data:
                    labels += len(data["instance"].filter(classes=ROAD_USERS, min_area=100))
            save_s.append(time.perf_counter() - t1)
            got += 1
        wall = time.perf_counter() - t_all
    finally:
        for s in sensors:
            s.destroy()
    n = len(poses)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    res = {
        "poses": n,
        "kinds": kinds,
        "every": every,
        "captures": captures,
        "wall_s_per_capture": wall / captures,
        "wall_s_per_frame": wall / (captures * n),
        "idle_tick_s": mean(idle_s),
        "capture_tick_s": mean(cap_s),
        "save_s_per_capture": mean(save_s),
        "labels_per_frame": labels / (captures * n),
        "timing_idle_example": timings["idle"][-1] if timings["idle"] else None,
        "timing_capture_example": timings["capture"][-1] if timings["capture"] else None,
    }
    print(
        f"P={n}: {res['wall_s_per_capture']:.2f} s/capture, {res['wall_s_per_frame']:.2f} s/frame "
        f"(idle tick {res['idle_tick_s']:.3f} s, capture tick {res['capture_tick_s']:.3f} s, "
        f"save {res['save_s_per_capture']:.2f} s), {res['labels_per_frame']:.1f} road users/frame",
        flush=True,
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", default="1,2,4")
    ap.add_argument("--captures", type=int, default=12)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--kinds", default="rgb,instance")
    ap.add_argument("--vehicles", type=int, default=20)
    ap.add_argument("--out", default="/f/out/budget")
    a = ap.parse_args()
    kinds = a.kinds.split(",")
    client = valdrada.Client()
    client.set_timeout(120.0)
    world = client.get_world()
    info = client.get_server_info()
    print(f"server on {info.get('gpu')}", flush=True)
    world.apply_settings(valdrada.WorldSettings(synchronous_mode=True, fixed_delta_seconds=DT))
    m = world.get_map()
    here = m.geolocation_to_location(*AMSTERDAM_120)
    world.get_spectator().set_transform(Transform(here + Location(0, 0, 30)))
    world.wait_until_loaded()
    junction = m.get_junctions(center=here, radius=60)[0]
    lib = world.get_blueprint_library()
    actors = spawn_traffic(world, m, lib, junction, a.vehicles, random.Random(3))
    results = []
    try:
        for _ in range(int(6.0 / DT)):
            world.tick()
        for n in (int(x) for x in a.poses.split(",")):
            results.append(
                run(
                    world,
                    lib,
                    pole_poses(junction.location, n),
                    kinds,
                    a.captures,
                    a.every,
                    os.path.join(a.out, f"P{n}"),
                )
            )
    finally:
        for x in reversed(actors):
            x.destroy()
    with open(os.path.join(a.out, "budget.json"), "w") as f:
        json.dump({"gpu": info.get("gpu"), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
