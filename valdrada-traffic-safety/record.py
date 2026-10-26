"""Record a pole camera at a New York junction as a FileSource folder, ground truth in a sidecar.

    python record.py --out /f/out/rec_harlem [--lat 40.80955 --lon -73.95905] [--frames 100]
                     [--bearing 0 --dist 14 --height 8] [--time day --rain 0] [--exposure manual]

Writes <out>/frames/NNNNNN.png, timestamps.csv, camera.json (what any pipeline may read) and
<out>/gt/NNNNNN.json (instance labels: visible and amodal boxes, occlusion, actor ids),
gt/NNNNNN_instance.png and gt/NNNNNN_depth.png (16-bit millimetres): what only the evaluator reads.
The default pose is the one the pole search in check3.py ranked first at W 120th St & Amsterdam Ave.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict

import cv2
import valdrada
from sources import ValdradaSource
from valdrada import Location, Rotation, Transform, WeatherParameters

DT = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lat", type=float, default=40.80955)
    ap.add_argument("--lon", type=float, default=-73.95905)
    ap.add_argument("--bearing", type=float, default=0.0)
    ap.add_argument("--dist", type=float, default=14.0)
    ap.add_argument("--height", type=float, default=8.0)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--every", type=int, default=2, help="steps of 0.05 s per frame (2 = 10 fps)")
    ap.add_argument("--time", default="day", choices=["day", "golden", "dusk", "night"])
    ap.add_argument("--rain", type=float, default=0.0)
    ap.add_argument("--exposure", default=None, choices=[None, "auto", "manual"])
    ap.add_argument("--vehicles", type=int, default=20)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()

    client = valdrada.Client()
    client.set_timeout(300.0)
    world = client.get_world()
    world.apply_settings(valdrada.WorldSettings(synchronous_mode=True, fixed_delta_seconds=DT))
    m = world.get_map()
    here = m.geolocation_to_location(a.lat, a.lon)
    world.get_spectator().set_transform(Transform(here + Location(0, 0, 30)))
    world.wait_until_loaded()
    centre = m.get_junctions(center=here, radius=60)[0].location
    world.set_weather(WeatherParameters(a.time, a.rain))

    rng = random.Random(a.seed)
    lib = world.get_blueprint_library()
    bps = [
        b for b in lib.filter("vehicle.*") if b.id not in ("vehicle.firetruck", "vehicle.ambulance")
    ]
    sps = m.get_spawn_points(center=centre, radius=120, spacing=16)
    rng.shuffle(sps)
    actors = []
    for sp in sps[: a.vehicles]:
        v = world.try_spawn_actor(rng.choice(bps), sp)
        if v is not None:
            v.set_autopilot(True)
            actors.append(v)

    b = math.radians(a.bearing)
    loc = centre + Location(a.dist * math.cos(b), a.dist * math.sin(b), a.height)
    pitch = -math.degrees(math.atan2(a.height, a.dist)) + 6  # aim a little past the centre
    pose = Transform(loc, Rotation(pitch=pitch, yaw=a.bearing + 180))

    src = None
    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "gt"), exist_ok=True)
    try:
        for _ in range(int(6.0 / DT)):  # let traffic start moving
            world.tick()
        src = ValdradaSource(
            world, pose, every=a.every, dt=DT, gt_kinds=("instance", "depth"), exposure=a.exposure
        )
        t_wall, times = time.perf_counter(), []
        for frame, data in src.iter_with_gt(a.frames):
            name = f"{frame.index:06d}"
            cv2.imwrite(os.path.join(a.out, "frames", name + ".png"), frame.image)
            data["instance"].save_to_disk(os.path.join(a.out, "gt", name + "_instance.png"))
            os.replace(
                os.path.join(a.out, "gt", name + "_instance.json"),
                os.path.join(a.out, "gt", name + ".json"),
            )
            data["depth"].save_to_disk(os.path.join(a.out, "gt", name + "_depth.png"))
            times.append((frame.index, frame.t))
        wall = time.perf_counter() - t_wall
    finally:
        if src:
            src.destroy()
        for x in reversed(actors):
            x.destroy()

    with open(os.path.join(a.out, "timestamps.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "t"])
        w.writerows((i, f"{t:.3f}") for i, t in times)
    with open(os.path.join(a.out, "camera.json"), "w") as f:
        json.dump(asdict(src.camera), f, indent=1)
    with open(os.path.join(a.out, "recording.json"), "w") as f:
        json.dump(
            {
                "args": vars(a),
                "frames": len(times),
                "wall_s": wall,
                "server": client.get_server_info(),
            },
            f,
            indent=1,
            default=str,
        )
    print(f"{len(times)} frames in {wall:.0f} s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
