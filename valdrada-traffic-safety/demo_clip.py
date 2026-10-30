"""The post's opening clip: a pole camera at W 120th St & Amsterdam Ave with moving traffic.

    python demo_clip.py [--out /f/out/demo_clip] [--seconds 120] [--active 14] [--walkers 120]
                        [--gate]

In Valdrada v0.5.2 the city's ambient vehicles in this view are parked, and the autopilot does
not get through this junction: with nothing else on the street, cars stop at the line through
every green or make a U-turn. So the vehicles here are spawned 70 to 110 m out on the junction's
arms and driven by this script. The driver is adapted from the LaneFollower in Valdrada's
examples/intersection_120_amsterdam.py (MIT): pure pursuit along the lane, straight through the
junction, a stop at the line while its light is not green. Changes that keep the junction clear:
- no vehicle longer than 8 m (bounding-box half length over 4 m is destroyed at spawn): an
  articulated bus swung across the box on the straight path;
- a vehicle enters only with at least 6 s of green left, so it clears during yellow + all-red;
- it yields to a pedestrian only within 10 m ahead and to vehicles within 35 m, but not to a
  script vehicle heading the other way (over 120 degrees off): the obstacle corridor clips the
  next lane when a car hugs the centre line, and two cars facing each other deadlocked;
- a vehicle is removed after 75 s without moving, or once it is 120 m past the junction.
Per frame it counts only the script's vehicles (actor_id > 0); the parked cars have actor_id 0.
--gate: no cameras, 90 s after the warm-up, prints why each car is stopped and exits 1 unless 8
or more vehicles got through. Without it: after 30 s of warm-up the pole-search pose records
--seconds at 10 fps (RGB, semantic, instance, depth), every second road-user control update at
0.1 s. Writes <out>/NNNN_*.png|json and <out>/demo_clip.json.
"""

import argparse
import json
import math
import os
import random
import time
from collections import Counter

from check3 import DT, Rig, candidates, go_to, save, valdrada
from valdrada import Rotation, Transform, Vector3D, VehicleControl

VEHICLES = ("car", "bus", "truck", "bicycle")


def unit(yaw):
    return Vector3D(math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def centre(b):
    return b[0] + b[2] / 2, b[1] + b[3] / 2


class Driver:
    """Pure pursuit along the lane, straight through the junction; stops for light and traffic."""

    def __init__(self, world, vehicle, junction, cruise, peers):
        self.world, self.v, self.j, self.cruise, self.peers = (
            world,
            vehicle,
            junction,
            cruise,
            peers,
        )
        self.map = world.get_map()
        self.why = None
        self.obs = None
        self.along = None

    def step(self):
        t = self.v.get_transform()
        speed = self.v.get_speed()
        wp = self.map.get_waypoint(t.location, heading=t.rotation.yaw)
        ahead = wp.next(max(5.0, 0.9 * speed + 4.0)) if wp else None
        if not ahead:
            self.why = "no lane"
            return VehicleControl(brake=1.0)
        target = min(ahead, key=lambda w: abs(wrap(w.transform.rotation.yaw - t.rotation.yaw)))
        p = t.inverse_transform_point(target.transform.location)
        curvature = 2.0 * p.y / max(1.0, p.x * p.x + p.y * p.y)
        steer = max(-1.0, min(1.0, -math.degrees(math.atan(2.8 * curvature)) / 35.0))
        along = (self.j.location - t.location).dot(unit(t.rotation.yaw))
        want, self.why = self.cruise, None
        if 9.0 < along < 45.0:
            light = self.world.get_traffic_light_state(t.rotation.yaw)
            if light.state != "green" or (light.time_left < 6.0 and along > 16.0):
                stop_d = along - 12.5
                want = 0.0 if stop_d < 0.8 else min(self.cruise, math.sqrt(2.0 * 2.5 * stop_d))
                self.why = "light" if want == 0.0 else None
        obs = self.v.get_obstacle_ahead(max_distance=35.0, width=2.4)
        self.obs, self.along = obs, along
        peer = self.peers.get(obs["actor_id"]) if obs is not None else None
        if peer is not None and abs(wrap(peer.get_transform().rotation.yaw - t.rotation.yaw)) > 120:
            obs = None  # oncoming car in the next lane: the corridor clips it near the centre line
        if obs is not None and (obs["kind"] != "walker" or obs["distance"] < 10.0):
            gap = obs["distance"] - 3.0
            w = 0.0 if gap < 0.5 else math.sqrt(2.0 * 3.0 * gap)
            if w < want:
                want = w
                self.why = obs["kind"] if w == 0.0 else None
        err = want - speed
        if err >= 0:
            return VehicleControl(throttle=min(1.0, 0.25 + 0.35 * err), steer=steer)
        return VehicleControl(
            throttle=0.0, brake=min(1.0, -0.4 * err + (0.3 if want == 0.0 else 0.0)), steer=steer
        )


class Traffic:
    """Vehicles fed into the junction from its arms under Drivers; removed once gone or stuck."""

    def __init__(self, world, m, lib, junction, rng, active):
        self.world, self.m, self.j, self.rng, self.active = world, m, junction, rng, active
        self.bps = [
            b
            for b in lib.filter("vehicle.*")
            if b.id not in ("vehicle.firetruck", "vehicle.ambulance")
        ]
        self.cars = {}  # vehicle -> [Driver, last position, seconds still]
        self.by_id = {}  # actor id -> vehicle, for the drivers' oncoming check
        self.spawned = self.gone = self.stuck = self.failed = self.too_long = 0

    def spawn_one(self):
        arm = self.rng.choice(self.j.arms)
        toward = wrap(arm.heading + 180)
        wp = self.m.get_waypoint(
            self.j.location + unit(arm.heading) * self.rng.uniform(70, 110), heading=toward
        )
        if wp is None or abs(wrap(wp.transform.rotation.yaw - toward)) > 45:
            self.failed += 1
            return
        tf = Transform(
            wp.transform.location + Vector3D(0, 0, 0.3), Rotation(yaw=wp.transform.rotation.yaw)
        )
        v = self.world.try_spawn_actor(self.rng.choice(self.bps), tf)
        if v is None:
            self.failed += 1
            return
        if v.bounding_box is not None and v.bounding_box.extent.x > 4.0:
            v.destroy()
            self.too_long += 1
            return
        self.cars[v] = [
            Driver(self.world, v, self.j, self.rng.uniform(7.0, 10.0), self.by_id),
            v.get_location(),
            0.0,
        ]
        self.by_id[v.id] = v
        self.spawned += 1

    def update(self, dt):
        for v in list(self.cars):
            d, last, still = self.cars[v]
            loc = v.get_location()
            still = still + dt if loc.distance_2d(last) < 0.5 else 0.0
            far = loc.distance_2d(self.j.location) > 120
            if far or still > 75:
                v.destroy()
                del self.cars[v]
                del self.by_id[v.id]
                self.gone += far
                self.stuck += not far
            else:
                self.cars[v] = [d, loc, still]
        if len(self.cars) < self.active:
            self.spawn_one()

    def drive(self):
        for v, (d, _, _) in self.cars.items():
            v.apply_control(d.step())

    def why_stopped(self):
        return Counter(d.why for d, _, _ in self.cars.values() if d.why)

    def destroy(self):
        for v in self.cars:
            v.destroy()
        self.cars = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/f/out/demo_clip")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--active", type=int, default=14)
    ap.add_argument("--walkers", type=int, default=120)
    ap.add_argument(
        "--gate",
        action="store_true",
        help="no cameras: 90 s of traffic after warm-up, exit 1 unless 8 or more got through",
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(13)
    client = valdrada.Client()
    client.set_timeout(300.0)
    world = client.get_world()
    world.apply_settings(valdrada.WorldSettings(synchronous_mode=True, fixed_delta_seconds=DT))
    world.set_weather(valdrada.WeatherParameters("day", 0.0))
    m = world.get_map()
    lib = world.get_blueprint_library()
    ambient0 = world.get_ambient_traffic()

    junction, load_s, _ = go_to(world, m, "amsterdam_w120")
    pose = next(
        c
        for c in candidates(junction.location)
        if c["bearing"] == 0 and c["dist"] == 14 and c["height"] == 8
    )
    world.set_ambient_traffic(vehicles=0, walkers=a.walkers)
    traffic = Traffic(world, m, lib, junction, rng, a.active)
    rig = None
    frames, whys = [], []
    wall = 0.0

    def step(k):
        if k % 10 == 0:
            traffic.update(10 * DT)
        if k % 2 == 0:
            traffic.drive()
        if k % 20 == 0:
            whys.append(dict(traffic.why_stopped()))

    try:
        for k in range(int(30.0 / DT)):
            step(k)
            world.tick()
        print(
            f"warm-up: {len(traffic.cars)} vehicles, spawned {traffic.spawned}, "
            f"stuck {traffic.stuck}, "
            f"too long {traffic.too_long}, stopped by {whys[-1]}",
            flush=True,
        )
        if a.gate:
            for k in range(int(90.0 / DT)):
                step(k)
                world.tick()
                if k % 200 == 0:
                    print(
                        f"gate t={k * DT:.0f} s: through {traffic.gone}, stuck {traffic.stuck}, "
                        f"stopped by {whys[-1]}",
                        flush=True,
                    )
            for v, (d, _, _) in traffic.cars.items():
                t = v.get_transform()
                wp = m.get_waypoint(t.location, heading=t.rotation.yaw)
                print(
                    f"  car {v.id}: along {d.along:6.1f} yaw {t.rotation.yaw:6.1f} "
                    f"v {v.get_speed():4.1f} "
                    f"why {d.why} obs {d.obs} lane {wp}",
                    flush=True,
                )
            print(
                f"gate: through {traffic.gone}, stuck {traffic.stuck} of {traffic.spawned}",
                flush=True,
            )
            raise SystemExit(0 if traffic.gone >= 8 else 1)
        rig = Rig(world, lib, pose["tf"], ["rgb", "semantic", "instance", "depth"], every=2)
        rig.drain()
        prev, got, k, t0 = {}, 0, 0, time.perf_counter()
        while got < int(a.seconds * 10):
            step(k)
            k += 1
            frame, data = rig.tick()
            if data is None:
                continue
            save(data, a.out, f"{got:04d}")
            cur = {
                o.id: o.bbox
                for o in data["instance"].labels
                if o.class_name in VEHICLES and o.area >= 400 and o.actor_id > 0
            }
            moving = sum(
                math.dist(centre(b), centre(prev[i])) > 1.0 for i, b in cur.items() if i in prev
            )
            frames.append(
                {
                    "i": got,
                    "frame": frame,
                    "driven_in_view": len(cur),
                    "moving": moving,
                    "driven": len(traffic.cars),
                }
            )
            prev = cur
            got += 1
            if got % 50 == 0:
                f = frames[-1]
                print(
                    f"{got} frames: {f['driven_in_view']} driven vehicles in view, "
                    f"{f['moving']} moving; "
                    f"spawned {traffic.spawned}, through {traffic.gone}, stuck {traffic.stuck}; "
                    f"stopped by {whys[-1]}",
                    flush=True,
                )
        wall = time.perf_counter() - t0
    finally:
        if rig:
            rig.destroy()
        traffic.destroy()
        world.set_ambient_traffic(
            vehicles=ambient0.get("vehicles"), walkers=ambient0.get("walkers")
        )
        with open(os.path.join(a.out, "demo_clip.json"), "w") as f:
            json.dump(
                {
                    "load_s": load_s,
                    "pose": {k: v for k, v in pose.items() if k != "tf"},
                    "active": a.active,
                    "walkers": a.walkers,
                    "spawned": traffic.spawned,
                    "through": traffic.gone,
                    "stuck": traffic.stuck,
                    "too_long": traffic.too_long,
                    "spawn_failed": traffic.failed,
                    "wall_s": wall,
                    "stopped_by_per_second": whys,
                    "per_frame": frames,
                },
                f,
                indent=1,
            )
    if a.gate:
        return
    n = len(frames)
    print(
        f"done: {n} frames in {wall:.0f} s; mean "
        f"{sum(f['driven_in_view'] for f in frames) / n:.1f} driven vehicles in view, "
        f"{sum(f['moving'] for f in frames) / n:.1f} moving; through {traffic.gone}, "
        f"stuck {traffic.stuck} of {traffic.spawned}",
        flush=True,
    )
    print("ok", flush=True)


if __name__ == "__main__":
    main()
