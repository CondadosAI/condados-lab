"""Frame sources: the only way the pipeline gets images, so a simulated camera and a real one swap.

    from sources import open_source
    src = open_source("recordings/harlem")       # a FileSource folder
    src = open_source("clip.mp4")                # or any video file, RTSP URL or webcam index
    for f in src:                                # Frame(image BGR uint8, t seconds, index)
        ...
    src.camera                                   # Camera(width, height, K or None, pose or None)

A recording is a folder: frames/000000.png ... (or video.mp4), timestamps.csv (index,t), camera.json
(width, height, K, pose). Ground truth, when there is any, sits beside it in gt/ and is never read
here: only the evaluator opens it. ValdradaSource renders live from the simulator and needs a
running server.
"""

import csv
import glob
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Camera:
    width: int
    height: int
    K: list | None = None  # 3x3 intrinsics in pixels, when known
    pose: dict | None = (
        None  # where the camera is, when known (free-form: location/rotation, lon/lat)
    )


@dataclass
class Frame:
    image: np.ndarray  # H x W x 3, BGR, uint8
    t: float  # seconds since the first frame
    index: int


class FileSource:
    """A recording folder: frames/ (or video.mp4), timestamps.csv and camera.json."""

    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, "camera.json")) as f:
            self.camera = Camera(**json.load(f))
        self.times = {}
        ts = os.path.join(folder, "timestamps.csv")
        if os.path.exists(ts):
            with open(ts) as f:
                self.times = {int(r["index"]): float(r["t"]) for r in csv.DictReader(f)}
        self.files = sorted(glob.glob(os.path.join(folder, "frames", "*")))
        self.video = os.path.join(folder, "video.mp4")

    def __iter__(self) -> Iterator[Frame]:
        if self.files:
            for i, path in enumerate(self.files):
                yield Frame(cv2.imread(path, cv2.IMREAD_COLOR), self.times.get(i, float(i)), i)
        else:
            for f in StreamSource(self.video):
                yield Frame(f.image, self.times.get(f.index, f.t), f.index)


class StreamSource:
    """Anything OpenCV opens: a video file, an RTSP/HTTP URL, a webcam index."""

    def __init__(self, uri, camera_json=None):
        self.uri = int(uri) if str(uri).isdigit() else uri
        cap = cv2.VideoCapture(self.uri)
        if not cap.isOpened():
            raise OSError(f"cannot open {uri}")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        cam = {"width": w, "height": h}
        if camera_json:  # e.g. K from a calibration
            with open(camera_json) as f:
                cam.update(json.load(f))
        self.camera = Camera(**cam)

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.uri)
        live = isinstance(self.uri, int) or str(self.uri).startswith(
            ("rtsp://", "http://", "https://")
        )
        t0, i = time.monotonic(), 0
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    return
                t = time.monotonic() - t0 if live else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield Frame(img, t, i)
                i += 1
        finally:
            cap.release()


class ValdradaSource:
    """A pole camera in the running Valdrada server, stepped in synchronous mode.

    every: capture every N steps of dt seconds (2 x 0.05 s = 10 fps). gt_kinds adds cameras whose
    data iter_with_gt() returns beside each frame, for the recorder; iterating the source itself
    yields frames only.
    """

    KIND = {
        "instance": "sensor.camera.instance_segmentation",
        "semantic": "sensor.camera.semantic_segmentation",
        "depth": "sensor.camera.depth",
    }

    def __init__(
        self, world, transform, fov=70.0, every=2, dt=0.05, gt_kinds=(), amodal=8, exposure=None
    ):
        import queue

        self.world, self.every, self.dt, self.kinds = world, every, dt, ["rgb", *gt_kinds]
        lib = world.get_blueprint_library()
        self.cams, self.q = {}, {}
        for k in self.kinds:
            bp = lib.find("sensor.camera.rgb" if k == "rgb" else self.KIND[k])
            bp.set_attribute("fov", fov)
            bp.set_attribute("sensor_tick", every * dt if every > 1 else 0)
            if k == "instance":
                bp.set_attribute("amodal", amodal)
            if k == "rgb" and exposure:
                bp.set_attribute("exposure_mode", exposure)
            cam = world.spawn_actor(bp, transform)
            self.q[k] = queue.Queue()
            cam.listen(self.q[k].put)
            self.cams[k] = cam
        self.camera = None  # filled from the first frame's metadata

    def iter_with_gt(self, n=None):
        i, t0 = 0, None
        while n is None or i < n:
            frame = self.world.tick()
            if self.q["rgb"].empty():
                continue
            data = {k: self.q[k].get() for k in self.kinds}
            assert all(d.frame == frame for d in data.values()), "cameras out of sync"
            rgb = data["rgb"]
            img = np.ascontiguousarray(rgb.to_numpy()[..., :3][..., ::-1])  # RGB(A) -> BGR
            t0 = rgb.timestamp if t0 is None else t0
            if self.camera is None:
                self.camera = Camera(img.shape[1], img.shape[0], *_intrinsics_and_pose(data))
            yield Frame(img, rgb.timestamp - t0, i), data
            i += 1

    def __iter__(self) -> Iterator[Frame]:
        for frame, _ in self.iter_with_gt():
            yield frame

    def destroy(self):
        for c in self.cams.values():
            c.destroy()


def _intrinsics_and_pose(data):
    """K from the field of view (horizontal, as the simulator defines it) and the camera's pose."""
    rgb = data["rgb"]
    f = rgb.width / 2 / np.tan(np.radians(rgb.fov) / 2)
    pose = {"transform": rgb.transform.to_dict()}
    if "instance" in data:  # the labels also carry the camera's longitude/latitude
        pose["lonlat"] = data["instance"].camera.get("lonlat")
    return [[f, 0.0, rgb.width / 2], [0.0, f, rgb.height / 2], [0.0, 0.0, 1.0]], pose


def open_source(spec, camera_json=None):
    """A folder with camera.json is a recording; anything else goes to OpenCV."""
    if os.path.isdir(spec) and os.path.exists(os.path.join(spec, "camera.json")):
        return FileSource(spec)
    return StreamSource(spec, camera_json)
