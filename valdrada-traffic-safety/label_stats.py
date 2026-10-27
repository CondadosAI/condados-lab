"""What a recording's labels contain: brightness, who has an actor id, whether instance ids hold.

    python label_stats.py <recording>... [--out output/recording_labels.json]

For each recording folder written by record.py: the mean pixel value of its frames (to compare day
and night), the count of road-user labels over 400 px with and without an actor id, and, for every
pair of consecutive frames, how many boxes that overlap at IoU > 0.7 keep the same instance id.
Reads gt/ (ground truth), so it is evaluator-side code.
"""

import argparse
import collections
import glob
import json
import os

import cv2

ROAD = ("car", "bus", "truck", "bicycle", "pedestrian")


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def stats(rec):
    frames = sorted(glob.glob(os.path.join(rec, "frames", "*.png")))
    brightness = [float(cv2.imread(f).mean()) for f in frames]
    labels = []
    for f in frames:
        name = os.path.splitext(os.path.basename(f))[0]
        with open(os.path.join(rec, "gt", name + ".json")) as fh:
            objs = json.load(fh)["objects"]
        labels.append([o for o in objs if o["class_name"] in ROAD and o["area"] >= 400])
    actor = collections.Counter(
        "actor_id > 0" if o["actor_id"] else "actor_id 0" for fr in labels for o in fr
    )
    classes = collections.Counter(o["class_name"] for fr in labels for o in fr)
    same = changed = 0
    for a, b in zip(labels, labels[1:], strict=False):
        for o in a:
            best = max(b, key=lambda p, o=o: iou(o["bbox"], p["bbox"]), default=None)
            if best is not None and iou(o["bbox"], best["bbox"]) > 0.7:
                same += best["id"] == o["id"]
                changed += best["id"] != o["id"]
    ids = collections.Counter(o["id"] for fr in labels for o in fr)
    return {
        "frames": len(frames),
        "mean_pixel": {
            "mean": sum(brightness) / len(brightness),
            "min": min(brightness),
            "max": max(brightness),
        },
        "road_user_labels_over_400px": dict(actor),
        "classes": dict(classes),
        "consecutive_matches_iou_0.7": {"same_id": same, "different_id": changed},
        "distinct_ids": len(ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--out", default="output/recording_labels.json")
    a = ap.parse_args()
    res = {os.path.basename(os.path.normpath(r)): stats(r) for r in a.recordings}
    for name, s in res.items():
        print(name, json.dumps(s))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
