"""Valdrada instance labels -> an RF-DETR COCO dataset (train/valid/test folders).

    python to_coco.py --train <dir>... --valid <dir>... --test <dir>... --out /f/out/coco

Each split takes whole capture directories, so a split never shares a frame (or a 10 fps
neighbour of one) with another. Reads both layouts: <name>_rgb.png + <name>_instance.json (the
probe scripts) and rgb/*.png + instance/*.json (the author's record_dataset.py). Boxes are the
visible bbox of road users with area >= 100 px, the same filter the zero-shot scorer uses for
ground truth.
"""

import argparse
import glob
import json
import os
import shutil

from PIL import Image

CLASSES = ["car", "bus", "truck", "bicycle", "pedestrian"]
MIN_AREA = 100


def frames(d):
    for rgb in sorted(glob.glob(os.path.join(d, "**", "*_rgb.png"), recursive=True)):
        js = rgb[: -len("_rgb.png")] + "_instance.json"
        if os.path.exists(js):
            yield rgb, js
    for rgb in sorted(glob.glob(os.path.join(d, "rgb", "*.png"))):
        js = os.path.join(d, "instance", os.path.basename(rgb)[:-4] + ".json")
        if os.path.exists(js):
            yield rgb, js


def build(dirs, out):
    os.makedirs(out, exist_ok=True)
    images, anns = [], []
    for d in dirs:
        tag = os.path.basename(os.path.normpath(d))
        for rgb, js in frames(d):
            name = f"{tag}__{os.path.relpath(rgb, d).replace(os.sep, '__')}"
            shutil.copy(rgb, os.path.join(out, name))
            w, h = Image.open(rgb).size
            img_id = len(images) + 1
            images.append({"id": img_id, "file_name": name, "width": w, "height": h})
            with open(js) as f:
                objects = json.load(f)["objects"]
            for o in objects:
                if o["class_name"] not in CLASSES or o["area"] < MIN_AREA:
                    continue
                x, y, bw, bh = (float(v) for v in o["bbox"])
                anns.append(
                    {
                        "id": len(anns) + 1,
                        "image_id": img_id,
                        "category_id": CLASSES.index(o["class_name"]) + 1,
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                    }
                )
    cats = [{"id": i + 1, "name": c, "supercategory": "road_user"} for i, c in enumerate(CLASSES)]
    with open(os.path.join(out, "_annotations.coco.json"), "w") as f:
        json.dump({"images": images, "annotations": anns, "categories": cats}, f)
    per = {c: sum(a["category_id"] == i + 1 for a in anns) for i, c in enumerate(CLASSES)}
    print(f"{out}: {len(images)} images, {len(anns)} boxes {per}", flush=True)
    return {"images": len(images), "boxes": len(anns), "per_class": per, "dirs": dirs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--valid", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out", default="/f/out/coco")
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    summary = {s: build(getattr(a, s), os.path.join(a.out, s)) for s in ("train", "valid", "test")}
    with open(os.path.join(a.out, "splits.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
