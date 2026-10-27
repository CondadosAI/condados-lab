"""P0 check 4: zero-shot RF-DETR (COCO weights) recall on simulator frames, easy cases first.

    python zeroshot.py <dir>... [--model small] [--threshold 0.5] [--out /f/out/zeroshot.json]
                                [--weights ckpt.pth]

With --weights, scores a fine-tuned checkpoint (finetune.py) the same way; its class names come
from the checkpoint and "pedestrian" counts as COCO "person", so both models are matched by the
same rule.

Reads every <name>_rgb.png with its <name>_instance.json (Valdrada's saved instance labels) under
the given directories. Ground truth: visible bbox of road users. "Easy" = vehicle, occlusion
measured and < 0.1, area > 2,000 px, not truncated. A ground-truth box is found when a detection
of a matching COCO class overlaps it at IoU >= 0.5 (greedy, highest score first). Prints recall
per slice and false positives per frame.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

VEHICLE = {"car", "bus", "truck"}
COCO_FOR = {
    "car": {"car", "truck", "bus"},
    "bus": {"bus", "truck", "car"},
    "truck": {"truck", "car", "bus"},
    "pedestrian": {"person"},
    "bicycle": {"bicycle", "motorcycle"},
}


def iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def slices(o):
    s = ["all " + ("vehicle" if o["class_name"] in VEHICLE else o["class_name"])]
    occ = o.get("occlusion")
    if (
        o["class_name"] in VEHICLE
        and occ is not None
        and occ < 0.1
        and o["area"] > 2000
        and not o.get("truncated")
    ):
        s.append("EASY vehicle")
    if occ is not None:
        s.append(
            f"{'vehicle' if o['class_name'] in VEHICLE else o['class_name']} occl "
            + ("<0.1" if occ < 0.1 else "0.1-0.5" if occ < 0.5 else ">=0.5")
        )
    if o["area"] < 400:
        s.append("tiny (<400 px)")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--model", default="small")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="/f/out/zeroshot.json")
    ap.add_argument("--weights", default=None)
    a = ap.parse_args()
    import rfdetr
    from rfdetr.assets.coco_classes import COCO_CLASSES

    cls = getattr(
        rfdetr, {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}[a.model]
    )
    if a.weights:
        model = cls(pretrain_weights=a.weights)
        names = list(model.class_names)
        print("fine-tuned class names:", names, flush=True)

        def name_of(c):
            return {"pedestrian": "person"}.get(names[c], names[c]) if 0 <= c < len(names) else "?"

    else:
        model = cls()

        def name_of(c):
            return COCO_CLASSES.get(c, "?")

    pairs = []
    for d in a.dirs:
        for rgb in sorted(glob.glob(os.path.join(d, "**", "*_rgb.png"), recursive=True)):
            js = rgb[: -len("_rgb.png")] + "_instance.json"
            if os.path.exists(js):
                pairs.append((rgb, js))
        for rgb in sorted(glob.glob(os.path.join(d, "rgb", "*.png"))):  # record_dataset.py layout
            js = os.path.join(d, "instance", os.path.basename(rgb)[:-4] + ".json")
            if os.path.exists(js):
                pairs.append((rgb, js))
    print(f"{len(pairs)} frames", flush=True)
    found, total, fps, per_frame, confusion = {}, {}, 0, [], {}
    for rgb, js in pairs:
        with open(js) as f:
            lab = json.load(f)
        gts = [o for o in lab["objects"] if o["class_name"] in COCO_FOR and o["area"] >= 100]
        img = Image.open(rgb).convert("RGB")
        det = model.predict(img, threshold=a.threshold)
        dets = sorted(
            (
                (float(s), [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], name_of(int(c)))
                for (x0, y0, x1, y1), s, c in zip(
                    det.xyxy, det.confidence, det.class_id, strict=True
                )
            ),
            reverse=True,
        )
        used = [False] * len(gts)
        for _s, box, cname in dets:  # class-agnostic best overlap: checks the class mapping
            ious = [iou(box, g["bbox"]) for g in gts]
            if ious and max(ious) >= 0.5:
                k = f"{cname}<-{gts[int(np.argmax(ious))]['class_name']}"
                confusion[k] = confusion.get(k, 0) + 1
        for _s, box, cname in dets:
            best, bi = 0.5, -1
            for i, g in enumerate(gts):
                if not used[i] and cname in COCO_FOR[g["class_name"]]:
                    v = iou(box, g["bbox"])
                    if v >= best:
                        best, bi = v, i
            if bi >= 0:
                used[bi] = True
            elif cname in {"car", "truck", "bus", "person", "bicycle", "motorcycle"}:
                fps += 1
        for g, u in zip(gts, used, strict=True):
            for sl in slices(g):
                total[sl] = total.get(sl, 0) + 1
                found[sl] = found.get(sl, 0) + int(u)
        per_frame.append({"rgb": rgb, "gt": len(gts), "found": sum(used), "dets": len(dets)})
    rows = {sl: {"n": total[sl], "recall": found[sl] / total[sl]} for sl in sorted(total)}
    for sl, r in rows.items():
        print(f"{sl:28s} n={r['n']:5d}  recall {r['recall']:.3f}")
    print(
        "predicted<-ground truth at IoU>=0.5:",
        dict(sorted(confusion.items(), key=lambda kv: -kv[1])[:12]),
    )
    print(
        f"false positives per frame (road-user classes, no match): {fps / max(1, len(pairs)):.2f}"
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(
            {
                "model": a.model,
                "threshold": a.threshold,
                "frames": len(pairs),
                "slices": rows,
                "fp_per_frame": fps / max(1, len(pairs)),
                "per_frame": per_frame,
                "weights": a.weights,
                "confusion": confusion,
                "rfdetr": getattr(rfdetr, "__version__", "?"),
                "argv": sys.argv,
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
