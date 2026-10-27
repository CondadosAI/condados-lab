"""Fine-tune RF-DETR (COCO weights) on the simulator dataset written by to_coco.py.

    python finetune.py --dataset /f/out/coco --out /f/out/ft_small [--model small] [--epochs 40]

The valid split picks the best checkpoint; the test split is never seen here, it is scored by
zeroshot.py --weights.
"""

import argparse
import json
import os
import time

import rfdetr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/f/out/coco")
    ap.add_argument("--out", default="/f/out/ft_small")
    ap.add_argument("--model", default="small")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    model = getattr(
        rfdetr, {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}[a.model]
    )()
    t0 = time.perf_counter()
    model.train(
        dataset_dir=a.dataset,
        output_dir=a.out,
        epochs=a.epochs,
        batch_size=a.batch,
        grad_accum_steps=a.accum,
        lr=a.lr,
        checkpoint_interval=10,
    )
    with open(os.path.join(a.out, "finetune.json"), "w") as f:
        json.dump(
            {
                "model": a.model,
                "epochs": a.epochs,
                "batch": a.batch,
                "accum": a.accum,
                "lr": a.lr,
                "train_s": time.perf_counter() - t0,
                "rfdetr": getattr(rfdetr, "__version__", "?"),
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
