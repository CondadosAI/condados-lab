# Traffic-safety cameras in a simulated New York: running Valdrada headless on Linux

Companion code for a condados.ai series (in progress). [Valdrada](https://github.com/mkturkcan/valdrada) is an
open-source simulator of New York City built from public map data, with a CARLA-like Python API and sensors that
come with exact labels: semantic and instance segmentation, depth in metres, and per-object visible and amodal
boxes with occlusion ratios. This folder is the first step of a traffic-safety system built on it: get the server
rendering on an NVIDIA GPU with no monitor, measure what a labelled frame costs, check that the data is usable,
and see how far a COCO detector gets on it before and after a short fine-tune.

The pipeline is built so that the camera is a swappable source: the simulator is one frame source among others
(a recorded file, an RTSP camera), and the ground truth feeds only the evaluator.

## What we found

Everything below was measured on one desktop, RTX 5060 (8 GB), Valdrada v0.5.2, server at 1280x720.

**1. The GPU-backed display is the whole game.** Under Xvfb the server falls back to software GL and does not
finish loading the city. A headless Xorg running the NVIDIA driver inside the container
(`docker/Dockerfile.xorg`, `docker/xorg-nvidia.conf`) plus Chromium's ANGLE-on-GL backend
(`--use-gl=angle --use-angle=gl --ignore-gpu-blocklist`) renders on the GPU. Two other routes failed: ANGLE over
EGL on Xvfb (still software), and the Vulkan flags on the NVIDIA Xorg (no WebGL context).

| Seconds per tick (`PythonAPI/examples/benchmark.py`) | Headless Xorg + ANGLE/GL |
|---|---|
| No sensors | 0.09–0.11 |
| RGB + semantic + instance | 0.22 |
| + 8 amodal masks | 0.28 |

The GPU sat at about 28% busy at 45 W during these runs: the remaining cost is on the CPU side of the server.

**2. Several cameras on one tick are almost free.** Sensors capture every N steps (`sensor_tick`), and the steps
in between cost only the world update (0.10 s). Put P pole cameras on the same capture step and they share it
(`budget.py`, `output/budget.json`, 12 captures per arm, RGB + instance with 8 amodal masks, about 55 road users
per frame):

| Pole cameras | Seconds per capture | Seconds per labelled frame | 20,000 frames |
|---|---|---|---|
| 1 | 2.06 | 2.06 | 11.4 h |
| 2 | 2.43 | 1.21 | 6.7 h |
| 4 | 2.99 | **0.75** | 4.2 h |

**3. The data checks** (`check3.py`, `output/site_checks.json`):
- The author's `record_dataset.py` runs unchanged: 24 frames, 1,862 boxes, a valid COCO file.
- Camera poles need a per-site search. Of 48 candidate poses around W 120th St & Amsterdam Ave, 34 pass (less than
  10% vegetation, less than 5% of the frame closer than 5 m, more than 20% road). At a second junction
  (Broadway & W 96th St) with fixed poses and no search, one camera landed inside a tree.
- Weather and time of day work (day, golden hour, dusk, night, rain), but the RGB camera's default auto exposure
  meters every frame, so night comes out brighter than day (mean pixel 107 against 96). Night data needs
  `exposure_mode="manual"`.
- A second junction streams in within 9 s of moving there.
- An articulated bus is labelled as two boxes, one per body.

**4. A COCO detector on simulated frames** (`zeroshot.py`, RF-DETR Small, COCO weights, no training, all 218
frames, `output/zeroshot_all.json`). Recall counts a ground-truth box as found when a detection of a matching class
overlaps it at IoU ≥ 0.5:

| Slice | Boxes | Recall |
|---|---|---|
| Easy vehicles (occlusion < 0.1, area > 2,000 px, not truncated) | 443 | **0.995** |
| Vehicles, occlusion < 0.1 | 573 | 0.953 |
| Vehicles, occlusion 0.1–0.5 | 191 | 0.806 |
| All vehicles | 4,548 | 0.375 |
| All pedestrians | 6,431 | 0.174 |
| Tiny (< 400 px) | 6,192 | 0.064 |

1.1 false positives per frame. The render is close enough to real footage that a COCO model finds nearly every
easy vehicle; what it misses is small, distant and occluded, which is what fine-tuning is for.

**5. A first fine-tune** (`to_coco.py`, `finetune.py`, `train.sh`). The frames above, split by capture run so no
split shares a frame or a 10 fps neighbour of one (`output/splits.json`):

| Split | Frames | Boxes | Where from |
|---|---|---|---|
| train | 84 | 4,709 | `budget.py`: four poles on the junction's diagonals, 7 m high |
| valid | 24 | 1,862 | the author's `record_dataset.py` pose (picks the checkpoint) |
| test | 110 | 4,408 | `check3.py`: the pole-search pose (clip + six conditions) and 4 frames at Broadway & W 96th |

RF-DETR Small from its COCO weights, 40 epochs, batch 4 × 4 accumulation, learning rate 1e-4, 9 min on the RTX
5060; the best EMA checkpoint (epoch 21, valid mAP50:95 0.32) is scored on the test split, both models at
confidence 0.5 (`output/zeroshot_test.json`, `output/finetuned_test.json`):

| Test slice | Boxes | COCO weights | Fine-tuned |
|---|---|---|---|
| Easy vehicles | 332 | 0.994 | 0.997 |
| Vehicles, occlusion 0.1–0.5 | 173 | 0.832 | **0.983** |
| All vehicles | 2,574 | 0.354 | 0.517 |
| All pedestrians | 1,834 | 0.333 | **0.736** |
| Tiny (< 400 px) | 2,202 | 0.124 | 0.374 |
| False positives per frame | | 1.11 | 3.83 |

Read it as a pipeline check, not as the result of fine-tuning on synthetic data. The train and test frames come
from the same junction (only 4 test frames are from another one), from different poles and runs, so this measures
"new viewpoint, same street", not a new city. Recall rises on everything the COCO model missed, and so do the false
positives: part of them may be objects under the 100 px ground-truth floor, and the 0.5 threshold was not tuned for
either model. The scorer also prints which class each detection overlapped (`confusion` in the JSON): the
fine-tuned model calls 131 trucks "car", which this recall counts as found, as it does for the COCO model.

## Run it

The server is an Electron app; everything runs in one container with the GPU. From this folder:

```bash
git clone -b v0.5.2 https://github.com/mkturkcan/valdrada src
bash docker/create.sh          # images + the valdrada-gpu container (this folder is /f inside it)
bash setup.sh                  # npm installs, the v0.5.2 asset banks from Hugging Face, the Python API venv
docker exec valdrada-gpu /f/home/venv/bin/pip install "rfdetr[train]==1.11.2"

# headless NVIDIA Xorg + ANGLE/GL, then the checks (record_dataset, check3, zero-shot):
MODE=xorg CHECKS=1 bash probe.sh C2 "--use-gl=angle --use-angle=gl --ignore-gpu-blocklist"
# the seconds-per-frame budget with 1, 2 and 4 poles:
MODE=xorg BUDGET="--poses 1,2,4 --captures 12" bash probe.sh B1 "--use-gl=angle --use-angle=gl --ignore-gpu-blocklist"
# COCO split, fine-tune, score both models on the test split:
bash train.sh T2
```

Each batch writes `logs/<name>.batch.log` and ends with a `logs/<name>_DONE` or `_FAILED` marker naming the step
that failed. `probe.sh` and `train.sh` were written for a shared machine: they record which GPU processes exist at
launch and stop the container if a new one appears.

`xorg-nvidia.conf` carries the card's PCI bus id, and `create.sh` mounts the two NVIDIA Xorg modules from the host
driver: both depend on the machine.

## Licences and credits

- Valdrada (code MIT) by Mehmet Kerem Turkcan. The compiled city is ODbL 1.0; images and labels rendered from it
  carry the attribution: contains information from NYC Open Data, NY State open data and © OpenStreetMap
  contributors. Vehicle, prop and pedestrian models are CC BY 4.0, mostly from the CARLA Simulator; Valdrada's
  `ACKNOWLEDGEMENTS.md` lists every required credit.
- Every frame here was rendered with `--pedestrians procedural` (the built-in procedural pedestrians). Valdrada's
  photoreal pedestrians contain MetaHuman components, whose licence does not allow training or testing AI models.
- RF-DETR (Apache-2.0), Roboflow.
- The code in this folder is Apache-2.0, like the rest of the repository.
