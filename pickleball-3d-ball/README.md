# Reconstructing pickleball flights with bounces and physics

Companion code for the condados.ai post. One fixed broadcast camera sees the ball as a pixel, and a pixel is a
viewing ray, not a point. Between two bounces the ball is a projectile, and at each bounce it is on the floor. Those
two facts are enough to place the whole flight in 3D: one trajectory under gravity and drag has to pass through every
ray of the flight and touch the floor at both bounces. `fit.py` finds it by least squares.

## Results

80 groundstrokes (bounce, paddle, bounce) from one match, one camera.

| check | what it tests | result | file |
|---|---|---|---|
| Gravity left free in every fit | is the reconstruction physically plausible on average? | median **9.55 ± 0.12** m/s² over 80 flights (true 9.81; ± is the bootstrap standard error of the median); 10.06 ± 0.15 with the camera's lens distortion switched off | `output/gravity.json` |
| Synthetic control | does the fit recover g when everything is known? | 9.78 ± 0.04 (lens model right), 10.20 or 9.36 (lens model wrong) | `output/gravity.json` |
| Pixel fit, g fixed at 9.81 | do the arcs explain the detections? | median residual 1.6 px | `output/fits.json` |
| The net | does every flight that crossed it clear it? | 78 cross; **65 clear**, 3 miss by under 5 cm, **10 miss by more** (up to 0.80 m); 8 of those 10 have a good pixel fit (≤ 3 px) | `output/net.json` |
| The 8 good fits under the net | does drag, the paddle time, spin or the lens explain them? | no single change lifts all 8 over | `output/diagnose_net.json` |

One flight's gravity is not a measurement (interquartile range 9.07 to 10.65 m/s²); the median over 80 is a check
that the method is plausible on average. It does not say that every flight's height is right, and the net check
shows that some are not.

## Data and licences

| data | licence | what this repo keeps |
|---|---|---|
| Women's doubles gold-medal match, [pickleball4you](https://www.youtube.com/watch?v=T5rmWjvt8Os) (YouTube `T5rmWjvt8Os`) | CC BY 3.0 | `flights.json`: per flight, the ball's pixel positions and times, the bounce and hit times; the fitted camera; court dimensions. No frames |
| Ball detections, bounce and hit events | made with a ball detector and event model that are not part of this repo | only their output, in `flights.json` |

The video itself is not redistributed; `render.py` needs a copy you download yourself.

## Run it

From the repository root, `uv sync` once, then:

```bash
uv run python pickleball-3d-ball/check_gravity.py   # g free, synthetic + real -> output/gravity.json, fits_<camera>.json (~20 min, 20 cores)
uv run python pickleball-3d-ball/fit_all.py         # g = 9.81, the arcs the post draws -> output/fits.json (~2 min)
uv run python pickleball-3d-ball/check_net.py       # clearance at the net -> output/net.json
uv run python pickleball-3d-ball/check_net.py pickleball-3d-ball/output/fits_radial.json   # same with g free
uv run python pickleball-3d-ball/diagnose_net.py    # the 8 good fits under the net, suspect by suspect -> output/diagnose_net.json
uv run python pickleball-3d-ball/render.py VIDEO out.mp4 51 52 53 54          # the post's demo (rally 51-54)
uv run --with jupyter jupyter lab pickleball-3d-ball/notebook.ipynb
```

`notebook.ipynb` refits one flight live (flight 52, 10 s on a laptop CPU), checks it against the net, and loads the
saved tables for all 80. No GPU anywhere.

## What the fit assumes

- **Gravity and quadratic drag**, a = −g ẑ − k|v|v with k = 0.05 per metre. k comes from the ball's mass and diameter
  in the USA Pickleball rules (2.D) and an assumed drag coefficient of 0.4 to 0.5 for a perforated ball; I did not
  find a published measurement for it.
- **No spin.**
- **The bounce is on the floor**: the ball's centre at height 0, which ignores its 3.6 cm radius at contact.
- **The paddle time may move** up to 0.15 s from the detected hit; the fit keeps the time that explains the pixels
  best. On the demo rally it moves 1 to 3 frames later.
- **One camera**, calibrated on the court lines with one radial distortion term.
