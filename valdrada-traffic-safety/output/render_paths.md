# Render paths tried for the headless server (Valdrada v0.5.2, RTX 5060, driver 590.48.01, in Docker)

Every row is one `probe.sh` batch (24–25 Oct 2026). "Renderer reported" is the renderer string the server reported; "s/tick" is
`PythonAPI/examples/benchmark.py --ticks 60` at 1280x720, two runs where the batch got that far.

| Variant | Display | Chromium flags | Renderer reported | s/tick, no sensors | s/tick, RGB + semantic + instance | Outcome |
|---|---|---|---|---|---|---|
| A | Xvfb | (none) | ANGLE on Mesa llvmpipe (software) | – | – | the city did not finish loading within 210 s (`wait_until_loaded` timeout) |
| B | Xvfb | `--use-gl=angle --use-angle=vulkan --enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE --ignore-gpu-blocklist` | ANGLE on Vulkan 1.4.325, RTX 5060 | 0.130 / 0.125 | 0.334 / 0.335 | works on the GPU |
| C | none | `--ozone-platform=headless` + the flags of B | – | – | – | GPU process exits with code 139 one second after start, before any other job used the GPU |
| D | Xvfb | `--use-gl=angle --use-angle=gl-egl --ignore-gpu-blocklist` | ANGLE on Mesa llvmpipe (software) | – | – | software; the city did not finish loading within 210 s |
| E | Xvfb | `--use-gl=egl --ignore-gpu-blocklist` | – | – | – | the server never answered (readiness step failed) |
| X1 | headless NVIDIA Xorg | `--use-gl=angle --use-angle=gl --ignore-gpu-blocklist` | ANGLE on OpenGL 4.5, RTX 5060 | 0.091 / 0.109 | 0.217 / 0.220 | works on the GPU, fastest |
| X2 | headless NVIDIA Xorg | the flags of B | – | – | – | "Can't create buffer -- unsupported config" (GBM), no WebGL context |

Variant C is close to, but not the same as, the route in Valdrada's `docs/install.md` (`--headless
--ozone-platform=headless --use-angle=vulkan --enable-features=Vulkan`); that exact command was not run, and nothing
here was run outside the container.

GPU load during X1, from `nvidia-smi` sampled every 2 s over the whole batch (server start, two benchmark runs, a
20-frame quick start): mean utilisation 29% and mean power 45 W over the 93 of 131 samples where utilisation was above
zero.

## Benchmark output, variant X1 (run 1, run 2)

```
server 0.1.0 at [1280, 720] on ANGLE (NVIDIA Corporation, NVIDIA GeForce RTX 5060/PCIe/SSE2, OpenGL 4.5.0)
setup                                            s/tick  x real time
no sensors                                        0.091         0.55
pole rgb                                          0.101         0.50   (60 frames)
pole rgb + semantic + instance                    0.217         0.23   (180 frames)
  ... with amodal boxes (8)                       0.279         0.18   (180 frames)
pole rgb + semantic + instance, dashcam rgb       0.280         0.18   (240 frames)
depth only                                        0.157         0.32   (60 frames)

setup                                            s/tick  x real time
no sensors                                        0.109         0.46
pole rgb                                          0.119         0.42   (60 frames)
pole rgb + semantic + instance                    0.220         0.23   (180 frames)
  ... with amodal boxes (8)                       0.283         0.18   (180 frames)
pole rgb + semantic + instance, dashcam rgb       0.274         0.18   (240 frames)
depth only                                        0.163         0.31   (60 frames)
```

## Benchmark output, variant B (run 1, run 2)

```
server 0.1.0 at [1280, 720] on ANGLE (NVIDIA, Vulkan 1.4.325 (NVIDIA NVIDIA GeForce RTX 5060 (0x00002D05)), NVIDIA)
setup                                            s/tick  x real time
no sensors                                        0.130         0.39
pole rgb                                          0.206         0.24   (60 frames)
pole rgb + semantic + instance                    0.334         0.15   (180 frames)
  ... with amodal boxes (8)                       0.397         0.13   (180 frames)
pole rgb + semantic + instance, dashcam rgb       0.405         0.12   (240 frames)
depth only                                        0.256         0.20   (60 frames)

setup                                            s/tick  x real time
no sensors                                        0.125         0.40
pole rgb                                          0.213         0.23   (60 frames)
pole rgb + semantic + instance                    0.335         0.15   (180 frames)
  ... with amodal boxes (8)                       0.395         0.13   (180 frames)
pole rgb + semantic + instance, dashcam rgb       0.406         0.12   (240 frames)
depth only                                        0.258         0.19   (60 frames)
```
