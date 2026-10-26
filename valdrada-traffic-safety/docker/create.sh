#!/bin/bash
# Build the images and create the long-lived container the scripts exec into. Run from this folder (it is /f inside),
# with the Valdrada v0.5.2 source cloned into ./src (git clone -b v0.5.2 https://github.com/mkturkcan/valdrada src).
# The two NVIDIA Xorg modules come from the host driver; adjust the libglxserver version to yours (ls /usr/lib/nvidia/xorg).
set -euo pipefail
W=${VALDRADA_WORK:-$PWD}
docker build -t valdrada-env:v0.5.2 -f docker/Dockerfile docker
docker build -t valdrada-env:v0.5.2-xorg -f docker/Dockerfile.xorg docker
GLX=$(ls /usr/lib/nvidia/xorg/libglxserver_nvidia.so.* | head -1)
mkdir -p "$W/home" "$W/logs" "$W/out"
docker run -d --name valdrada-gpu --gpus all --shm-size 4g --user "$(id -u):$(id -g)" --network host \
  -v "$W:/f" -w /f/src \
  -v /usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json:ro \
  -v /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro \
  -v /usr/lib/xorg/modules/drivers/nvidia_drv.so:/usr/lib/xorg/modules/drivers/nvidia_drv.so:ro \
  -v "$GLX:/usr/lib/xorg/modules/extensions/libglxserver_nvidia.so:ro" \
  -e HOME=/f/home -e NVIDIA_DRIVER_CAPABILITIES=all -e HF_HOME=/f/home/.cache/huggingface \
  valdrada-env:v0.5.2-xorg sleep infinity
