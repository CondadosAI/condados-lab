# Valdrada v0.5.2 server from source: Node 22 + Electron runtime deps + Xvfb, GPU through the NVIDIA container toolkit.
FROM node:22-bookworm
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      xvfb xauth libnss3 libatk1.0-0 libatk-bridge2.0-0 libgtk-3-0 libgbm1 libasound2 libxss1 libdrm2 \
      libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libcups2 libpango-1.0-0 libvulkan1 vulkan-tools \
      libegl1 libgl1 libgles2 mesa-utils python3 python3-dev python3-pip python3-venv python3-numpy ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages -q "huggingface_hub>=1.0"
ENV NVIDIA_DRIVER_CAPABILITIES=all
