# Nemotron-3-Nano with llama.cpp — DGX Spark (GB10, sm_121)
# Base image: CUDA 12.8 devel on Ubuntu 22.04
FROM nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        build-essential \
        python3 \
        python3-pip \
        python3-venv \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Hugging Face CLI ──────────────────────────────────────────────────────────
RUN python3 -m venv /opt/hf-venv \
    && /opt/hf-venv/bin/pip install --upgrade pip \
    && /opt/hf-venv/bin/pip install -U "huggingface_hub[cli]"

ENV PATH="/opt/hf-venv/bin:${PATH}"

# ── Clone & build llama.cpp with CUDA + GB10 (sm_121) ────────────────────────
WORKDIR /opt
RUN git clone https://github.com/ggml-org/llama.cpp

WORKDIR /opt/llama.cpp
RUN mkdir build && cd build \
    && cmake .. \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="121" \
        -DLLAMA_CURL=OFF \
    && make -j$(nproc)

# ── Model download (build-time, optional) ────────────────────────────────────
# Set HF_TOKEN as a build arg if the repo requires authentication.
# Leave blank to skip download at build time and bind-mount at runtime instead.
ARG HF_TOKEN=""
ARG SKIP_MODEL_DOWNLOAD="false"

RUN if [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then \
        mkdir -p /models/nemotron3-gguf \
        && HF_TOKEN=${HF_TOKEN} hf download unsloth/Nemotron-3-Nano-30B-A3B-GGUF \
            Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \
            --local-dir /models/nemotron3-gguf; \
    fi

# ── Runtime defaults ─────────────────────────────────────────────────────────
ENV MODEL_PATH="/models/nemotron3-gguf/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf" \
    SERVER_HOST="0.0.0.0" \
    SERVER_PORT="30000" \
    N_GPU_LAYERS="99" \
    CTX_SIZE="8192" \
    THREADS="8"

EXPOSE 30000

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
