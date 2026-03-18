# Nemotron-3-Nano on DGX Spark — Docker Container

Runs **Nemotron-3-Nano-30B** via `llama.cpp` with full GPU offload on a DGX Spark (GB10, `sm_121`).  
Exposes an **OpenAI-compatible API** on port `30000`.

---

## Quick start

### Option A — Download model at build time (recommended)

```bash
# If the HF repo is gated, log in first:
export HF_TOKEN=hf_your_token_here

docker compose build
docker compose up -d
```

The model (~38 GB) is downloaded during `docker build` and cached in the
`nemotron-models` Docker volume.

### Option B — Skip download at build time, download on first run

```bash
export SKIP_MODEL_DOWNLOAD=true
export HF_TOKEN=hf_your_token_here   # required at runtime too

docker compose build
docker compose up -d
```

The model is downloaded the first time the container starts and is persisted in
the named volume, so subsequent starts are instant.

### Option C — Use a pre-downloaded model

If you already have the GGUF file on disk, bind-mount it:

```bash
# In docker-compose.yml, replace the named volume with:
# volumes:
#   - /path/to/your/models:/models
docker compose up -d
```

---

## Verify the server is running

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron",
    "messages": [{"role": "user", "content": "New York is a great city because..."}],
    "max_tokens": 100
  }'
```

---

## Environment variables

| Variable        | Default  | Description                                      |
|-----------------|----------|--------------------------------------------------|
| `HF_TOKEN`      | *(none)* | Hugging Face token (required for gated models)   |
| `CTX_SIZE`      | `8192`   | Context window; can be set up to `1048576` (1M)  |
| `N_GPU_LAYERS`  | `99`     | Layers offloaded to GPU (`99` = all)             |
| `THREADS`       | `8`      | CPU threads for non-GPU ops                      |
| `SERVER_PORT`   | `30000`  | Port the API listens on                          |

Override at runtime without rebuilding:

```bash
CTX_SIZE=131072 docker compose up -d
```

---

## Accessing from other machines on the local network

The server binds to `0.0.0.0:30000`.  Point any OpenAI-compatible client at:

```
http://<dgx-spark-ip>:30000/v1
```

---

## Cleanup

```bash
docker compose down
docker volume rm nemotron-dgx_nemotron-models   # removes the downloaded model
```
