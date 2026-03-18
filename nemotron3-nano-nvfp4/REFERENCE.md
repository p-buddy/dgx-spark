# Running Nemotron Super on DGX Spark with Claude Code

## Introduction

NVIDIA's Nemotron 3 Super family represents a new class of reasoning-capable small models that punch well above their weight. The star of this setup is `cybermotaz/nemotron3-nano-nvfp4-w4a16` — a 4-bit quantized variant of Nemotron 3 Nano optimized specifically for the DGX Spark's GB10 Grace Blackwell chip. It runs at FP4 precision for weights and FP8 for the KV cache, squeezing maximum throughput out of unified memory while preserving reasoning quality.

What makes this model interesting for a local coding assistant is its dual capability: it uses a DeepSeek-R1-style chain-of-thought reasoning pass before answering, and it speaks Qwen3-style tool calls natively. This means it can plan, reason, write code, and call tools — all locally, all on a device that fits on your desk.

In this guide we will:

- Run Nemotron Super on DGX Spark using a purpose-built vLLM Docker image
- Benchmark its token throughput and time-to-first-token
- Set up a LiteLLM proxy to route requests intelligently
- Configure Claude Code to use the local model for fast coding tasks and Claude Sonnet 4.6 for planning

## Prerequisites

- NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)
- Docker with NVIDIA Container Toolkit installed
- A Hugging Face account (to download the model weights)
- An Anthropic API key (for Claude Sonnet 4.6)
- Python 3.10+ with pip

---

## Step 1: Run Nemotron Super with vLLM

NVIDIA's community has published a DGX Spark-specific vLLM image (`avarok/vllm-dgx-spark`) that includes all the necessary FlashInfer kernels and Blackwell optimizations. Start the server with:

```bash
docker run --rm -it --gpus all --ipc=host -p 8000:8000 \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/vllm-dgx-spark:v11 \
  serve cybermotaz/nemotron3-nano-nvfp4-w4a16 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser deepseek_r1
```

> The first run will download ~8 GB of weights to `~/.cache/huggingface`. Subsequent starts are fast.

### Verify the server is up

```bash
curl http://localhost:8000/v1/models
```

Expected response:

```json
{
  "data": [{
    "id": "cybermotaz/nemotron3-nano-nvfp4-w4a16",
    "max_model_len": 131072
  }]
}
```

---

## Step 2: Benchmark Token Throughput

Before integrating anything, it is worth knowing exactly what you are working with. Here is a streaming benchmark that measures both time-to-first-token (TTFT) and sustained generation speed:

```python
import requests, time, json

def benchmark(name, url, model, headers, prompt, max_tokens=1024):
    start = time.time()
    resp = requests.post(url, headers=headers, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True
    }, stream=True)

    ttft = None
    tokens = 0
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            text = chunk["choices"][0].get("delta", {}).get("content") or ""
            if text:
                if ttft is None:
                    ttft = time.time() - start
                tokens += 1

    total = time.time() - start
    tps = tokens / (total - ttft) if ttft else 0
    print(f"{name}")
    print(f"  TTFT:       {(ttft or 0)*1000:.0f} ms")
    print(f"  Tokens:     {tokens}")
    print(f"  Tokens/sec: {tps:.1f} tok/s\n")

PROMPT = "Write a Python class implementing a binary search tree with insert, delete, search, inorder traversal, and height methods. Include type hints and docstrings."
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer dummy"}

benchmark("Nemotron Super (local)", "http://localhost:8000/v1/chat/completions",
          "cybermotaz/nemotron3-nano-nvfp4-w4a16", HEADERS, PROMPT)
```

### Results on DGX Spark

| Model | TTFT | Tokens/sec |
|---|---|---|
| Nemotron Super (local) | ~800 ms | ~68 tok/s |
| Claude Sonnet 4.6 (API) | ~300 ms | ~19 tok/s |

The local model is ~3.6× faster at raw generation. The higher TTFT on the local model comes from its DeepSeek-R1-style reasoning pass — it thinks through the problem before producing output. The Anthropic API wins on latency to the first token because it skips that reasoning step by default.

This makes the two models naturally complementary:

- **Nemotron** — high-throughput code generation, long completions, fast iteration
- **Sonnet 4.6** — planning, architecture decisions, and nuanced reasoning where quality matters more than speed

---

## Step 3: Install LiteLLM as a Translation Proxy

Claude Code speaks the Anthropic API format. vLLM speaks the OpenAI API format. LiteLLM bridges the two, acting as a local router that translates between them and lets you point Claude Code at a single endpoint.

```bash
pip install 'litellm[proxy]'
```

Create `~/litellm_config.yaml`:

```yaml
model_list:
  # Haiku requests → local Nemotron on DGX Spark
  - model_name: claude-haiku-4-5-20251001
    litellm_params:
      model: openai/cybermotaz/nemotron3-nano-nvfp4-w4a16
      api_base: http://localhost:8000/v1
      api_key: "dummy"

  # Sonnet 4.6 requests → real Anthropic API
  - model_name: claude-sonnet-4-6
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: "YOUR_ANTHROPIC_API_KEY"

litellm_settings:
  drop_params: true
  set_verbose: false
```

> **Why map Nemotron to `claude-haiku`?**
> Claude Code uses different models for different tasks internally. By mapping Nemotron to the `claude-haiku-4-5-20251001` slot, it automatically becomes the model used for fast, high-frequency operations — inline completions, short edits, tool call parsing — while Sonnet 4.6 handles the heavier planning work. You get the best of both without any manual routing logic.

Start the proxy:

```bash
litellm --config ~/litellm_config.yaml --port 4000
```

Test both routes:

```bash
# Should respond from local Nemotron
curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer dummy" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-haiku-4-5-20251001","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'

# Should respond from Anthropic
curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer dummy" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'
```

---

## Step 4: Keep LiteLLM Running with systemd

Register LiteLLM as a user systemd service so it survives reboots:

```bash
mkdir -p ~/.config/systemd/user
```

Create `~/.config/systemd/user/litellm.service`:

```ini
[Unit]
Description=LiteLLM Proxy (routes Haiku to local vLLM, Sonnet 4.6 to Anthropic)
After=network.target

[Service]
ExecStart=/home/YOUR_USERNAME/.local/bin/litellm \
  --config /home/YOUR_USERNAME/litellm_config.yaml \
  --port 4000
Restart=on-failure
RestartSec=5
StandardOutput=append:/home/YOUR_USERNAME/litellm.log
StandardError=append:/home/YOUR_USERNAME/litellm.log

[Install]
WantedBy=default.target
```

Enable and start it:

```bash
systemctl --user daemon-reload
systemctl --user enable litellm
systemctl --user start litellm
systemctl --user status litellm
```

Logs are written to `~/litellm.log`.

---

## Step 5: Configure Claude Code

Edit `~/.claude/settings.json`:

```json
{
  "model": "claude-sonnet-4-6",
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:4000"
  }
}
```

That's all. Claude Code now:

- Uses **Claude Sonnet 4.6** (via Anthropic API) as the primary model for planning, reasoning, and architecture decisions
- Routes **Haiku** (fast/short) tasks to Nemotron Super on your DGX Spark at ~68 tok/s
- Falls back gracefully if the vLLM server is not running — Haiku requests will fail loudly, so you know to restart it

---

## Summary

| Component | Role |
|---|---|
| vLLM + Nemotron Super | Local inference at ~68 tok/s for fast coding tasks |
| LiteLLM proxy | Translates Anthropic ↔ OpenAI formats, routes by model name |
| Claude Sonnet 4.6 | Cloud model for planning and complex reasoning |
| Claude Code | Orchestrates everything via a single `ANTHROPIC_BASE_URL` |

Running a local model at 68 tok/s on a device the size of a Mac mini while routing planning tasks to the best cloud model is exactly the kind of hybrid setup that makes local AI actually practical. The DGX Spark's GB10 chip was built for exactly this — and with vLLM, LiteLLM, and Claude Code, the wiring is surprisingly straightforward.
