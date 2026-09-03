# NVIDIA RTX 4090 local inference

This repository records the quality-gated optimization of Qwen3.8-27B for one
24 GiB RTX 4090. The selected full-window service sustains **136.47 tok/s** with
vision, automatic prefix caching and the complete 262,144-token model context.
A matched BF16 benchmark profile reaches **178.72 tok/s** decode and **2,299
tok/s** on a 32,817-token cold prefill, versus 62.61 tok/s on the previous
llama.cpp setup.

The selected target is Huihui's abliterated model, quantized as
`ababaka/Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound` at revision
`c20530baefe3e77ccfc6891c2b50cce7ea28bf1e`.  The local fast variant uses the
already-qualified int4-GPTQ head/MTP assets at revision
`124c14e7e8c7d2f5402933b9af368e772a9fcf0c`; their source tensors are byte-for-byte
identical between the Huihui and stock checkpoints.  The DFlash2 W4A16 revision is
`4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`.  The stock target is retained only as
a rollback and historical benchmark baseline.

The implementation is based on commit
`dfee877366ff0db341d5d685784154f17b3a2f64` of
[`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090),
with a reproducible CUDA 12.9 image and an RTX 4090-specific qualified profile.

## Result

The service exposes `qwen3.8-27b` on unauthenticated `0.0.0.0:19622`. Keep it
on a trusted network.

| Selected setting | Value |
|---|---|
| GPU | One RTX 4090, 24 GiB, 280 W limit |
| Target | Huihui Qwen3.8-27B Abliterated W4A16 |
| Speculator | DFlash2 W4A16, k=7 |
| Context | 262,144 server; 245,760 input + 8,192 output for OpenCode |
| KV cache | KVarN K4V2, 272,781 reported tokens |
| Required features | Vision and automatic prefix caching |
| API | OpenAI compatible, port 19622, no key |

## Boundaries

- Do not use the upstream CUDA 13 prebuilt image while Ulmus runs driver
  `550.163.01`.  `docker/Dockerfile.cu129` pins the official vLLM 0.27.1
  CUDA 12.9 wheel instead.  It also gives the host R550 library precedence
  over CUDA's datacenter-only forward-compatibility shim; otherwise GeForce
  initialization fails with CUDA error 804.
- Do not change Ulmus's 280 W GPU limit for this campaign.
- Vision and automatic prefix caching are mandatory.  Every profile sets
  `VISION=1`, `VISION_OFFLOAD=1`, and `PREFIX_CACHE=1`.
  An Ulmus A/B with uncached 2,097,152-pixel images measured 1.033 s with the
  0.85 GiB tower offloaded and 0.984 s with it resident.  Keeping it resident
  left only 235 MiB free, reduced the retained decode fixture from 175.3 to
  169.3 tok/s, and the engine failed after the combined 32K/cache workload.
  The roughly 49 ms image penalty is therefore the stable overall trade.
  The Huihui and stock checkpoints' 333 vision tensors were compared directly
  and are byte-identical, so that A/B remains applicable after the target switch.
- The endpoint is deliberately published without an API key on all Ulmus
  interfaces (`0.0.0.0:19622`) for trusted-LAN use.  Do not forward this port
  through the Internet edge.
- `models/`, `cache/`, profiles, source, and results remain under this folder.

## Profiles

`compose.yaml` defaults `MODEL` to
`/app/models/Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound-fast`; an explicit
`MODEL` override is reserved for controlled A/B runs.

`max` is the deployed profile: DFlash2 k=7, KVarN K4V2 KV, one request slot,
and the full 262,144-token server context.  `fast` is the matched performance
benchmark with BF16 KV and a 65,536-token context.  `long` trades cold-prefill speed for a
131,072-token INT8 KV context.  `mtp-long` is the 150,000-token FP8-KV native
MTP control.  `huge` is the earlier 245,760-token KVarN profile.  Both KVarN
profiles use a lossy cache; `max` accepts that measured trade to keep the
model's complete native window available.

Select a profile and start the server:

```bash
git clone https://github.com/AnnoyingTechnology/nvidia-4090-llm-inference
cd nvidia-4090-llm-inference
cp profiles/max.env .env
sudo docker compose build
sudo docker compose up -d
sudo docker compose logs -f qwen
```

The first start downloads the pinned Huihui target, assembles its local fast
variant, fetches the DFlash2 sidecar, then compiles CUDA/Triton kernels.  The
build, models and compiled cache are persistent inside this folder.  The model
repository and immutable revision are declared in `compose.yaml`, so a fresh
models volume cannot silently fall back to the stock target.

Validate the non-negotiable features and collect the comparison cell from the
host after `/health` becomes ready:

```bash
python3 bench/ulmus_validate.py --benchmark --profile max --prefill-target 32768
```

The `fast` performance benchmark uses the same approximate p512/g512 text fixture as
the dual-3090 campaign (565 prompt tokens after Qwen's chat template), plus its
p8,221 cold-prefill fixture, so the resulting cells are directly comparable.

Run the test once to warm the stack, then keep the second run.  Stop before
changing profiles:

```bash
sudo docker compose down
cp profiles/fast.env .env
sudo docker compose up -d
```

`docker compose down` removes the container and private bridge only.  It does
not delete `models/`, `cache/`, the built image, or benchmark results.

## Qualified results (2026-09-03)

The selected Huihui target was measured after warmup on Ulmus's RTX 4090 at its
unchanged 280 W power limit.  The retained cells use a cold cache namespace for
each performance request.  The stock results remain in their original result files
for provenance; do not relabel them as Huihui measurements.

| Profile | KV / context | p565/g512 decode | Cold prefill | Vision | Prefix cache |
|---|---|---:|---:|---|---|
| `max` (active) | KVarN K4V2 / 262,144 | 136.47 tok/s | 2,230 tok/s at p32,817 | PASS | PASS, 32,640 tokens reused |
| `fast` (article benchmark) | BF16 / 65,536 | 178.72 tok/s | 2,299 tok/s at p32,817 | PASS | PASS, 33,600/34,231 tokens reused |

Median decode board power was 278.3 W in `max` and 278.8 W in the retained
`fast` article run.  The active `max` profile passed all 12 request-level checks
in `bench/api_smoke.py`.  DFlash2 k=5 was slower at 129.02 tok/s.  k=3 was slower
again at 124.03 tok/s and failed the prefix-cache canary, so k=7 remains selected.

For the article's comparable ~32K cell, `fast` processed a 32,817-token cold
prompt at 2,298.5 tok/s and 278.0 W.  The deployed `max` profile allocates a
272,781-token GPU KV pool and completed exactly 262,136 prompt tokens plus eight
forced output tokens—262,144 total—in 210.8 seconds.  The GPU held about
23,698 MiB during that request with roughly 513 MiB free, which is the intended
transient margin rather than unused capacity.  KVarN's 4/2-bit cache is lossy and
its deep-context decode is slower than BF16, but the earlier stock INT4 control
took 608.3 seconds at the same boundary.

Current machine-readable results are in
`results/huihui-fast-32k-qualified.json`, `results/huihui-max-32k-qualified.json`,
`results/huihui-max-exact-boundary.json`, and `results/huihui-max-api-smoke.txt`.
The k=3 and k=5 rejection evidence is retained alongside them.  The reusable
vision residency A/B remains `results/vision-offload-ab.json`.
