# NVIDIA RTX 4090 local inference

Qwen3.8-27B on one 24 GiB RTX 4090 at 280 W, with vision and automatic prefix
caching. The selected capacity profile uses vLLM 0.32.0 on CUDA 12.9, DFlash2 k=7,
KVarN K4V2 KV and dense recurrent-state retention. Its supported server limit is
**262,136 tokens**, reserving eight DFlash lookahead positions inside the native
window. The owner accepted that reserve on 2026-11-08. OpenCode retains 245,760
input tokens plus an 8,192-token output reserve.

The selected fixture gives about **158 tok/s decode** and **2,265 tok/s at 32K
cold prefill**. The reproducible improvement is warm-prefix latency:
**0.95 to 0.20 seconds** on the 34K fixture. The earlier 137-to-158 (+15.2%)
decode comparison is not a dependable engine speedup: a restarted baseline reached
169 tok/s with higher DFlash acceptance. See the
[qualification and attribution record](updates/2026-11-08/README.md).

## Model and source

The selected target and calibrated assets are unchanged:

- `ababaka/Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound`, revision
  `c20530baefe3e77ccfc6891c2b50cce7ea28bf1e`;
- fast output-head/MTP assets at `124c14e7e8c7d2f5402933b9af368e772a9fcf0c`;
- DFlash2 W4A16 at `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`.

The update exports HyperQwen commit `6ab9bd05cb84b538a6595d634f3e337ac8a80856`
and uses the official vLLM CUDA 12.9 wheel. The retained base image comes from
upstream `dfee877366ff0db341d5d685784154f17b3a2f64`. The R550 host driver and
280 W efficiency policy remain unchanged. The 0.85 GiB vision tower stays in
pinned host RAM; image computation still runs on the GPU.

The service remains `qwen3.8-27b` on unauthenticated `0.0.0.0:19622` for a trusted
LAN. Do not forward the port through the Internet edge. Neither model service
is enabled at boot; the deployment adds no cross-service stopping or GPU gates.

## Build and run

```bash
git clone https://github.com/AnnoyingTechnology/nvidia-4090-llm-inference
cd nvidia-4090-llm-inference
bash updates/2026-11-08/build.sh
cp profiles/max.env .env
docker compose up -d --no-build
python3 bench/ulmus_validate.py --benchmark --profile max-032 --prefill-target 32768
```

`compose.override.yaml` selects the new runtime and a separate compiler cache.
The existing preparation helper provisions the declared Huihui checkpoint and
pinned fast/drafter assets. Docker Compose must support `!reset` (qualified with
5.7.0). Use the update build script rather than Compose's build command.

Latency measurements do not poll GPU power. Optional `--sample-power` requests
separate 1 Hz sampling. The follow-up 10 Hz on/off diagnostic did not explain the
decode discrepancy, but that cadence is unnecessary for these latency tests.

If the installed user unit owns the service, stop/start that unit instead of
mixing a manual Compose invocation with systemd. Models, caches and results remain
under this project. The prior image and private configuration backup are retained.

## Profiles and historical results

`max` is the selected 262,136-token KVarN capacity profile; `max-032` retains the
same versioned settings. `max-027` preserves the previous 262,144-token settings.
For rollback, stop the user unit, move the installed override aside, restore the
saved `.env` (or `profiles/max-027.env`), then start the unit again. The base Compose
file still selects vLLM 0.27.1.

The September BF16 `fast` benchmark at 65,536 context reached 178.72 tok/s decode
and 2,298.5 tok/s at 32K prefill. The September full-window KVarN profile reached
136.47 and 2,229.5 tok/s respectively. These remain historical measurements of
their original runtime, not the current capacity-service comparison. The original
setup and full context are preserved in [the September commit](https://github.com/AnnoyingTechnology/nvidia-4090-llm-inference/tree/5d36a06).

The `long`, `mtp-long`, `huge` and `fast` tiers and optional FA2/PTH plugins were
not requalified by this update. Use the base Compose file explicitly for those
historical experiments. Original result files remain under `results/`.
