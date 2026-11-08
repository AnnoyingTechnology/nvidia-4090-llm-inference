# November 2026 runtime candidate

The candidate is qualified but **not selected for the service** while the recorded
requirement for exactly 262,144 server tokens remains in force. Its supported
limit is eight tokens lower. The original service and profile remain available.

The qualified capacity candidate uses HyperQwen commit
`6ab9bd05cb84b538a6595d634f3e337ac8a80856`, vLLM 0.32.0's official CUDA 12.9 wheel,
and dense prefix-state retention. The target, calibrated head, DFlash2 sidecar,
KVarN K4V2 cache, vision offload, one request slot and 280 W limit are retained.
No driver update is required on the qualified R550 installation.

The software A/B ran on the same upgraded CPU/RAM in both arms. It does not
attribute the software gain to the CPU replacement.

| Matched fixture | Previous runtime | Selected runtime |
|---|---:|---:|
| p565/g512 decode, two starts and six requests per arm | 137.20 tok/s | 158.05 tok/s (+15.2%) |
| p32,817 uncached prefill | 2,254 tok/s | 2,265 tok/s (effectively unchanged) |
| 34,231-token prefix fixture, first warm repeat | 0.951 s | 0.196 s |
| Cached prompt tokens on that repeat | 32,640 | 34,048 |
| Frozen 30 DevOps/Python functions, greedy, thinking off | 26/30 | 27/30 |

All coding responses stopped naturally within the identical 8,192-token cap.
There was one gained case and no lost cases. Positive and negative controls
passed for all 30 functions on the same isolated grader image. This is a narrow
operational coding gate, not a claim of broad quality parity.

The selected profile also passed maximum-size 2,097,152-pixel images followed by
long-prefix reuse, all 128 prompt-length cache alignments, request-level API checks,
and 262,128 prompt tokens plus eight output tokens in 210.53 seconds.
The exact results and artifact hashes are in `summary.json` and `build-manifest.json`.

## Why the cache setting matters

The new launcher's default retains one recurrent-state snapshot in six to make
room for multiple long conversations. The default diagnostic reached 142.10 tok/s
and took 1.94 seconds on the warm-prefix fixture. This deployment keeps dense
retention with `PREFIX_RETENTION=` for its single-conversation use. Two fresh
starts of that profile reached 157.97 and 158.10 tok/s.

The selected warm repeat reprocesses 183 prompt tokens instead of the previous
1,591. That directly explains its reduced prompt-processing work. The decode
comparison covers the complete runtime and its KVarN/DFlash patches; individual
kernel contributions were not isolated. Do not add the diagnostic deltas.

## Context accounting

The new runtime rejects the previous 262,144 server limit: seven DFlash drafts
and their bonus position would exceed the sidecar's native positional capacity.
`profiles/max-032.env` reserves those eight positions and permits **262,136 server
tokens**. The model and drafter configurations are unchanged; their positional
capacity was not artificially extended.

OpenCode retains 245,760 input tokens and an 8,192-token output reserve. Its
253,952-token client contract is unchanged. The near-boundary test is a capacity
check, not a throughput comparison with a smaller-context profile.

## Reproduce

After accepting the eight-position reserve, run from the repository root. If
the installed user unit owns the service, stop it before a manual Compose run;
do not mix the two lifecycle owners.

For a manual run:

```bash
bash updates/2026-11-08/build.sh
cp profiles/max-032.env .env
docker compose -f compose.yaml -f compose.update.yaml up -d --no-build
python3 bench/ulmus_validate.py --benchmark --profile max-032 --prefill-target 32768
```

For the Ulmus user unit, copy the accepted `compose.update.yaml` to
`compose.override.yaml` after stopping the unit, select `profiles/max-032.env`,
then start the unit normally. This is staged, not the current selection.

The script builds the existing CUDA 12.9 baseline image if absent, exports the
pinned HyperQwen source into a fresh build context, then builds the new image.
`compose.update.yaml` selects it explicitly and keeps its compiler cache separate under
`cache/runtime-032`. Docker Compose must support the `!reset` override tag
(qualified with Compose 5.7.0). Use the build script rather than Compose's build
command for this update.

The startup wrapper retains the pinned Huihui preparation helper and checks the
new installation before launching `/next/single-user/start_qwen.sh`. The upstream
full verifier rejects the deliberate unauthenticated LAN binding; this deployment
preserves its existing exposure contract. Model and tokenizer behavior is covered
by the retained qualification gates and artifact manifest. No upstream source
patch was added to suppress that verifier's policy.

Optional FA2/PTH plugins and other capacity tiers were not qualified by this update.
The older result tables remain historical observations of their original runtime.

## Roll back

Keep the previous `.env` in a private backup before selecting the new profile.
Stop the service, restore that file, and use the base Compose file explicitly:

```bash
docker compose stop qwen
# Restore the saved .env here.
docker compose -f compose.yaml up -d --no-build qwen
```

The base file still selects `ulmus/qwen38-vllm:0.27.1-cu129`. For the installed
systemd unit, move the installed `compose.override.yaml` into the dated rollback directory
before starting it again.
On Ulmus the candidate override is staged separately and the installed unit still
selects the original base configuration.
The previous image, models and compiler cache remain available.
