# November 2026 selected runtime

Selected by the owner on 2026-11-08, including the eight-position DFlash reserve.
The supported server limit is 262,136 tokens. OpenCode's limits are unchanged.
The previous image and 262,144-token profile remain available for rollback.

The selected capacity service uses HyperQwen commit
`6ab9bd05cb84b538a6595d634f3e337ac8a80856`, vLLM 0.32.0's official CUDA 12.9 wheel,
and dense prefix-state retention. The target, calibrated head, DFlash2 sidecar,
KVarN K4V2 cache, vision offload, one request slot and 280 W limit are retained.
No driver update is required on the qualified R550 installation.

## Measurements and correction

The original two-start cohort measured 137.20 to 158.05 tok/s (+15.2%). That
number is retained as historical evidence, **not a dependable engine speedup**.
During attribution, the restarted old runtime reached 169.18 tok/s on the same
input, while the selected runtime remained at 158.27 tok/s. The generated output
hashes differ. This is a counterexample to a general +15% claim, rather than a
matched-output kernel comparison.

| Fixture | Previous runtime | Selected runtime |
|---|---:|---:|
| Original two-start decode cohort | 137.20 tok/s | 158.05 tok/s |
| Follow-up streamed decode, identical input | 169.18 tok/s | 158.27 tok/s |
| p32,817 uncached prefill | 2,254 tok/s | 2,265 tok/s (effectively unchanged) |
| 34,231-token prefix fixture, warm repeat | 0.951 s | 0.196 s |
| Cached prompt tokens on that repeat | 32,640 | 34,048 |
| Frozen 30 DevOps/Python functions, greedy, thinking off | 26/30 | 27/30 |

The original server log windows around decode report roughly 3.0–3.1 tokens per
DFlash verification step. The selected follow-up returns 512 tokens in 149 steps
(3.44 tokens/step). The restored old runtime instead needed 137 steps (3.74),
explaining its higher throughput. Observed generation time per verification step
is about 22.0 ms old and 21.7 ms new; those are whole-step wall times, not isolated
GPU kernel timings. Higher draft acceptance accounts for most of the original
gap. Its variation across runs prevents a universal upgrade-gain claim.

Both software arms used the same upgraded CPU/RAM, target/drafter files and 280 W
limit. Streaming versus non-streaming and 10 Hz board-power polling on/off did
not explain the restored baseline's higher rate. Latency benchmarks now leave
power sampling off by default; optional collection is 1 Hz.

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
starts of that profile reached 157.97 and 158.10 tok/s. These historical diagnostics
did not control generated-output hashes or step counts.

The selected warm repeat reprocesses 183 prompt tokens instead of the previous
1,591. That directly explains its reduced prompt-processing work. The original decode
comparison mixed different acceptance/output trajectories; individual kernel
contributions were not isolated. Do not add the diagnostic deltas.

## Context accounting

The new runtime rejects the previous 262,144 server limit: seven DFlash drafts
and their bonus position would exceed the sidecar's native positional capacity.
`profiles/max.env` reserves those eight positions and permits **262,136 server
tokens**. The model and drafter configurations are unchanged; their positional
capacity was not artificially extended.

OpenCode retains 245,760 input tokens and an 8,192-token output reserve. Its
253,952-token client contract is unchanged. The near-boundary test is a capacity
check, not a throughput comparison with a smaller-context profile.

## Reproduce

Run from the repository root. If
the installed user unit owns the service, stop it before a manual Compose run;
do not mix the two lifecycle owners.

For a manual run:

```bash
bash updates/2026-11-08/build.sh
cp profiles/max.env .env
docker compose up -d --no-build
python3 bench/ulmus_validate.py --benchmark --profile max-032 --prefill-target 32768
```

The Ulmus user unit selects `compose.override.yaml` through Compose
autodiscovery. Stop the unit before changing `.env`, then start it normally.
The existing unit and on-demand lifecycle are retained.

The script builds the existing CUDA 12.9 baseline image if absent, exports the
pinned HyperQwen source into a fresh build context, then builds the new image.
`compose.override.yaml` selects it automatically and keeps its compiler cache separate under
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
`profiles/max-027.env` also retains the previous 262,144-token profile.
Restore it when selecting the base image; the current `profiles/max.env` is the new profile.
The previous image, models and compiler cache remain available.
