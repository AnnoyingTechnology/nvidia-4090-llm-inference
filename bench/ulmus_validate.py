#!/usr/bin/env python3
"""Validate Ulmus vision/cache behavior and measure one-GPU latency.

The script runs on the host against the loopback-only vLLM endpoint.  It uses
only the Python standard library and samples board power through nvidia-smi.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
import zlib


# Match the p565/g512 decode and p8221/g1 prefill fixtures used for the dual
# RTX 3090 article measurements.  The target is approximate before the chat
# template is applied; with Qwen's tokenizer these become 565 and 8,221 prompt
# tokens respectively.
FILLER = (
    "The RTX 3090 has 24 GB of GDDR6X and 82 streaming multiprocessors. "
    "Memory bandwidth is 936 GB/s, which is what decode is bound by. "
)


def make_prompt(target_tokens: int) -> str:
    repeats = max(1, round(target_tokens / 44))
    return (
        FILLER * repeats
        + "\n\nWrite a detailed technical explanation of why memory bandwidth, "
        "not compute, limits single-stream decoding on this hardware. Be thorough."
    )


def headers() -> dict[str, str]:
    result = {"Content-Type": "application/json"}
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        result["Authorization"] = f"Bearer {api_key}"
    return result


def post_json(url: str, payload: dict, timeout: int = 1200) -> tuple[dict, float]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers()
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode(errors="replace")) from exc
    return result, time.monotonic() - started


def cached_tokens(result: dict) -> int:
    return int(
        result.get("usage", {})
        .get("prompt_tokens_details", {})
        .get("cached_tokens", 0)
        or 0
    )


def png_data_url() -> str:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", checksum)
        )

    width, height = 64, 32
    row = b"\x00" + (b"\xff\x00\x00" * 32) + (b"\x00\x00\xff" * 32)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(row * height))
    png += chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def vision_canary(api: str, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": png_data_url()}},
                    {
                        "type": "text",
                        "text": (
                            "Name the color on the left and the color on the right. "
                            "Reply exactly: LEFT, RIGHT"
                        ),
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    result, elapsed = post_json(api + "/chat/completions", payload)
    answer = (result["choices"][0]["message"].get("content") or "").strip()
    return {"pass": answer == "RED, BLUE", "answer": answer, "elapsed_s": elapsed}


def cache_canary(api: str, model: str) -> dict:
    stable = "".join(
        f"Record {index:05d}: alpha beta gamma delta epsilon zeta eta theta.\n"
        for index in range(1800)
    )
    salt = "ulmus-cache-" + uuid.uuid4().hex
    messages = [
        {"role": "system", "content": "Follow the user's output format literally."},
        {"role": "user", "content": stable + "\nReply exactly: CACHE-READY"},
    ]

    def ask(current_messages: list[dict]) -> tuple[dict, float, str]:
        result, elapsed = post_json(
            api + "/chat/completions",
            {
                "model": model,
                "messages": current_messages,
                "max_tokens": 32,
                "temperature": 0,
                "cache_salt": salt,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        content = (result["choices"][0]["message"].get("content") or "").strip()
        return result, elapsed, content

    cold, cold_s, cold_answer = ask(messages)
    warm, warm_s, warm_answer = ask(messages)
    continued_messages = messages + [
        {"role": "assistant", "content": warm_answer},
        {"role": "user", "content": "Reply exactly: CACHE-CONTINUED"},
    ]
    continued, continued_s, continued_answer = ask(continued_messages)
    warm_cached = cached_tokens(warm)
    continued_cached = cached_tokens(continued)
    passed = (
        cold_answer == "CACHE-READY"
        and warm_answer == "CACHE-READY"
        and continued_answer == "CACHE-CONTINUED"
        and warm_cached > 0
        and continued_cached > 0
    )
    return {
        "pass": passed,
        "cold_s": cold_s,
        "warm_s": warm_s,
        "continued_s": continued_s,
        "cold_cached_tokens": cached_tokens(cold),
        "warm_cached_tokens": warm_cached,
        "continued_cached_tokens": continued_cached,
        "prompt_tokens": warm.get("usage", {}).get("prompt_tokens"),
        "answers": [cold_answer, warm_answer, continued_answer],
    }


class PowerSampler:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "/usr/bin/nvidia-smi",
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                self.samples.append((time.monotonic(), float(output.strip().splitlines()[0])))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(self.interval)

    def __enter__(self) -> "PowerSampler":
        self.thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)

    def mean_between(self, start: float, end: float) -> float | None:
        values = [power for instant, power in self.samples if start <= instant <= end]
        return statistics.mean(values) if values else None


def stream_measure(api: str, payload: dict, sampler: PowerSampler) -> dict:
    body = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(
        api + "/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers(),
    )
    started = time.monotonic()
    first = None
    last = None
    usage: dict = {}
    try:
        with urllib.request.urlopen(request, timeout=1200) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith("data: "):
                    continue
                event = line[6:]
                if event == "[DONE]":
                    break
                chunk = json.loads(event)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    now = time.monotonic()
                    first = now if first is None else first
                    last = now
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode(errors="replace")) from exc
    ended = time.monotonic()
    if first is None or last is None or not usage:
        raise RuntimeError("stream returned no timed content or final usage")
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    decode_s = max(last - first, 1e-9)
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "completion_tokens": completion_tokens,
        "ttft_s": first - started,
        "decode_s": decode_s,
        "total_s": ended - started,
        "prefill_tok_s": (prompt_tokens - cached) / max(first - started, 1e-9),
        "mean_prefill_board_power_w": sampler.mean_between(started, first),
        "decode_tok_s": max(completion_tokens - 1, 0) / decode_s,
        "mean_board_power_w": sampler.mean_between(first, last),
    }


def benchmark(api: str, model: str, prefill_target: int = 8192) -> dict:
    common = {
        "model": model,
        "temperature": 0,
        "seed": 4242,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    decode_prompt = make_prompt(512)
    prefill_prompt = make_prompt(prefill_target)

    with PowerSampler() as sampler:
        stream_measure(
            api,
            {
                **common,
                "messages": [{"role": "user", "content": "Explain RAID-Z2 in detail."}],
                "max_tokens": 64,
                "cache_salt": "warmup-" + uuid.uuid4().hex,
            },
            sampler,
        )
        decode_rows = []
        for _ in range(3):
            decode_rows.append(
                stream_measure(
                    api,
                    {
                        **common,
                        "messages": [{"role": "user", "content": decode_prompt}],
                        "max_tokens": 512,
                        "cache_salt": "decode-" + uuid.uuid4().hex,
                    },
                    sampler,
                )
            )
        prefill = stream_measure(
            api,
            {
                **common,
                "messages": [{"role": "user", "content": prefill_prompt}],
                "max_tokens": 1,
                "cache_salt": "prefill-" + uuid.uuid4().hex,
            },
            sampler,
        )

    return {
        "decode_rows": decode_rows,
        "decode_tok_s_median": statistics.median(row["decode_tok_s"] for row in decode_rows),
        "decode_power_w_median": (
            statistics.median(power_values)
            if (power_values := [
                row["mean_board_power_w"]
                for row in decode_rows
                if row["mean_board_power_w"] is not None
            ])
            else None
        ),
        "prefill": prefill,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:19622/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--profile", default="unknown")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--prefill-target", type=int, default=8192)
    args = parser.parse_args()

    report = {
        "profile": args.profile,
        "vision": vision_canary(args.api, args.model),
        "prefix_cache": cache_canary(args.api, args.model),
    }
    if args.benchmark:
        report["benchmark"] = benchmark(
            args.api, args.model, prefill_target=args.prefill_target
        )
    report["pass"] = report["vision"]["pass"] and report["prefix_cache"]["pass"]
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
