"""Measure draft acceptance and request time on the matched decode fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bench'))
from ulmus_validate import make_prompt, post_json, vision_canary, cache_canary, PowerSampler

ap = argparse.ArgumentParser()
ap.add_argument('--phase', required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--stream', action='store_true')
ap.add_argument('--condition-caches', action='store_true')
ap.add_argument('--sample-power', action='store_true')
ap.add_argument('--power-interval', type=float, default=1.0)
a = ap.parse_args()
api = 'http://127.0.0.1:19622'
names = ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens')

def counters():
    with urllib.request.urlopen(api + '/metrics', timeout=10) as response:
        text = response.read().decode()
    result = {}
    for name in names:
        matches = re.findall(r'^vllm:spec_decode_' + name + r'_total\{[^\n]*\}\s+([0-9.eE+-]+)', text, re.M)
        assert len(matches) == 1, (name, matches)
        result[name] = float(matches[0])
    return result

common = {'model': 'qwen3.8-27b', 'temperature': 0, 'seed': 4242,
          'chat_template_kwargs': {'enable_thinking': False}}
conditioning = None
if a.condition_caches:
    conditioning = {'vision': vision_canary(api + '/v1', 'qwen3.8-27b'),
                    'prefix_cache': cache_canary(api + '/v1', 'qwen3.8-27b')}
    assert all(value['pass'] for value in conditioning.values())
sampler = PowerSampler(interval=a.power_interval) if a.sample_power else None
if sampler:
    sampler.__enter__()
post_json(api + '/v1/chat/completions', dict(common, messages=[{'role': 'user', 'content': 'Explain RAID-Z2 in detail.'}], max_tokens=64, cache_salt=a.phase + '-warmup'))
time.sleep(12)
before = counters()
rows = []
for index in range(3):
    payload = dict(common, messages=[{'role': 'user', 'content': make_prompt(512)}],
                   max_tokens=512, cache_salt=a.phase + '-' + str(index))
    stream_info = {}
    if a.stream:
        payload.update(stream=True, stream_options={'include_usage': True})
        started = time.monotonic()
        content = ''
        response = {}
        choice = {}
        first = last = None
        chunks = 0
        with urllib.request.urlopen(urllib.request.Request(api + '/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}), timeout=1200) as connection:
            for raw in connection:
                if not raw.startswith(b'data: '):
                    continue
                raw = raw[6:].strip()
                if raw == b'[DONE]':
                    break
                event = json.loads(raw)
                if event.get('usage'):
                    response['usage'] = event['usage']
                if event.get('metrics'):
                    response['metrics'] = event['metrics']
                for item in event.get('choices') or []:
                    if item.get('finish_reason'):
                        choice['finish_reason'] = item['finish_reason']
                    text = item.get('delta', {}).get('content') or ''
                    if text:
                        now = time.monotonic()
                        first = now if first is None else first
                        last = now
                        chunks += 1
                        content += text
        elapsed = time.monotonic() - started
        stream_info = {'content_chunks': chunks, 'ttft_s': first - started, 'decode_s': last - first,
                       'decode_tok_s': (response['usage']['completion_tokens'] - 1) / (last - first)}
    else:
        response, elapsed = post_json(api + '/v1/chat/completions', payload)
        choice = response['choices'][0]
        content = choice['message'].get('content') or ''
    rows.append({'elapsed_s': elapsed, 'usage': response['usage'],
                 'output_sha256': hashlib.sha256(content.encode()).hexdigest(),
                 'finish_reason': choice['finish_reason'], 'metrics': response.get('metrics'), **stream_info})
if sampler:
    sampler.__exit__()
time.sleep(12)
after = counters()
delta = {key: after[key] - before[key] for key in names}
report = {'phase': a.phase, 'streaming': a.stream, 'sample_power': a.sample_power,
          'power_samples': len(sampler.samples) if sampler else 0,
          'power_interval_s': a.power_interval if sampler else None,
          'conditioning': conditioning, 'rows': rows, 'spec_counters_delta': delta,
          'mean_accepted_per_step': delta['num_accepted_tokens'] / delta['num_drafts'],
          'mean_tokens_per_step': 1 + delta['num_accepted_tokens'] / delta['num_drafts']}
a.out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
