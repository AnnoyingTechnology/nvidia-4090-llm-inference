"""Summarize the paired local update evidence; refuse incomplete gates."""
import argparse,hashlib,json,re,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
read=lambda n:json.loads((a.evidence/n).read_text())
def perf(names):
 reports=[read(n) for n in names];rows=[r for d in reports for r in d['benchmark']['decode_rows']]
 assert len(rows)==6 and all(r['completion_tokens']==512 and r['prompt_tokens']==565 and r['cached_tokens']==0 for r in rows)
 assert all(d['pass'] and d['benchmark']['prefill']['prompt_tokens']==32817 and d['benchmark']['prefill']['cached_tokens']==0 for d in reports)
 return {'fresh_starts':2,'decode_median_tok_s':statistics.median(r['decode_tok_s'] for r in rows),'per_start_decode_medians':[d['benchmark']['decode_tok_s_median'] for d in reports],'prefill_median_tok_s':statistics.median(d['benchmark']['prefill']['prefill_tok_s'] for d in reports),'decode_board_power_median_w':statistics.median(d['benchmark']['decode_power_w_median'] for d in reports),'warm_cache_median_s':statistics.median(d['prefix_cache']['warm_s'] for d in reports),'warm_cached_tokens':[d['prefix_cache']['warm_cached_tokens'] for d in reports]}
old=perf(['baseline-max.json','baseline-max-boot2.json']);new=perf(['candidate-dense-max.json','candidate-dense-max-boot2.json'])
x,y=read('baseline-code.json'),read('candidate-code.json')
assert x['source_sha256']==y['source_sha256'] and x['grader_image']==y['grader_image'] and x['evaluator_sha256']==y['evaluator_sha256']
assert len(x['controls'])==len(y['controls'])==30
cases=[]
for b,c in zip(x['results'],y['results']):
 assert b['id']==c['id'] and b['request_sha256']==c['request_sha256']
 assert b['response']['choices'][0]['finish_reason']==c['response']['choices'][0]['finish_reason']=='stop'
 cases.append({'id':b['id'],'request_sha256':b['request_sha256'],'baseline':b['verdict']['pass'],'candidate':c['verdict']['pass']})
assert len(cases)==30
for q in (x,y):
 assert all(c['positive']['pass'] and not c['negative']['pass'] for c in q['controls'].values())
images=read('candidate-max-images.json');boundary=read('candidate-dense-boundary.json');assert images['pass'] and boundary['pass'] and boundary['usage']['total_tokens']==262136
log=(a.evidence/'candidate-residue-sweep.log').read_text();rows=re.findall(r'residue=\s*(\d+) ok cov=([0-9.]+).*sha=([0-9a-f]+)',log)
assert len(rows)==128 and len({int(r[0]) for r in rows})==128 and 'DONE: 0 broken of 128 residues' in log
assert 'SUMMARY 12 / 12 passed' in (a.evidence/'candidate-dense-api-smoke.txt').read_text()
manifest=read('build-manifest.json');assert x['runtime']['Image']==manifest['baseline_image'] and y['runtime']['Image']==manifest['candidate_image']
report={'observed_date':'2026-11-08','baseline':old,'selected':new,'decode_gain_pct':100*(new['decode_median_tok_s']/old['decode_median_tok_s']-1),'prefill_gain_pct':100*(new['prefill_median_tok_s']/old['prefill_median_tok_s']-1),'quality':{'baseline':x['summary'],'candidate':y['summary'],'source_sha256':x['source_sha256'],'grader_image':x['grader_image'],'positive_and_negative_controls':30,'gains':[c['id'] for c in cases if not c['baseline'] and c['candidate']],'losses':[c['id'] for c in cases if c['baseline'] and not c['candidate']],'cases':cases},'api_checks':{'passed':12,'total':12},'maximum_images':images,'boundary':boundary,'cache_alignment':{'tested_residues':128,'broken':0,'unique_answer_hashes':len({r[2] for r in rows})},'context':manifest['context'],'scope':'Capacity profile on one RTX 4090; same target and drafter files, same upgraded CPU/RAM and 280 W limit. Other profiles and broad quality parity are not certified.','attribution':'Matched complete runtime A/B, with dense retention on the selected candidate. Individual kernel contributions were not isolated. The sparse-default diagnostic measured 142.10 tok/s and a 1.94 s warm-cache request; selected dense retention reused 34048 of 34231 prompt tokens. Do not add these deltas.'}
a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('decode_gain_pct','prefill_gain_pct','quality','cache_alignment')},indent=2))
