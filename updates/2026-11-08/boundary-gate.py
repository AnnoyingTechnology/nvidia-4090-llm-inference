"""Verify exact prompt-plus-output capacity with synthetic token IDs."""
import argparse,hashlib,json,time,urllib.request
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--api',required=True);ap.add_argument('--total',type=int,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
p={'model':'qwen3.8-27b','prompt':[42]*(a.total-8),'max_tokens':8,'min_tokens':8,'ignore_eos':True,'temperature':0,'cache_salt':'boundary-20261108-dense'}
t=time.monotonic()
with urllib.request.urlopen(urllib.request.Request(a.api+'/v1/completions',data=json.dumps(p).encode(),headers={'Content-Type':'application/json'}),timeout=1200) as r:d=json.load(r)
u=d.get('usage',{});report={'request_sha256':hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),'usage':u,'elapsed_s':time.monotonic()-t,'pass':u.get('prompt_tokens')==a.total-8 and u.get('completion_tokens')==8};a.out.write_text(json.dumps(report,indent=2));print(report['pass']);raise SystemExit(0 if report['pass'] else 1)
