"""Record public runtime and artifact identities without prompts or credentials."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
models={}
for label,name in [('target','Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound-fast'),('drafter','Qwen3.8-27B-DFlash2-W4A16')]:
 d=root/'models'/name;idx=d/'model.safetensors.index.json'
 files=set(json.loads(idx.read_text())['weight_map'].values()) if idx.exists() else {'model.safetensors'}
 files.update(f for f in ('config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json','mtp_draft_vocab_ids.pt') if (d/f).exists())
 artifacts={}
 for filename in sorted(files):
  p=d/filename;h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
  artifacts[filename]={'sha256':h.hexdigest(),'bytes':p.stat().st_size,'modified_unix_s':p.stat().st_mtime}
 models[label]={'directory':name,'artifacts':artifacts}
packages=json.loads(subprocess.check_output(['docker','exec','ulmus-27b-update-eval','/app/venv/bin/python','-m','pip','list','--format=json'],text=True))
image=lambda tag:subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'],text=True).strip()
report={'hyperqwen_commit':'6ab9bd05cb84b538a6595d634f3e337ac8a80856','baseline_image':image('ulmus/qwen38-vllm:0.27.1-cu129'),'candidate_image':image('ulmus/qwen38-vllm:0.32.0-cu129-eval'),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,power.limit','--format=csv,noheader'],text=True).strip(),'packages':packages,'models':models,'context':{'target_server_limit':262136,'draft_native_positions':262144,'lookahead_positions':8,'opencode_input':245760,'opencode_output_reserve':8192},'target_revisions':{'huihui':'c20530baefe3e77ccfc6891c2b50cce7ea28bf1e','fast_head':'124c14e7e8c7d2f5402933b9af368e772a9fcf0c','dflash':'4d30ec736ffc6b8688dc2ae2b502d9b48bdec279'}}
Path(__file__).with_name('build-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
