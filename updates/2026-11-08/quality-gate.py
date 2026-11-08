"""Paired frozen function gate; generated code runs only in the existing isolated grader."""
import argparse,datetime,hashlib,json,re,subprocess,time,urllib.request,uuid
from pathlib import Path

def grade(problem,code):
 name='ulmus-27b-code-eval-'+uuid.uuid4().hex[:12]
 cmd=['docker','run','--rm','--init','-i','--name',name,'--network','none','--read-only','--cap-drop','ALL','--tmpfs','/tmp:rw,noexec,nosuid,size=64m','--security-opt','no-new-privileges','--memory','512m','--memory-swap','512m','--cpus','1','--pids-limit','64','--user','1000:1000','ulmus/eval:lcb-28fef95']
 try:
  r=subprocess.run(cmd,input=json.dumps({'problem':problem,'code':code}),text=True,capture_output=True,timeout=180,check=True)
  return json.loads(r.stdout)
 except subprocess.TimeoutExpired:
  subprocess.run(['docker','rm','--force',name],capture_output=True,timeout=15)
  raise

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--api',required=True);ap.add_argument('--container',required=True);ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 inspect=lambda:json.loads(subprocess.check_output(['docker','inspect',a.container],text=True))[0]
 initial=inspect();identity={k:initial[k] for k in ('Id','Image')}
 rows=[json.loads(x) for x in a.cases.read_text().splitlines()]
 report={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runtime':identity,'source_sha256':hashlib.sha256(a.cases.read_bytes()).hexdigest(),'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'grader_image':subprocess.check_output(['docker','image','inspect','ulmus/eval:lcb-28fef95','--format','{{.Id}}'],text=True).strip(),'controls':{},'results':[]}
 for c in rows:
  assert c['kind']=='code'
  good=grade(c['problem'],c['control']);bad=grade(c['problem'],'def solve(*args): return "incorrect"')
  assert good['pass'] and not bad['pass'],c['id']
  report['controls'][c['id']]={'positive':good,'negative':bad}
 a.out.write_text(json.dumps(report,indent=2))
 for c in rows:
  current=inspect();assert all(current[k]==v for k,v in identity.items()) and current['State']['Running']
  payload={'model':'qwen3.8-27b','messages':[{'role':'user','content':c['prompt']}],'max_tokens':8192,'temperature':0,'seed':4242,'chat_template_kwargs':{'enable_thinking':False},'cache_salt':'frozen-'+c['id']}
  started=time.monotonic()
  with urllib.request.urlopen(urllib.request.Request(a.api+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=1200) as r: response=json.load(r)
  choice=response['choices'][0];text=choice['message'].get('content') or '';blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S)
  verdict=grade(c['problem'],blocks[0]) if choice['finish_reason']=='stop' and blocks else {'pass':False,'error':'unfinished or missing code'}
  report['results'].append({'id':c['id'],'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),'response':response,'verdict':verdict,'elapsed_s':time.monotonic()-started})
  report['summary']={'passed':sum(r['verdict']['pass'] for r in report['results']),'completed':len(report['results']),'total':len(rows),'all_natural':all(r['response']['choices'][0]['finish_reason']=='stop' for r in report['results'])}
  a.out.write_text(json.dumps(report,indent=2));print(c['id'],verdict['pass'],report['summary'],flush=True)
 current=inspect();assert all(current[k]==v for k,v in identity.items())
if __name__=='__main__': main()
