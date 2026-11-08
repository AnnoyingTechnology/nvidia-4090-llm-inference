"""Exercise the declared maximum image size, then the long-prefix cache path."""
import argparse,base64,json,struct,sys,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'bench'))
from ulmus_validate import post_json,cache_canary

def png(swapped):
 def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
 colors=(b'\xff\x00\x00',b'\x00\x00\xff')
 if swapped:colors=colors[::-1]
 row=b'\0'+colors[0]*1024+colors[1]*1024
 data=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',2048,1024,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(row*1024))+chunk(b'IEND',b'')
 return 'data:image/png;base64,'+base64.b64encode(data).decode()

ap=argparse.ArgumentParser();ap.add_argument('--api',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
report={'image_pixels':2097152,'images':[]}
for swapped in (False,True):
 p={'model':'qwen3.8-27b','messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':png(swapped)}},{'type':'text','text':'Name the color on the left and the color on the right. Reply exactly: LEFT, RIGHT'}]}],'temperature':0,'max_tokens':32,'chat_template_kwargs':{'enable_thinking':False}}
 d,t=post_json(a.api+'/chat/completions',p);answer=(d['choices'][0]['message'].get('content') or '').strip()
 report['images'].append({'pass':answer==('BLUE, RED' if swapped else 'RED, BLUE'),'answer':answer,'elapsed_s':t,'usage':d.get('usage')});a.out.write_text(json.dumps(report,indent=2))
report['cache_after_images']=cache_canary(a.api,'qwen3.8-27b');report['pass']=all(x['pass'] for x in report['images']) and report['cache_after_images']['pass'];a.out.write_text(json.dumps(report,indent=2));print(report['pass'])
raise SystemExit(0 if report['pass'] else 1)
