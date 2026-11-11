"""Real-operation deployment probes, not benchmark or perceptual-quality scores.

Run with the installed edit_cpu interpreter after starting the selected services.
The default creates synthetic English speech via the newly deployed TTS; the
optional --source must contain the exact built-in diagnostic sentence. Full
qualification calls TTS before voice conversion and Chinese diarization.
Requests, failures and waveforms are retained. Success certifies operation only.
"""
import argparse,base64,hashlib,io,json,sys,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parent))
import stack
from check_asr import qualify as qualify_asr
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,help='Optional recording of the built-in English diagnostic sentence');p.add_argument('--tools',type=Path);p.add_argument('services',nargs='*');a=p.parse_args()
if a.tools and a.services:p.error('Use either --tools or service names')
if a.tools:
 import yaml
 from x_editor.tools.selection import ToolSelection
 a.services=ToolSelection.model_validate(yaml.safe_load(a.tools.read_text())).deployment_services()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);cfg,m,_,expand=stack.context(a.data)
external=cfg.get('external',{})
stack.require_external_bindings(m,stack.selected(m,a.services or ['all'],external),external)
source=a.source.resolve() if a.source else out/'source.wav'
text='The next train will leave at six fifteen.'
zh='下一班火车将在六点十五分出发。'

def wave(raw):
 if isinstance(raw,bytes):return raw
 for k in ['audio','output_audio']:
  v=raw.get(k)
  if isinstance(v,dict) and v.get('data'):return base64.b64decode(v['data'])
 for k in ['output_audio_path','output_path','audio_path','target_output_path','target_audio_path']:
  if raw.get(k) and Path(raw[k]).is_file():return Path(raw[k]).read_bytes()
 raise ValueError('No delivered audio in response')

if not source.exists():
 payload={'text':text,'speaker':'Ryan','language':'English','seed':0}
 req=urllib.request.Request(expand('{url:tts}')+'/synthesize_custom',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=900) as response:
  raw=response.read(); ctype=response.headers.get('Content-Type','')
 value=json.loads(raw) if 'json' in ctype else raw
 source.write_bytes(wave(value))
 (out/'source-provenance.json').write_text(json.dumps({'origin':'deployed_qwen_tts','text':text,'speaker':'Ryan','language':'English','seed':0,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2))
audio={'data':base64.b64encode(source.read_bytes()).decode(),'format':'wav'}

requests={
'edit_prosody':('/execute',{'audio':audio,'source_text':text,'instruction_st':'The next train will leave at <rate, factor=1.25>six fifteen</rate>.','evaluate':False},'audio'),
'edit_emotion':('/execute',{'audio':audio,'source_text':text,'instruction_st':'<emo type="happy" alpha="0.83">'+text+'</emo>','evaluate':False,'max_candidates':1},'audio'),
'edit_pause':('/execute',{'audio':audio,'source_text':text,'instruction_st':'The next train<pause act="ins" level="2"/> will leave at six fifteen.','evaluate':False},'audio'),
'smoother':('/smooth',{'audio':audio,'text':text,'points':[{'mode':'smooth_only','position_sec':1.0,'regen_margin_sec':0.15}],'aligner_backend':'qwen3','f5_seed':0},'audio'),
'assembly':('/assemble',{'source_audio':audio,'target_audio':audio,'edit_tagged_text':'<target>'+text+'</target>','aligner_backend':'qwen3'},'audio'),
'pitch':('/shift',{'audio':audio,'method':'psola','semitones':2.0},'audio'),
'rate':('/stretch',{'audio':audio,'method':'wsola','rate':1.25},'audio'),
'volume':('/normalize',{'audio':audio,'method':'peak','target_db':-12},'audio'),
'vad':('/detect',{'audio':audio},'object'),
'asr':('/transcribe',{'audio':audio,'language':'English'},'text'),
'aligner_backend':('/align',{'audio':audio,'text':text,'word_level':True},'object'),
'aligner':('/align',{'audio':audio,'text':text,'word_level':True,'backend':'qwen3'},'object'),
'planner':('/v1/chat/completions',{'model':'Qwen/Qwen3.8-27B','messages':[{'role':'user','content':'Return only the result of 2+2.'}],'temperature':0,'max_tokens':32,'chat_template_kwargs':{'enable_thinking':False}},'chat'),
'omni':('/v1/chat/completions',{'model':'Qwen3Omni-Instruct','messages':[{'role':'user','content':[{'type':'input_audio','input_audio':audio},{'type':'text','text':'Transcribe the speech in this audio.'}]}],'temperature':0,'max_tokens':128},'chat'),
'edit_text':('/execute',{'audio':audio,'source_text':text,'instruction_st':'The next <sub targ="bus">train</sub> will leave at six fifteen.','evaluate':False,'seed':0},'audio'),
'tts':('/synthesize_custom',{'text':zh,'speaker':'Vivian','language':'Chinese','seed':0},'audio'),
'tts_clone':('/synthesize_clone',{'text':'The next bus will leave soon.','language':'English','ref_audio':str(source),'ref_text':text,'seed':0},'audio'),
'seed_vc':('/convert',{'source_audio':str(source),'reference_audio':str(out/'tts.wav'),'seed':0,'diffusion_steps':30},'audio'),
'indextts':('/synthesize',{'text':text,'spk_audio_prompt':str(source),'emo_vector':[.7,0,0,0,0,0,0,0],'seed':0},'audio'),
'flexsed':('/detect',{'audio_path':str(source),'events':['Speech'],'save_heatmap_png':False,'output_dir':str(out/'flexsed-assets')},'object'),
'sedicow':('/transcribe_speaker',{'audio_path':str(source),'qwen_text':text,'language':'English'},'segments'),
'diarization':('/diarize',{'audio_path':str(source)},'segments'),
'moss':('/transcribe_diarize',{'audio_path':str(out/'tts.wav'),'qwen_text':zh,'language':'Chinese'},'segments'),
'auk':('/edit',{'input_audio_path':str(source),'instruction':'Make the speaker whisper, preserving the spoken words.','output_path':str(out/'auk.wav')},'audio'),
'enhancement':('/enhance',{'input_audio_path':str(source),'output_path':str(out/'enhanced.wav')},'audio'),
'separation':('/separate',{'input_audio_path':str(source),'output_dir':str(out/'stems'),'output_path':str(out/'separated.wav')},'audio'),
}
for operation,label,instruction in [
 ('text_edit','text','The next <sub targ="bus">train</sub> will leave at six fifteen.'),
 ('pitch_rate_edit','pitch','<pitch semitones="2">'+text+'</pitch>'),
 ('pitch_rate_edit','rate','<rate factor="1.25">'+text+'</rate>'),
 ('emotion_style_edit','emotion','<emo type="happy" alpha="0.83">'+text+'</emo>'),
 ('pause_edit','pause','The next train<pause act="ins" level="2"/> will leave at six fifteen.'),
]:
 requests['dots_tts_edit.'+label]=('/execute',{'operation':operation,'audio':audio,'source_text':text,'instruction_st':instruction},'audio')
rows=[]
selected=stack.selected(m,a.services or ['all'],external)
unprobed=[name for name in selected if name not in requests and name!='dots_tts_edit']
services=[name for name in requests if name.split('.')[0] in selected]
if any(name in services for name in ('seed_vc','moss')) and 'tts' not in services:services.insert(0,'tts')
if not services:raise ValueError('No direct operation probes selected; health is not qualification')
for probe in services:
 service=probe.split('.')[0]
 route,payload,kind=requests[probe];start=time.monotonic();row={'service':service,'probe':probe,'kind':kind,'passed':False}
 (out/(probe+'-request.json')).write_text(json.dumps(payload,ensure_ascii=False))
 try:
  if service=='asr':
   context_report=qualify_asr(expand('{url:asr}'),source,out/'asr-context')
   if not context_report['passed']:raise RuntimeError('ASR context operation failed; see asr-context/report.json')
  req=urllib.request.Request(expand('{url:'+service+'}')+route,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=900) as response:
   raw=response.read();ctype=response.headers.get('Content-Type','')
  value=json.loads(raw) if 'json' in ctype else raw
  (out/(probe+('-response.json' if isinstance(value,dict) else '-response.wav'))).write_bytes(raw)
  if isinstance(value,dict) and (value.get('error') or value.get('success') is False):raise ValueError(str(value)[:1000])
  if kind=='audio':
   data=wave(value);x,sr=sf.read(io.BytesIO(data));assert x.size and np.isfinite(x).all() and np.max(np.abs(x))>1e-6
   path=out/(probe+'.wav')
   if path.exists():assert path.read_bytes()==data
   else:path.write_bytes(data)
   row.update(frames=len(x),sample_rate=sr,sha256=hashlib.sha256(data).hexdigest(),artifact=str(path))
  elif kind=='chat':assert value['choices'][0]['message']['content'].strip()
  elif kind=='text':assert value.get('text','').strip()
  elif kind=='segments':assert value.get('segments') or value.get('speaker_segments'),list(value)
  else:assert isinstance(value,dict) and value
  row['passed']=True
 except Exception as e:
  if isinstance(e,urllib.error.HTTPError):row['response_error']=e.read().decode(errors='replace')[:5000]
  row['error']=repr(e);print(service,'FAILED',repr(e),flush=True)
 finally:
  row['seconds']=time.monotonic()-start;rows.append(row);stack.write(out/'summary.json',{'requests':rows,'all_passed':len(rows)==len(services) and all(r['passed'] for r in rows),'expected_probes':services,'services_without_direct_probe':unprobed,'all_capabilities_verified':False,'human_quality_verified':False});print(probe,row['passed'],round(row['seconds'],2),flush=True)
for service in {row['service'] for row in rows}:
 rec=a.data/'services'/f'{service}.json'
 if rec.exists():
  record=stack.read(rec)
  if stack.pid_identity(record['pid'])==record['pid_identity']:
   record.update(inference_verified=all(row['passed'] for row in rows if row['service']==service),qualification=str(out/'summary.json'));stack.write(rec,record)
sys.exit(0 if all(r['passed'] for r in rows) else 1)
