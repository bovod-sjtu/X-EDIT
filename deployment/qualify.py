"""Run real service requests and save evidence; health is not qualification.

Run with the installed edit_cpu interpreter. DSP mode creates a diagnostic tone.
Speech mode requires your own recording, its transcript and a tagged edit.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import time
import urllib.request

import numpy as np
import soundfile as sf
import stack


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['dsp','speech'],required=True)
    p.add_argument('--audio',type=Path);p.add_argument('--transcript');p.add_argument('--tagged');p.add_argument('--expected-transcript')
    a=p.parse_args();data=a.data.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg,m,_,expand=stack.context(data);checks=[]
    targets=['edit_text'] if a.mode=='speech' else ['rate','volume']
    stack.require_external_bindings(m,stack.selected(m,targets,cfg.get('external',{})),cfg.get('external',{}))
    def call(name,service,route,payload):
        url=expand('{url:'+service+'}')+route;t=time.monotonic()
        with urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=600) as response:
            raw=response.read();content_type=response.headers.get('Content-Type','')
        if 'json' in content_type:
            value=json.loads(raw);(out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2))
        else:
            value=raw;(out/(name+'.wav')).write_bytes(raw)
        checks.append({'request':name,'service':service,'seconds':time.monotonic()-t,'response_sha256':hashlib.sha256(raw).hexdigest()})
        return value
    def audio_bytes(value):
        if isinstance(value,bytes):return value
        audio=value.get('audio')
        if isinstance(audio,dict) and audio.get('data'):return base64.b64decode(audio['data'])
        for key in ['output_audio_path','audio_path','output_path']:
            if value.get(key):return Path(value[key]).read_bytes()
        raise ValueError('Response has no supported audio delivery; inspect saved JSON')
    def wav_payload(raw):return {'data':base64.b64encode(raw).decode(),'format':'wav'}
    success=False
    try:
        if a.mode=='dsp':
            sr=24000;x=.1*np.sin(2*np.pi*440*np.arange(sr*2)/sr);b=io.BytesIO();sf.write(b,x,sr,format='WAV',subtype='PCM_16');audio=wav_payload(b.getvalue())
            raw=audio_bytes(call('rate','rate','/stretch',{'audio':audio,'method':'wsola','rate':1.25}));y,ysr=sf.read(io.BytesIO(raw));ratio=(len(y)/ysr)/(len(x)/sr)
            if abs(ratio-.8)>.002:raise AssertionError(f'Unexpected duration ratio: {ratio}')
            raw=audio_bytes(call('volume','volume','/normalize',{'audio':audio,'method':'peak','target_db':-12}));y,ysr=sf.read(io.BytesIO(raw));peak=float(20*np.log10(np.max(np.abs(y))))
            if abs(peak+12)>.1:raise AssertionError(f'Unexpected peak dBFS: {peak}')
            checks.append({'duration_ratio':ratio,'peak_dbfs':peak})
        else:
            if not all([a.audio,a.transcript,a.tagged]):raise ValueError('speech requires --audio --transcript --tagged')
            source=a.audio.resolve(strict=True);audio=wav_payload(source.read_bytes())
            before=call('source_transcript','asr','/transcribe',{'audio':audio})
            call('alignment','aligner_backend','/align',{'audio':audio,'text':a.transcript,'word_level':True})
            result=call('text_edit','edit_text','/execute',{'audio':audio,'source_text':a.transcript,'instruction_st':a.tagged,'evaluate':False,'seed':0})
            raw=audio_bytes(result);y,sr=sf.read(io.BytesIO(raw))
            if not y.size or not np.isfinite(y).all():raise AssertionError('Invalid delivered waveform')
            (out/'edited.wav').write_bytes(raw)
            after=call('edited_transcript','asr','/transcribe',{'audio':wav_payload(raw)})
            checks.append({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(raw).hexdigest(),'output_frames':len(y),'sample_rate':sr,'transcript_before':before.get('text'),'transcript_after':after.get('text')})
            if a.expected_transcript:
                norm=lambda text:re.sub(r'[^\w]','',text.casefold())
                if norm(after.get('text',''))!=norm(a.expected_transcript):raise AssertionError('Independent ASR differs from expected transcript; inspect audio and evidence')
        success=True
    finally:
        stack.write(out/'qualification.json',{'passed':success,'mode':a.mode,'checks':checks,'human_quality_verified':False,'all_capabilities_verified':False})
    # Only mark the owned services that actually handled successful requests.
    for name in {c['service'] for c in checks if 'service' in c}:
        path=data/'services'/f'{name}.json'
        if path.exists():
            rec=stack.read(path)
            if stack.pid_identity(rec['pid'])==rec['pid_identity']:
                rec.update(inference_verified=True,qualification=str(out/'qualification.json'));stack.write(path,rec)
    print(out/'qualification.json')


if __name__=='__main__':main()
