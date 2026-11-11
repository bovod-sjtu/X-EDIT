"""Verify real CUDA arithmetic in each installed backend environment.

Package imports and nvidia-smi alone do not prove kernels can execute. This uses
service GPU assignments and saves both passing and failing subprocess results.
It does not load model checkpoints or qualify their output quality.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import stack

PROBE = '''import json,torch
assert torch.cuda.is_available(), 'CUDA initialization failed'
x=torch.arange(16,device='cuda',dtype=torch.float32).reshape(4,4)
y=x@x.T
torch.cuda.synchronize()
assert torch.allclose(y.cpu(),x.cpu()@x.cpu().T)
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'kernel_verified':True}))
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('services',nargs='+')
    a=p.parse_args();cfg,m,_,_=stack.context(a.data);seen=set();rows=[]
    for name in stack.selected(m,a.services,cfg.get('external',{})):
        service=m['services'][name];env=service['environment']
        if service.get('gpu') is None or name in cfg.get('external',{}) or service.get('external_only'):
            continue
        spec=stack.service_spec(a.data,name,allow_missing=True)
        identity=(spec['command'][0],spec['env'].get('CUDA_VISIBLE_DEVICES'))
        if identity in seen:continue
        seen.add(identity)
        row={'environment':env,'representative_service':name,'passed':False}
        try:
            result=subprocess.run([spec['command'][0],'-c',PROBE],env=stack.runtime_env(spec),
                                  text=True,capture_output=True,timeout=90)
            row.update(passed=result.returncode==0,exit_code=result.returncode,
                       stdout=result.stdout,stderr=result.stderr)
        except (OSError,subprocess.TimeoutExpired) as error:
            row['error']=str(error)
        rows.append(row)
        stack.write(a.output,{'environments':rows,'all_passed':all(r['passed'] for r in rows)})
        print(env,row['passed'],flush=True)
    if not rows:
        raise ValueError('No GPU environments selected')
    return 0 if all(r['passed'] for r in rows) else 1

if __name__=='__main__':
    sys.exit(main())
