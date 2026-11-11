"""Prepare the selected services from this source distribution and public pins.

This is a provisioning command, not an inference or benchmark qualification.
Use --phase plan first to inspect downloads, environments and GPU placement.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import stack
import pipeline
from client_envs import install_clients


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--env-root',type=Path)
    p.add_argument('--phase',choices=['plan','fetch','install','start','all'],default='plan')
    p.add_argument('--gpu',action='append',default=[])
    p.add_argument('--external',action='append',default=[],help='Explicit operator-installed NAME=http://host:port binding')
    p.add_argument('--port-offset',type=int,default=0)
    p.add_argument('--bind-address')
    p.add_argument('--tools',type=Path,help='Use the same tool selection file as runtime.yaml')
    p.add_argument('services',nargs='*',help='Service names or groups: cpu speech models experts editing optional all')
    a=p.parse_args();data=a.data.resolve();data.mkdir(parents=True,exist_ok=True)
    if bool(a.tools) == bool(a.services):
        p.error('Specify either --tools FILE or service names, not both')
    selection = None
    if a.tools:
        sys.path.insert(0, str(stack.ROOT))
        import yaml
        from x_editor.tools.selection import ToolSelection
        selection = ToolSelection.model_validate(yaml.safe_load(a.tools.read_text()))
        a.services = selection.deployment_services()
        os.environ['X_EDITOR_EDIT_BACKEND'] = selection.edit_backend
    m=stack.read(stack.MANIFEST)
    external=dict(value.split('=',1) for value in a.external)
    if (data/'stack_config.json').exists():
        cfg,_,_,_=stack.context(data)
        if a.external and external!=cfg.get('external',{}):
            raise ValueError('Different external bindings: use a new deployment')
        external=cfg.get('external',{})
    stack.validate_bindings(m, external)
    chosen=stack.selected(m,a.services,external)
    managed=[name for name in chosen if name not in external and not m['services'][name].get('external_only')]
    repos=sorted({r for n in managed for r in m['services'][n].get('repositories',[])})
    models=sorted({r for n in managed for r in m['services'][n]['models']})
    assets=sorted({r for n in managed for r in m['services'][n].get('service_assets',[])})
    environments={name:m['services'][name]['environment'] for name in managed}
    pipeline_needed=any(m['services'][n].get('pipeline_required') for n in managed)
    pipeline_tools=sorted({t for n in managed for t in m['services'][n].get('pipeline_tools',[])})
    plan={'services':chosen,'repositories':repos,'models':models,'service_assets':assets,'environments':environments,
          'tool_selection':selection.model_dump(mode='json') if selection else None,
          'pipeline_required':pipeline_needed,'pipeline_tools':pipeline_tools,'external':external,
          'operator_setup_required':{name:stack.external_setup_message(name) for name in chosen if m['services'][name].get('external_only') and name not in external},
          'default_gpu_placement':{n:m['services'][n]['gpu'] for n in chosen},'gpu_overrides':a.gpu}
    print(json.dumps(plan,indent=2),flush=True)
    if a.phase=='plan':return
    stack.require_external_bindings(m, chosen, external)
    uv=shutil.which('uv')
    if not uv:raise RuntimeError('Install uv==0.11.28 first')
    def run(args):subprocess.run(args,cwd=stack.ROOT,check=True)
    if a.phase in ['fetch','all']:
        if pipeline_needed:pipeline.inspect(pipeline_tools,fetch=True)
        if repos:run([sys.executable,str(stack.ROOT/'deployment/dependencies.py'),'repos',*repos])
        if assets:run([sys.executable,str(stack.ROOT/'deployment/dependencies.py'),'assets',*assets])
        if models:run([uv,'run','--no-project','--with','huggingface-hub==0.36.2','python',str(stack.ROOT/'deployment/dependencies.py'),'models',*models,'--cache',str(data/'cache/huggingface/hub'),'--output',str(data/'assets.json')])
        stack.write(data/'provisioning_plan.json',plan)
    if a.phase in ['install','start','all']:
        if not (data/'stack_config.json').exists():
            config=argparse.Namespace(assets=data/'assets.json' if models else None,env_root=a.env_root,gpu=a.gpu,external=[f'{name}={url}' for name,url in external.items()],port_offset=a.port_offset,bind_address=a.bind_address)
            stack.configure(data,config)
        else:
            cfg,_,_,_=stack.context(data)
            if a.env_root and str(a.env_root.resolve())!=cfg['environments']:raise ValueError('Different environment root: use a new deployment')
            if a.gpu and dict(v.split('=',1) for v in a.gpu)!=cfg['gpus']:raise ValueError('Different GPU allocation: use a new deployment')
            if a.bind_address and a.bind_address!=cfg.get('bind_address','127.0.0.1'):
                raise ValueError('Different bind address: use a new deployment')
        if a.phase in ['install','all']:
            cfg,_,_,_=stack.context(data)
            catalog_ids = selection.catalog_ids() if selection else {
                project.name for project in (stack.ROOT/'x_editor/tools/catalog').iterdir() if project.is_dir()}
            install_clients(Path(cfg['environments'])/'clients', catalog_ids)
            stack.install(data,chosen)
        if a.phase in ['start','all']:stack.start(data,chosen)
        stack.export_environment(data)


if __name__=='__main__':main()
