"""Install and launch an explicitly configured X·Edit service stack.

Uses only this checkout, newly installed environments, and declared model assets.
No cluster path discovery, system-site packages, or shared-service restarts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'deployment/stack.json'
EXTERNAL_POLICIES = json.loads((ROOT / 'x_editor/tools/external_services.json').read_text())


def external_setup_message(name):
    return EXTERNAL_POLICIES.get(name, {}).get('setup_message',
        f'Independently install and start {name}; see docs/DOTS_TTS_EDIT.md')


def validate_bindings(manifest, external):
    if set(external) - set(manifest['services']):
        raise ValueError('External bindings must name declared services')
    for name, endpoint in external.items():
        parsed = urlsplit(endpoint)
        parsed.port
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError(f'{name}: external service binding requires an HTTP(S) URL without credentials')


def require_external_bindings(manifest, chosen, external):
    validate_bindings(manifest, external)
    missing = [name for name in chosen if manifest['services'][name].get('external_only') and name not in external]
    if missing:
        raise ValueError('; '.join(external_setup_message(name) +
                         f' (operator-owned service required: --external {name}=http://host:port)' for name in missing))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def selected(manifest, names, external=()):
    chosen = []
    visiting = set()
    def add(name):
        if name in chosen:
            return
        if name in visiting:
            raise ValueError(f'Cyclic service dependency: {name}')
        visiting.add(name)
        if name not in external:
            for dep in manifest['services'][name]['depends_on']:
                add(dep)
        chosen.append(name)
        visiting.remove(name)
    for name in names:
        if name in manifest['services']:
            add(name)
        else:
            matches = [k for k,v in manifest['services'].items() if v['group'] == name or name == 'all']
            if not matches:
                raise ValueError(f'Unknown service/group: {name}')
            for key in matches:
                if manifest['services'][key].get('external_only') and manifest['services'][key]['group'] == 'optional' and key not in external:
                    continue
                add(key)
    return chosen


def configure(data, args):
    if (data/'stack_config.json').exists():
        raise FileExistsError('Use an existing deployment or a new data directory; configuration is immutable')
    manifest = read(MANIFEST)
    assets = read(args.assets) if args.assets else {}
    if 'models' in assets:
        assets = assets['models']
    overrides = dict(v.split('=',1) for v in args.gpu)
    external = dict(v.split('=',1) for v in args.external)
    validate_bindings(manifest, external)
    if set(overrides) - set(manifest['services']):
        raise ValueError('GPU override names must be service names')
    bind_address = getattr(args, 'bind_address', None) or '127.0.0.1'
    socket.inet_pton(socket.AF_INET, bind_address)
    cfg = {'root':str(ROOT),'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
           'assets':assets,'port_offset':args.port_offset,'gpus':overrides,
           'external':external,'environment_layout':'service','bind_address':bind_address,
           'environments':str(Path(args.env_root).resolve()) if args.env_root else str(data/'envs')}
    write(data/'stack_config.json',cfg)
    print(json.dumps({'configured':str(data),'model_count':len(assets)}))


def context(data):
    cfg = read(data/'stack_config.json'); manifest = read(MANIFEST)
    if cfg['root'] != str(ROOT) or cfg['manifest_sha256'] != hashlib.sha256(MANIFEST.read_bytes()).hexdigest():
        raise ValueError('Source/manifest changed; configure a new deployment to preserve provenance')
    ports = {k:v['port']+cfg['port_offset'] for k,v in manifest['services'].items()}
    if any(p<1024 or p>65535 for p in ports.values()) or len(set(ports.values())) != len(ports):
        raise ValueError('Invalid or colliding service ports')
    def expand(value, allow_missing=False):
        def replacement(match):
            key=match.group(1)
            simple={'root':str(ROOT),'data':str(data),'envs':cfg['environments']}
            if key in simple:return simple[key]
            kind,name=key.split(':',1)
            if kind=='port':return str(ports[name])
            if kind=='python':return str(environment_path(cfg,manifest,name)/'bin/python')
            if kind=='url':
                endpoint = cfg.get('external',{}).get(name)
                if manifest['services'][name].get('external_only') and not endpoint:
                    if allow_missing:return f'<operator-installed service required: {name}>'
                    require_external_bindings(manifest, [name], cfg.get('external',{}))
                host=cfg.get('bind_address','127.0.0.1')
                if host=='0.0.0.0':host='127.0.0.1'
                return endpoint or f'http://{host}:{ports[name]}'
            if kind=='model':
                model=cfg['assets'].get(name)
                path=model.get('path') if isinstance(model,dict) else model
                if not path:
                    if allow_missing:return f'<missing model: {name}>'
                    raise ValueError(f'Missing model {name}; provide download receipt/model map at configure')
                return str(Path(path).resolve())
            raise ValueError(f'Unknown expansion {key}')
        return re.sub(r'\{([^{}]+)\}',replacement,value)
    return cfg,manifest,ports,expand


def environment_path(cfg, manifest, name):
    base = Path(cfg['environments'])
    if cfg.get('environment_layout') == 'service':
        return base / 'services' / name
    return base / manifest['services'][name]['environment']


def service_spec(data,name,allow_missing=False):
    cfg,m,ports,expand=context(data);s=m['services'][name]
    if s.get('external_only') or name in cfg.get('external', {}):
        endpoint = cfg.get('external', {}).get(name)
        if not allow_missing:
            require_external_bindings(m, [name], cfg.get('external', {}))
        return {'name':name, 'external':True, 'owned':False, 'url':endpoint,
                'setup_required':not bool(endpoint), 'inference_verified':False,
                'message':external_setup_message(name) if not endpoint else None}
    python=environment_path(cfg,m,name)/'bin/python'
    ex=lambda v:expand(v,allow_missing)
    command=[str(python),*[ex(x) for x in s['entry']+s['args']]]
    bind_address=cfg.get('bind_address','127.0.0.1')
    if '--host' in command:
        command[command.index('--host')+1]=bind_address
    probe_host='127.0.0.1' if bind_address=='0.0.0.0' else bind_address
    env={'PYTHONNOUSERSITE':'1','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
         'HF_HOME':str(data/'cache/huggingface'),'HF_HUB_CACHE':str(data/'cache/huggingface/hub'),
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'CHECKPOINTS_DIR':str(data/'models'),'OMP_NUM_THREADS':'4',
         'PYTHONPATH':os.pathsep.join(ex(x) for x in s['pythonpath'])}
    if s.get('gpu') is not None:env['CUDA_VISIBLE_DEVICES']=cfg['gpus'].get(name,s['gpu'])
    env.update({k:ex(v) for k,v in s['env'].items()})
    return {'name':name,'command':command,'env':env,'cwd':ex(s.get('cwd','{root}/third_party/dots_tts_edit')),
            'url':f'http://{probe_host}:{ports[name]}','bind_address':bind_address,'health_path':s['health_path'],'timeout':s['timeout_seconds'],
            'models':s['models'],'inference_verified':False}


def install(data,names):
    cfg,m,_,_=context(data);uv=shutil.which('uv')
    if not uv:raise RuntimeError('Install uv==0.11.28 first')
    # Service names take precedence over same-named environments (e.g. asr).
    # An explicitly external service must not install its local GPU stack.
    if names and all(n in m['environments'] and n not in m['services'] for n in names):
        services = [name for name, service in m['services'].items()
                    if service.get('environment') in names
                    and name not in cfg.get('external', {}) and not service.get('external_only')]
    else:
        external = cfg.get('external', {})
        require_external_bindings(m, selected(m, names, external), external)
        services = [name for name in selected(m, names, external)
                    if name not in external and not m['services'][name].get('external_only')]
    environments = {environment_path(cfg,m,name):m['services'][name]['environment'] for name in services}
    for env, name in sorted(environments.items()):
        spec=m['environments'][name];receipt=env/'install_receipt.json'
        env.parent.mkdir(parents=True,exist_ok=True)
        minimum_gb=2 if name=='edit_cpu' else 20
        if not env.exists() and shutil.disk_usage(env.parent).free < minimum_gb*1024**3:
            raise RuntimeError(f'{name} needs at least {minimum_gb} GiB free staging space; choose a larger --env-root')
        receipt.unlink(missing_ok=True)
        install_env=dict(os.environ)
        for key in ['PYTHONPATH','PYTHONHOME','UV_PROJECT_ENVIRONMENT','VIRTUAL_ENV']:
            install_env.pop(key,None)
        subprocess.run([uv,'venv','--allow-existing','--python',spec['python'],str(env)],cwd=data,env=install_env,check=True)
        cmd=[uv,'pip','sync','--python',str(env/'bin/python'),str(ROOT/spec['lock'])]
        if '+cu128' in (ROOT/spec['lock']).read_text():
            # Prefer PyPI for equal-version wheels, avoiding PyTorch-index
            # redirects to a separate NVIDIA host. Keep best-match resolution:
            # pins without a local suffix can still require the tested +cu128
            # builds (torchao/torchcodec). First-match would change those builds.
            cmd += ['--index','https://pypi.org/simple',
                    '--default-index','https://download.pytorch.org/whl/cu128',
                    '--index-strategy','unsafe-best-match']
        subprocess.run(cmd,cwd=data,env=install_env,check=True)
        for source in spec.get('editable_sources',[]):
            subprocess.run([uv,'pip','install','--python',str(env/'bin/python'),'--no-deps','-e',str((ROOT/source).resolve(strict=True))],cwd=data,env=install_env,check=True)
        check=subprocess.run([uv,'pip','check','--python',str(env/'bin/python')],env=install_env,text=True,capture_output=True)
        diagnostics=check.stdout+check.stderr
        if check.returncode:
            # The serving snapshot deliberately retains its tested versions.
            # Accept only specifically documented metadata conflicts, never a
            # general --ignore-dependencies switch or a silent package upgrade.
            conflicts=[line for line in diagnostics.splitlines() if line.startswith('The package ')]
            permitted=spec.get('metadata_exceptions',[])
            if not conflicts or any(not any(re.fullmatch(rule['pattern'],line) for rule in permitted) for line in conflicts):
                raise RuntimeError(diagnostics)
            if spec.get('metadata_probe'):
                subprocess.run([str(env/'bin/python'),str(ROOT/spec['metadata_probe']),
                                '--output',str(env/'metadata_qualification.json')],
                               env=install_env,check=True)
            print(diagnostics,flush=True)
        packages=subprocess.check_output([uv,'pip','freeze','--python',str(env/'bin/python')],text=True,env=install_env)
        write(receipt,{'environment':name,'lock_sha256':hashlib.sha256((ROOT/spec['lock']).read_bytes()).hexdigest(),'packages':packages,'dependency_check':diagnostics,'metadata_exceptions':spec.get('metadata_exceptions',[]) if check.returncode else []})


def pid_identity(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError):return None


def runtime_env(spec):
    env=dict(os.environ)
    for k in ['PYTHONHOME','PYTHONPATH','UV_PROJECT_ENVIRONMENT','VIRTUAL_ENV']:env.pop(k,None)
    env.update(spec['env']);return env


def link_asset(source, target):
    source=Path(source).resolve(strict=True);target=Path(target)
    if target.exists() or target.is_symlink():
        if target.resolve()!=source:
            raise RuntimeError(f'Refusing to replace asset at {target}')
        return
    target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source)


def prepare_assets(data, name):
    cfg,m,_,expand=context(data)
    if m['services'][name].get('pipeline_required'):
        from pipeline import inspect
        inspect(m['services'][name].get('pipeline_tools', []))
    groups = m['services'][name].get('service_assets', [])
    for row in read(ROOT/'third_party/manifest.json').get('service_assets', []):
        if Path(row['file']).parts[1] in groups:
            path = ROOT/'third_party'/row['file']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Missing or changed service asset; run dependencies.py assets '
                                 + Path(row['file']).parts[1])
    pins={r['repo_id']:r['revision'] for r in read(ROOT/'third_party/manifest.json')['models']}
    for model in m['services'][name]['models']:
        supplied=cfg['assets'].get(model)
        if isinstance(supplied,dict) and supplied.get('revision',pins[model])!=pins[model]:
            raise ValueError(f'{model}: provided revision differs from the release pin')
        source=Path(expand('{model:'+model+'}')).resolve(strict=True)
        cache=data/'cache/huggingface/hub'/('models--'+model.replace('/','--'))
        link_asset(source,cache/'snapshots'/pins[model])
        ref=cache/'refs/main'
        if ref.exists() and ref.read_text().strip()!=pins[model]:
            raise ValueError(f'Conflicting default model ref: {model}')
        ref.parent.mkdir(parents=True,exist_ok=True);ref.write_text(pins[model])  # HF cache refs are exact commit bytes, without a newline.
    for target,source in m['services'][name].get('asset_links',{}).items():
        link_asset(expand(source),data/target)


def start(data,names):
    cfg,m,_,_=context(data)
    require_external_bindings(m, selected(m, names, cfg.get('external',{})), cfg.get('external',{}))
    for name in selected(m,names,cfg.get('external',{})):
        if name in cfg.get('external',{}):
            with urllib.request.urlopen(cfg['external'][name].rstrip('/')+m['services'][name]['health_path'],timeout=10) as response:
                response.read()
            check_service_contract(name, cfg['external'][name])
            print(name,'explicit external service; not owned or modified',flush=True)
            continue
        s=service_spec(data,name)
        if not Path(s['command'][0]).exists():raise FileNotFoundError(f"Install environment for {name} first")
        install_receipt=Path(s['command'][0]).parents[1]/'install_receipt.json'
        expected_lock=ROOT/m['environments'][m['services'][name]['environment']]['lock']
        if not install_receipt.exists() or read(install_receipt)['lock_sha256']!=hashlib.sha256(expected_lock.read_bytes()).hexdigest():
            raise RuntimeError(f'{name}: environment not qualified against current dependency lock; run install')
        for model in s['models']:
            _,_,_,expand=context(data)
            if not Path(expand('{model:'+model+'}')).is_dir():raise FileNotFoundError(model)
        prepare_assets(data,name)
        for value in [s['env']['HF_HUB_CACHE'],s['env']['CHECKPOINTS_DIR']]:Path(value).mkdir(parents=True,exist_ok=True)
        receipt=data/'services'/f'{name}.json'
        if receipt.exists():
            old=read(receipt)
            if pid_identity(old['pid'])==old['pid_identity']:
                # Starting another group may revisit shared dependencies. Reuse
                # only the same owned process with the same launch definition.
                for field in ('command','env','cwd','url','models'):
                    if old.get(field)!=s[field]:
                        raise RuntimeError(f'{name}: owned process has a different {field}; stop it before reconfiguration')
                with urllib.request.urlopen(s['url']+s['health_path'],timeout=3) as response:
                    response.read()
                check_service_contract(name, s['url'])
                print(name,'same owned service already HTTP ready; retaining inference record',flush=True)
                continue
        with socket.socket() as sock:
            sock.bind((s['bind_address'],int(s['url'].rsplit(':',1)[1])))
        logs=data/'logs';logs.mkdir(parents=True,exist_ok=True)
        with (logs/f'{name}.log').open('ab') as log:
            proc=subprocess.Popen(s['command'],cwd=s['cwd'],env=runtime_env(s),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record={**s,'pid':proc.pid,'pid_identity':pid_identity(proc.pid),'status':'starting'};write(receipt,record)
        try:
            deadline=time.monotonic()+s['timeout']
            while time.monotonic()<deadline:
                if proc.poll() is not None:raise RuntimeError(f'{name} exited ({proc.returncode}); see {logs/name}.log')
                try:
                    with urllib.request.urlopen(s['url']+s['health_path'],timeout=3) as response:
                        payload=response.read().decode();record.update(status='http_ready',health=payload)
                    check_service_contract(name, s['url'])
                    write(receipt,record);print(name,'HTTP ready (model operation still requires qualification)',flush=True);break
                except OSError:time.sleep(1)
            else:raise TimeoutError(f'{name} startup timed out')
        except BaseException:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            record['status']='failed';write(receipt,record);raise


def check_service_contract(name, url):
    if name == 'asr':
        sys.path.insert(0, str(ROOT))
        from x_editor.tools.asr_contract import check_context_contract
        result = check_context_contract(url)
        if not result['ok']:
            raise RuntimeError('ASR context deployment is incomplete: ' + result['error'])


def stop(data,names):
    for path in (data/'services').glob('*.json'):
        if names and path.stem not in names:continue
        record=read(path)
        if pid_identity(record['pid'])==record['pid_identity']:
            os.killpg(record['pid'],signal.SIGTERM)
            deadline=time.monotonic()+30
            while pid_identity(record['pid'])==record['pid_identity'] and time.monotonic()<deadline:
                time.sleep(.2)
            if pid_identity(record['pid'])==record['pid_identity']:
                raise RuntimeError(f'{path.stem} did not stop within 30s; inspect its owned process')
            record['status']='stopped';write(path,record);print('stopped owned service',path.stem)


def export_environment(data):
    cfg,m,_,expand=context(data)
    edit_backend=os.getenv('X_EDITOR_EDIT_BACKEND','dots_tts_edit')
    if edit_backend not in {'dots_tts_edit','edit_data_pipeline'}:
        raise ValueError('X_EDITOR_EDIT_BACKEND must be dots_tts_edit or edit_data_pipeline')
    external=cfg.get('external',{})
    env={k:(expand('{url:'+v+'}')+('/v1' if k.startswith('XEDIT_') else '')
            if not m['services'][v].get('external_only') or v in external else '')
         for k,v in m['client_bindings'].items()}
    for name, policy in EXTERNAL_POLICIES.items():
        env[policy['enable_env']]='1' if name in external else '0'
        env[policy['url_env']]=external.get(name,'')
    env.update({'EDIT_DATA_PIPELINE_ROOT':str(ROOT/'third_party/dots_tts_edit'),
        'X_EDITOR_EDIT_BACKEND':edit_backend,
        'X_EDITOR_MCP_ENV_ROOT':str(Path(cfg['environments'])/'clients'),
        'X_EDITOR_ALIGNER_BACKEND':'qwen3','X_EDITOR_ENABLE_SERVICE_BOOTSTRAP':'0',
        'X_EDITOR_ENABLE_PERCEPTION_SERVICE_BOOTSTRAP':'0'})
    path=data/'client.env';path.write_text(''.join(f'export {k}={shlex.quote(v)}\n' for k,v in env.items()));print(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('configure');q.add_argument('--assets',type=Path);q.add_argument('--env-root');q.add_argument('--port-offset',type=int,default=0);q.add_argument('--gpu',action='append',default=[]);q.add_argument('--external',action='append',default=[],help='Explicitly reuse NAME=http://host:port; never launch/stop that service')
    q.add_argument('--bind-address',default='127.0.0.1',help='IPv4 listen address; use a protected network for remote clients')
    for action in ['plan','install','start','stop','status']:
        sub.add_parser(action).add_argument('names',nargs='*')
    sub.add_parser('export-env')
    args=p.parse_args();data=args.data.resolve();data.mkdir(parents=True,exist_ok=True)
    if args.action=='configure':configure(data,args)
    elif args.action=='install':install(data,args.names or ['cpu'])
    elif args.action=='start':start(data,args.names or ['cpu'])
    elif args.action=='stop':stop(data,args.names)
    elif args.action=='export-env':export_environment(data)
    elif args.action=='plan':
        cfg,m,_,_=context(data)
        external=cfg.get('external',{})
        print(json.dumps([
            {'name': n, 'external': True, 'url': external[n], 'owned': False}
            if n in external else service_spec(data,n,True)
            for n in selected(m,args.names or ['all'],external)
        ],indent=2))
    else:
        for path in sorted((data/'services').glob('*.json')):
            rec=read(path);print(json.dumps({'service':path.stem,'status':rec['status'],'alive':pid_identity(rec['pid'])==rec['pid_identity'],'inference_verified':rec['inference_verified']}))


if __name__=='__main__':main()
