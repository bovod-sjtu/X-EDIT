"""Qualify decord 0.6.0's native decoder despite its inconsistent wheel tag.

The PyPI py3-none-manylinux2010_x86_64 wheel contains a cp36 WHEEL tag.
No package code or metadata is modified. This checks the exact affected release
and decodes actual video through its ctypes/native interface before an installer
may accept that one platform-metadata diagnostic.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    dist=importlib.metadata.distribution('decord')
    assert dist.version=='0.6.0',dist.version
    wheel=dist.read_text('WHEEL')
    assert 'Tag: cp36-cp36m-manylinux2010_x86_64' in wheel,wheel
    assert platform.system()=='Linux' and platform.machine()=='x86_64'
    import decord
    with tempfile.TemporaryDirectory(prefix='xedit-decord-') as tmp:
        video=Path(tmp)/'red.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi','-i',
                        'color=c=red:s=64x64:r=8:d=0.5','-c:v','mpeg4','-pix_fmt','yuv420p',str(video)],check=True)
        reader=decord.VideoReader(str(video),ctx=decord.cpu(0))
        frames=reader.get_batch([0,len(reader)-1]).asnumpy()
        assert frames.shape==(2,64,64,3),frames.shape
        assert frames[...,0].mean()>200 and frames[...,1:].mean()<20
        result={'decord':dist.version,'python':platform.python_version(),'wheel_metadata':wheel,
                'frames':len(reader),'decoded_shape':list(frames.shape),'native_decode_verified':True}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
