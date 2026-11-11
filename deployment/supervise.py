"""Keep an owned service deployment alive as a foreground task."""
import argparse
from pathlib import Path
import signal
import time
import stack


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    a=p.parse_args();data=a.data.resolve()
    def stopping(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stopping)
    try:
        while True:
            records=[stack.read(p) for p in (data/'services').glob('*.json')]
            active=[r for r in records if r['status']=='http_ready']
            if not active:raise RuntimeError('No owned ready services to supervise')
            for r in active:
                if stack.pid_identity(r['pid'])!=r['pid_identity']:
                    raise RuntimeError(f"Owned service exited: {r['name']}; inspect deployment logs")
            time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        stack.stop(data,[])


if __name__=='__main__':main()
