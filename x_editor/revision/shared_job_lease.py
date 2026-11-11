"""Production review support extracted without changing function bodies."""

from contextlib import contextmanager


import json


import os


from pathlib import Path


import socket


@contextmanager
def directory_lease(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as error:
        raise BlockingIOError(f'shared work is already owned: {path}') from error
    owner = path/'owner.json'
    try:
        owner.write_text(json.dumps({'host': socket.gethostname(), 'pid': os.getpid()}))
        yield
    finally:
        owner.unlink(missing_ok=True)
        path.rmdir()
