"""Production review support extracted without changing function bodies."""

from __future__ import annotations


import contextvars


import threading


import time


from contextlib import contextmanager


from dataclasses import dataclass, field


from pathlib import Path


from urllib.parse import urlsplit


import httpx


import requests


from x_editor.revision.artifact_io import write_json


@dataclass
class AssignmentTransport:
    directory: Path
    maximum_chat_calls: int
    chat_calls: int = 0
    http_calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def begin(self, method, url, kwargs):
        parsed = urlsplit(str(url))
        if parsed.username or parsed.password:
            raise ValueError("credentials must not be embedded in service URLs")
        address = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        chat = parsed.path.endswith("/chat/completions")
        with self.lock:
            if chat and self.chat_calls >= self.maximum_chat_calls:
                raise RuntimeError("predeclared assignment chat HTTP budget exhausted")
            self.chat_calls += int(chat)
            self.http_calls += 1
            index = self.http_calls
        destination = self.directory / f"{index:04d}"
        destination.mkdir(parents=True, exist_ok=False)
        write_json(
            destination / "request.json",
            {
                "method": method,
                "url": address,
                "chat_payload": kwargs.get("json") if chat else None,
            },
        )
        return destination, chat, time.monotonic()

    @staticmethod
    def finish(start, response=None, error=None):
        destination, chat, began = start
        if response is not None and chat:
            with (destination / "response.txt").open("x") as handle:
                handle.write(response.text)
        write_json(
            destination / "receipt.json",
            {
                "elapsed_sec": time.monotonic() - began,
                "status_code": response.status_code if response is not None else None,
                "error_type": type(error).__name__ if error else None,
            },
        )


current_assignment = contextvars.ContextVar("module_chain_assignment", default=None)


@contextmanager
def recorded_http():
    original_sync = requests.sessions.Session.request
    original_async = httpx.AsyncClient.request

    def sync(session, method, url, **kwargs):
        owner = current_assignment.get()
        if owner is None:
            return original_sync(session, method, url, **kwargs)
        start = owner.begin(method, url, kwargs)
        response = error = None
        try:
            response = original_sync(session, method, url, **kwargs)
            return response
        except Exception as exc:
            error = exc
            raise
        finally:
            owner.finish(start, response, error)

    async def asynchronous(client, method, url, **kwargs):
        owner = current_assignment.get()
        if owner is None:
            return await original_async(client, method, url, **kwargs)
        start = owner.begin(method, url, kwargs)
        response = error = None
        try:
            response = await original_async(client, method, url, **kwargs)
            return response
        except Exception as exc:
            error = exc
            raise
        finally:
            owner.finish(start, response, error)

    requests.sessions.Session.request = sync
    httpx.AsyncClient.request = asynchronous
    try:
        yield
    finally:
        requests.sessions.Session.request = original_sync
        httpx.AsyncClient.request = original_async
