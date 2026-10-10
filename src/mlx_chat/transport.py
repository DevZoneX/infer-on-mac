"""HTTP streaming and cancellation."""

from __future__ import annotations

import contextlib
import http.client
import queue
import socket
import threading
from urllib.parse import urlsplit


class StreamingResponse:
    """Read HTTP in a worker so cancellation can interrupt headers and idle reads.

    The queue applies backpressure. Closing shuts down the exact connected socket;
    a connect still in progress checks the stop event before sending a request
    and closes its socket when it returns. OS hostname resolution runs in the
    worker so it cannot delay cancellation in the UI.
    """

    def __init__(self, request, *, timeout, connect_timeout, cancel_event=None):
        self.request = request
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.cancel_event = cancel_event
        self._stopped = threading.Event()
        self._packets = queue.Queue(maxsize=8)
        self._socket = None
        self._worker = threading.Thread(target=self._receive, daemon=True)

    def _put(self, kind, value=None):
        while not self._stopped.is_set():
            try:
                self._packets.put((kind, value), timeout=0.05)
                return
            except queue.Full:
                pass

    def _receive(self):
        connection = None
        try:
            url = urlsplit(self.request.full_url)
            connection_type = (
                http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
            )
            connection = connection_type(url.hostname, url.port, timeout=self.connect_timeout)
            connection.connect()
            self._socket = connection.sock
            if self._stopped.is_set():
                return
            connection.sock.settimeout(self.timeout)
            path = url.path or "/"
            if url.query:
                path += "?" + url.query
            connection.request(
                self.request.get_method(),
                path,
                self.request.data,
                dict(self.request.header_items()),
            )
            response = connection.getresponse()
            self._put("headers", (response.status, response.headers))
            with response:
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    self._put("line", response.read(4096))
                else:
                    while not self._stopped.is_set():
                        line = response.readline(1_048_577)
                        if not line:
                            break
                        if len(line) > 1_048_576:
                            raise ValueError("A streaming event exceeded the 1 MiB limit.")
                        self._put("line", line)
            self._put("end")
        except Exception as exc:
            self._put("error", exc)
        finally:
            if connection is not None:
                connection.close()
            self._socket = None

    def _next(self):
        while True:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise KeyboardInterrupt
            try:
                kind, value = self._packets.get(timeout=0.05)
            except queue.Empty:
                continue
            if kind == "error":
                raise value
            return kind, value

    def __enter__(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise KeyboardInterrupt
        self._worker.start()
        try:
            kind, value = self._next()
            if kind != "headers":
                raise http.client.HTTPException("The server did not return response headers.")
            self.status, self.headers = value
            return self
        except BaseException:
            self.close()
            raise

    def __iter__(self):
        while True:
            kind, value = self._next()
            if kind == "end":
                return
            yield value

    def read(self):
        return b"".join(self)

    def close(self):
        self._stopped.set()
        connected_socket = self._socket
        if connected_socket is not None:
            with contextlib.suppress(OSError):
                connected_socket.shutdown(socket.SHUT_RDWR)
        if self._worker.ident is not None:
            self._worker.join(timeout=0.2)

    def __exit__(self, *_args):
        self.close()
