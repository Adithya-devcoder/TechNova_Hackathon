"""Run the Kaggle 1st-place TFLite model in its own process.

Inside the Streamlit app, the TFLite engine occasionally crashed the whole process (access violation)
when running next to MediaPipe, which bundles its own copy of TFLite. Running the model in a separate
Python process isolates it: if that process ever crashes or hangs, it is restarted and the app keeps going.

The helper is started as `python tflite_server.py <model> <key>`, independent of how the app itself was
launched. It loads the model, prints a local port, and answers one clip at a time over that connection.
This file imports nothing heavy (no torch, no MediaPipe), so the helper starts quickly.
"""
import os
import secrets
import subprocess
import sys
import threading
from multiprocessing.connection import Client, Listener

import numpy as np


def _serve(model_path, authkey):
    """Helper process: load the model, publish a port, then answer clips until the app disconnects."""
    try:   # LiteRT is the small, supported TFLite runtime; full TensorFlow also works
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter
    interpreter = Interpreter(model_path=model_path)
    signatures = interpreter.get_signature_list()
    key = 'serving_default' if 'serving_default' in signatures else next(iter(signatures))
    runner = interpreter.get_signature_runner(key)
    input_name, output_name = signatures[key]['inputs'][0], signatures[key]['outputs'][0]
    with Listener(('127.0.0.1', 0), authkey=authkey) as listener:
        print(f'PORT {listener.address[1]}', flush=True)
        with listener.accept() as conn:
            while True:
                try:
                    clip = conn.recv()
                except EOFError:   # the app closed the connection or exited
                    break
                conn.send(np.asarray(runner(**{input_name: clip})[output_name]))


class RemoteModel:
    """Callable clip -> model output, computed in a helper process that is restarted if it dies."""

    CALL_TIMEOUT_S = 10

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self._lock = threading.Lock()
        self._proc = self._conn = None
        self.restarts = -1
        self._start()

    def _start(self):
        authkey = secrets.token_bytes(16)
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        self._proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), self.model_path, authkey.hex()],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                      creationflags=flags)
        port = None
        for line in self._proc.stdout:   # ends if the helper exits before it is ready
            if line.startswith('PORT '):
                port = int(line.split()[1])
                break
        if port is None:
            self._proc.kill()
            raise RuntimeError('the TFLite model process did not start')
        self._conn = Client(('127.0.0.1', port), authkey=authkey)
        self.restarts += 1

    def _ask(self, clip):
        self._conn.send(clip)
        if not self._conn.poll(self.CALL_TIMEOUT_S):   # stuck: treat it like a crash
            raise TimeoutError('the TFLite model process did not answer')
        return self._conn.recv()

    def __call__(self, clip):
        clip = np.ascontiguousarray(clip, np.float32)
        with self._lock:
            try:
                return self._ask(clip)
            except (EOFError, OSError, TimeoutError):   # the helper crashed or hung: restart, try once more
                self.close()
                self._start()
                return self._ask(clip)

    def close(self):
        try:
            self._conn.close()
        except (AttributeError, OSError):
            pass
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait(timeout=5)


if __name__ == '__main__':
    _serve(sys.argv[1], bytes.fromhex(sys.argv[2]))
