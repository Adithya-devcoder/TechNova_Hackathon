"""Run every native ML call (MediaPipe, TFLite) on one long-lived thread.

Streamlit runs each script rerun on a new thread. Creating, using and destroying MediaPipe and TFLite
objects from changing threads crashed the app with an access violation in the second camera session.
Routing all of those calls through a single worker thread avoids that, and also serialises them.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

_PREFIX = 'native-ml'
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=_PREFIX)


def run(fn, *args, **kwargs):
    """Call fn on the worker thread and return its result (directly, if already on that thread)."""
    if threading.current_thread().name.startswith(_PREFIX):
        return fn(*args, **kwargs)
    return _executor.submit(fn, *args, **kwargs).result()


def submit(fn, *args, **kwargs):
    """Start fn on the worker thread without waiting; returns a Future."""
    return _executor.submit(fn, *args, **kwargs)
