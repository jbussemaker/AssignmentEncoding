import ctypes
import threading

try:
    import gevent
except ImportError:
    gevent = None

__all__ = ['run_timeout']


class JoinThread(threading.Thread):

    def __init__(self, other_thread: threading.Thread):
        super().__init__(daemon=True)
        self._other = other_thread

    def run(self) -> None:
        while self._other.is_alive():
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(self._other.ident), ctypes.py_object(KeyboardInterrupt()))
            self._other.join()


def run_timeout(seconds: float, func, *args, **kwargs):

    return_val = []
    exception = []

    def _wrapper():
        try:
            return_val.append(func(*args, **kwargs))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            exception.append(e)

    thread = threading.Thread(target=_wrapper, daemon=True)
    thread.start()
    thread.join(seconds)

    if thread.is_alive():
        stopper_thread = JoinThread(thread)
        stopper_thread.start()
        stopper_thread.join()
        raise TimeoutError

    if exception:
        raise exception[0]
    if not return_val:
        raise KeyboardInterrupt
    return return_val[0]
