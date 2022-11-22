import _thread
import threading
from contextlib import contextmanager

__all__ = ['time_limiter']


@contextmanager
def time_limiter(seconds: float):  # https://stackoverflow.com/a/37648512
    # Start a timer
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()

    # Run the code in the with statement
    try:
        yield

    # If the timer is triggered, raise an error
    except KeyboardInterrupt:
        raise TimeoutError

    # Otherwise, cancel the timer
    finally:
        timer.cancel()
