# To check if it also works with gevent
# NOTE: do not patch thread!!
# from gevent import monkey; monkey.patch_all(thread=False)

import timeit
import itertools
import threading
from adsg_core.optimization.assign_enc.time_limiter import *


def test_time_limiter():

    exc = []

    def _do_run():
        try:
            s = timeit.default_timer()
            n = 0
            for _ in itertools.combinations_with_replacement(list(range(20)), 10):
                n += 1
                if n % 1000 == 0 and timeit.default_timer()-s > .01:
                    break

            def _func(n_run, a=20, b=10, ret_val=None, raise_exc=None):
                if raise_exc is not None:
                    raise raise_exc

                for i, _ in enumerate(itertools.combinations_with_replacement(list(range(a)), b)):
                    if i > n_run:
                        return ret_val

            obj = object()
            assert run_timeout(.2, _func, n*10, ret_val=obj) is obj

            assert run_timeout(.2, lambda: _func(n*10, ret_val=obj)) is obj

            try:
                run_timeout(.2, _func, n*10, raise_exc=RuntimeError())
                raise AssertionError
            except RuntimeError:
                pass

            try:
                run_timeout(.1, _func, n*50)
                raise AssertionError
            except TimeoutError:
                pass

        except Exception as e:
            exc.append(e)
            raise

    _do_run()
    assert len(exc) == 0

    try:
        thread = threading.Thread(target=_do_run)
        thread.start()
        thread.join()
    except KeyboardInterrupt:
        raise RuntimeError
    assert len(exc) == 0
