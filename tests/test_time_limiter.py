import time
import pytest
from assign_enc.time_limiter import time_limiter


def test_time_limiter():
    with time_limiter(.2):
        time.sleep(.1)

    with pytest.raises(TimeoutError):
        with time_limiter(.1):
            time.sleep(.2)
