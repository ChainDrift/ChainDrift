import subprocess
import time


MAXIMUM_STARTUP_TIME = 0.5


def test_startup_time():
    # warm up to generate pyc
    subprocess.run(["chaindrift", "-h"])

    start = time.time()
    subprocess.run(["chaindrift", "-h"])
    elapsed = time.time() - start
    assert (
        elapsed < MAXIMUM_STARTUP_TIME
    ), "The startup time is too long, try to use lazy import in the command entry function"
