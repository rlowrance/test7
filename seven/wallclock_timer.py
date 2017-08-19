'determine wall clock elapsed time'
import sys
import time


def seconds_since_epoch():
    'return time past the epoch in fractional seconds'
    # ref: https://stackoverflow.com/questions/1938048/high-precision-clock-in-python
    if sys.platform == 'win32':
        # return wall-clock seconds elapsed since first call to time.clock()
        return time.clock()
    else:
        # return time in seconds since the epoch as a floating-point number
        return time.time()

