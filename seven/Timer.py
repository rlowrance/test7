'''class Timer

Copyright 2017 Roy E. Lowrance

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on as "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing premission and
limitation under the license.
'''
import atexit
import os
import pdb
import time
from functools import reduce


class Timer(object):
    def __init__(self):
        # time.clock() returns:
        #  unix ==> processor time in seconds as float (cpu time)
        #  windows ==> wall-clock seconds since first call to this function
        #  NOTE: time.clock() is deprecated in python 3.3
        self._program_start_clock = time.clock()  # processor time in seconds
        # time.time() returns:
        #  unit & windows ==> time in seconds since epoch as float
        self._program_start_time = time.time()  # time in seconds since the epoch (on Unix)
        self._program = (self._program_start_clock, self._program_start_time)
        self._lap = (self._program_start_clock, self._program_start_time)
        atexit.register(self.endlaps)

    # initial API
    def elapsed_cpu_seconds(self):
        return time.clock() - self._program_start_clock

    def elapsed_wallclock_seconds(self):
        return time.time() - self._program_start_time

    # second API (keep first for backwards compatibility)
    def clock_time(self):
        return (time.clock(), time.time())

    def lap(self, s, verbose=True):
        'return (cpu seconds, wall clock seconds) in last lap; maybe print time of current lap'
        # NOTE: Cannot use the python standard library to find the elapsed CPU time on Windows
        # instead, Windows returns the wall clock time
        # inspired by Paul McGuire's timing.py
        # ref: http://stackoverflow.com/questions/1557571/how-to-get-time-of-a-python-program-execution

        def toStr(t):
            'convert seconds to hh:mm:ss.sss'
            # this code from Paul McGuire!
            return '%d:%02d:%02d.%03d' % reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                                                [(t * 1000,), 1000, 60, 60])

        def diff(start, now):
            return (
                toStr(now[0] - start[0]),
                toStr(now[1] - start[1])
            )

        def diff_float(start, now):
            return (now[0] - start[0], now[1] - start[1])

        clock_time = self.clock_time()
        cumulative_seconds = diff(self._program, clock_time)
        lap_seconds_str = diff(self._lap, clock_time)
        lap_seconds_float = diff_float(self._lap, clock_time)
        self._lap = clock_time  # reset lap time
        if verbose:
            visual_clue = '=' * 50
            print(visual_clue)
            print('lap: %s' % s)
            print('cumulative %s cpu %s wallclock' % cumulative_seconds)
            print('lap        %s cpu %s wallclock' % lap_seconds_str)
            print(visual_clue)
            print()
        return lap_seconds_float  # unix->(processor sec, wall clock sec) windows->(wall clock sec, wall clock sec)

    def endlaps(self):
        self.lap('**End Program**')
