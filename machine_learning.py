'''machine learning utility functions and classes'''
import datetime
import os
import sys
import typing
import pdb
import pprint
import unittest

import shared_configuration


class Logger(object):
    # ref: stack overflow: how do i duplicat sys stdout to a log file in python
    def __init__(self, logfile_path=None):
        self.terminal = sys.stdout
        clean_path = logfile_path.replace(':', '-') if os.name == 'posix' else logfile_path
        self.log = open(clean_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        pass

    
def main(
        argv: typing.List[str],
        program: str,
        unittest,
        do_work,
        out_log='out_log',
        ):
    config = shared_configuration.make(
        program=program,
        argv=argv[1:],
        )
    print('started %s with configuration' % program)
    print(str(config))
    # echo print statement to the log file
    sys.stdout = Logger(config.get(out_log))
    print(str(config))
    if config.get('debug', False):
        # enter pdb if run-time error
        # (useful during development)
        import debug
        if False:
            debug.info
    unittest(config)
    do_work(config)


def make_call_if(condition: bool, f):
    if condition:
        def ff(*args, **kwargs):
            f(*args, **kwargs)
    else:
        def ff(*args, **kwargs):
            pass
    return ff


def make_verbose_pp(verbose: bool):
    return make_call_if(
        condition=verbose,
        f=pprint.pprint,
        )


def make_verbose_print(verbose: bool):
    if verbose:
        def verbose_print(*args):
            print(*args)
    else:
        def verbose_print(*args):
            pass
    return verbose_print


def make_set_trace(flag: bool):
    if flag:
        def set_trace():
            # NOTE: see ~/anaconda3/lib/python3.6/pdb.py function set_trace() to understand this call
            pdb.Pdb().set_trace(sys._getframe().f_back)

    else:
        def set_trace():
            pass

    return set_trace


class Test(unittest.TestCase):
    def test_make_verbose_print(self):
        vp1 = make_verbose_print(False)
        vp1('a', 123)
        vp2 = make_verbose_print(False)
        vp2('a', 123)  # should print

    def test_make_set_trace(self):
        set_trace1 = make_set_trace(False)
        set_trace2 = make_set_trace(False)
        if False:
            set_trace1()
            set_trace2()


if False:
    # usage example
    sys.stdout = Logger('path/to/log/file')
    pdb
    # now print statements write on both stdout and the log file

if __name__ == '__main__':
    unittest.main()
    
