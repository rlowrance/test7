'''determine how to build a program: sources, targets, commands

Each function takes these args:
positional1, position2, ... : positional arguments on the command line
executable=String           : name of the executable, without the py suffix; ex: fit_predict
test=                       : whether the output goes to the test directory

Each function returns a dictionary with these keys:
'dir_out': path to the output directory, which depends on the positional args and whether test=True is present
'in*':  path to an input file
'out*':  path to an output file
'executable':  name of executable: example: fit_predict
'dep*:  another dependency to the build process; the 'in*" items are also dependencies
'command': invocation command

There is one such function for each program.
'''

from __future__ import division

import os
import pdb
import pprint
import sys

import seven.path
pp = pprint.pprint


def make_scons(paths):
    'return Dict with items sources, targets, command, based on the paths'
    def select(f):
        return [v for k, v in paths.items() if f(k)]

    result = {
        'commands': [paths['command']],
        'sources': select(lambda k: k.startswith('in') or k.startswith('executable') or k.startswith('dep')),
        'targets': select(lambda k: k.startswith('out')),
    }
    return result


def cusips(ticker, executable='cusips', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = seven.path.working()
    dir_out = os.path.join(dir_working, '%s-%s%s' % (
        executable,
        ticker,
        ('-test' if test else ''),
        )
    )

    result = {
        'in_trace': seven.path.input(ticker=ticker, logical_name='trace'),

        'out_cusips': os.path.join(dir_out, '%s.pickle' % ticker),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s' % (executable, ticker),
    }
    return result


def features(ticker, executable='features', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = seven.path.working()
    dir_out = os.path.join(dir_working, '%s-%s%s' % (
        executable,
        ticker,
        ('-test' if test else ''),
        )
    )
    assert ticker == 'orcl'
    representative_cusip = '68389XAC9.csv'

    result = {
        'in_etf_agg': seven.path.input(ticker, 'etf agg'),
        'in_etf_lqa': seven.path.input(ticker, 'etf lqd'),
        'in_fund': seven.path.input(ticker, 'fund'),
        'in_security_master': seven.path.input(ticker, 'security master'),
        'in_ohlc_equity_spx': seven.path.input(ticker, 'ohlc spx'),
        'in_ohlc_equity_ticker': seven.path.input(ticker, 'ohlc ticker'),
        'in_trace': seven.path.input(ticker, 'trace'),

        'out_cusips': os.path.join(dir_out, '%s.pickle' % representative_cusip),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s' % (executable, ticker),
    }
    return result


def targets(ticker, executable='targets', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = seven.path.working()
    dir_out = os.path.join(dir_working, '%s-%s%s' % (
        executable,
        ticker,
        ('-test' if test else ''),
        )
    )
    assert ticker == 'orcl'
    representative_cusip = '68389XAC9.csv'

    result = {
        'in_etf_agg': seven.path.input(ticker, 'etf agg'),
        'in_etf_lqa': seven.path.input(ticker, 'etf lqd'),
        'in_fund': seven.path.input(ticker, 'fund'),
        'in_security_master': seven.path.input(ticker, 'security master'),
        'in_ohlc_equity_spx': seven.path.input(ticker, 'ohlc spx'),
        'in_ohlc_equity_ticker': seven.path.input(ticker, 'ohlc ticker'),
        'in_trace': seven.path.input(ticker, 'trace'),

        'out_cusips': os.path.join(dir_out, '%s.pickle' % representative_cusip),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s' % (executable, ticker),
    }
    return result
if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pass
    main(sys.argv)
