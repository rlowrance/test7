'''determine CUSIPS in a ticker file

INVOCATION: python cusips.py {ticker}.csv [--test] [--trace]

INPUT FILES:
 MidPredictor/data/{ticker}.csv

OUTPUT FILES:
 WORKING/cusips-{ticker}[-test]/0log-{ticker}.txt  whatever is printed when this program last ran
 WORKING/cusips{ticker}[-test]/{ticker}.pickle  # Dict[cusip, count]
'''

from __future__ import division

import os
import pdb
import sys

import seven.path


def make_paths(ticker, executable='cusips', test=False):
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
        'in_executable': '%s.py' % executable,
        'in_imported': '%s_paths.py' % executable,
        'out_cusips': os.path.join(dir_out, '%s.pickle' % ticker),
        'out_log': os.path.join(dir_out, '0log.txt'),
        'executable': executable,
        'dir_out': dir_out,
    }
    return result


def make_scons(ticker, test=False):
    'return dict with items sources, targets, command'
    paths = make_paths(ticker, test=test)
    return {
        'sources': [v for k, v in paths.items() if k.startswith('in_')],
        'targets': [v for (k, v) in paths.items() if k.startswith('out_')],
        'commands': ['python %s %s' % (paths['in_executable'], ticker)],
        }


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pass
    main(sys.argv)
