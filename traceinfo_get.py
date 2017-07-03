'''return info on a trace_index

Read each ticker file and update the files association information with CUSIPs.

NOTE: This program looks at all trades for the CUSIP and related OTR CUSIPs from the
beginning of time. Thus trades 10 years ago are used to predict next oasspreads today.
That is way too much history. We should fix this when we have a streaming infrastructure.
Most likely, only the last 1000 or so trades are relevant.

INVOCATION
  python traceinfo_get.py {issuer} {trace_index} {--test} {--trace}
where
 issuer is the issuer (ex: AAPL)
 trace_index is the symbol for the company (ex: AAPL)
 effective_date: YYYY-MM-DD is the date of the trade
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

 EXAMPLE INVOCATIONS:
  python traceinfo_get.py AAPL 127076037
'''

from __future__ import division

import argparse
import pdb
from pprint import pprint
import random
import sys

import applied_data_science.debug
import applied_data_science.dirutility
import applied_data_science.lower_priority
import applied_data_science.pickle_utilities

from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.arg_type
import seven.build
import seven.read_csv

import traceinfo

pp = pprint
Logger
Timer


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('issuer', type=seven.arg_type.issuer)
    parser.add_argument('trace_index', type=seven.arg_type.trace_id)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')

    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    path = seven.build.traceinfo_get(arg.issuer, arg.trace_index, test=arg.test)
    applied_data_science.dirutility.assure_exists(path['dir_out'])

    return Bunch(
        arg=arg,
        path=path,
        random_seed=random_seed,
        # timer=Timer(),
    )


def do_work(control):
    'accumulate information on the trace prints for the issuer and write that info to the file system'
    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    by_trace_index = traceinfo.read_by_trace_index(control.arg.issuer)

    trace_index = int(control.arg.trace_index)
    if trace_index in by_trace_index:
        format = '%20s %s'
        info = by_trace_index[trace_index]
        print format % ('issuepriceid', info['issuepriceid'])
        print format % ('issuer', info['issuer'])
        print format % ('cusip', info['cusip'])
        print format % ('effective_date', info['effective_date'])
        print format % ('effective_datetime', info['effective_datetime'])
    else:
        print '%s not found in %s' % (trace_index, control.arg.issuer)

    return None


def main(argv):
    control = make_control(argv)
    # sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
    # print control
    # lap = control.timer.lap

    do_work(control)

    # lap('work completed')
    # if control.arg.test:
    #     print 'DISCARD OUTPUT: test'
    # # print control
    # print control.arg
    # print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb

    main(sys.argv)
