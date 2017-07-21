'''extract the structure of the trade from the trace_{issuer}.csv files

Read each ticker file and update the files association information with CUSIPs.

NOTE: This program looks at all trades for the CUSIP and related OTR CUSIPs from the
beginning of time. Thus trades 10 years ago are used to predict next oasspreads today.
That is way too much history. We should fix this when we have a streaming infrastructure.
Most likely, only the last 1000 or so trades are relevant.

INVOCATION
  python traceinfo.py {issuer} {--sqlite} {--test} {--trace}
where
 issuer is the symbol for the company (ex: AAPL)
-- sqlite means to create an SQlite database as the output. MAYBE: implement in future.
 --test means to set control.arg.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATIONS
  python buildinfo.py AAPL

See build.py for input and output files.

IDEAS FOR FUTURE:
1. Also scan all the trace files and create mappings across them. For example, it would be good
   to verify that every cusip has exactly one issuer and every issuepriceid occurs once.
2. automatic_feeds/secmaster.csv maps cusip -> issuer

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import datetime
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

# from seven.traceinfo_types import TraceInfo

import seven.arg_type
import seven.build
import seven.read_csv

# import TraceInfo

pp = pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('issuer', type=seven.arg_type.issuer)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')

    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    path = seven.build.traceinfo(arg.issuer, test=arg.test)
    applied_data_science.dirutility.assure_exists(path['dir_out'])

    return Bunch(
        arg=arg,
        first_date='2017-06-01',
        path=path,
        random_seed=random_seed,
        timer=Timer(),
    )


def read_and_transform_trace_prints(control):
    'return sorted DataFrame containing trace prints for the cusip and all related OTR cusips'
    trace_prints = seven.read_csv.input(
        control.arg.issuer,
        'trace',
        nrows=10000 if control.arg.test else None,
    )
    control.timer.lap('read_and_transform_trace_prints')
    return trace_prints


def make_effectivedatetime(trace_record):
    'return datetime.datetime'
    effective_date = trace_record['effectivedate'].date()
    effective_time = trace_record['effectivetime'].time()
    effective_datetime = datetime.datetime(
        effective_date.year,
        effective_date.month,
        effective_date.day,
        effective_time.hour,
        effective_time.minute,
        effective_time.second,
    )
    return effective_datetime


def make_datetime_date(s):
    year, month, day = s.split('-')
    return datetime.date(int(year), int(month), int(day))


def do_work(control):
    'accumulate information on the trace prints for the issuer and write that info to the file system'
    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    # input files are for a specific ticker
    trace_prints = read_and_transform_trace_prints(control)
    first_date = make_datetime_date(control.first_date)

    counter = collections.Counter()

    summary = []
    by_trace_index = {}
    by_issuer_cusip = {}
    by_trade_date = {}
    for trace_index, trace_record in trace_prints.iterrows():
        counter['n trace records read'] += 1
        effective_date = trace_record['effectivedate'].date()
        if effective_date < first_date:
            continue
        counter['n_info records created'] += 1
        trace_info = {
            'issuer': control.arg.issuer,
            'cusip': trace_record['cusip'],
            'issuepriceid': trace_index,
            'effective_date': effective_date,
            'effective_datetime': make_effectivedatetime(trace_record),
        }

        # summary
        summary.append(trace_info)

        # by_issuer_cusip
        key = (control.arg.issuer, trace_record['cusip'])
        if key not in by_issuer_cusip:
            by_issuer_cusip[key] = []
        by_issuer_cusip[key].append(trace_info)

        # by_trace_index
        assert trace_index not in by_trace_index
        by_trace_index[trace_index] = trace_info

        # by_trade_date
        key = trace_record['effectivedate'].date()
        if key not in by_trade_date:
            by_trade_date[key] = []
        by_trade_date[key].append(trace_info)

    control.timer.lap('accumulated all info')

    print 'counters'
    for k in sorted(counter.keys()):
        print k, counter[k]

    with open(control.path['out_by_issuer_cusip'], 'wb') as f:
        pickle.dump(by_issuer_cusip, f, pickle.HIGHEST_PROTOCOL)

    with open(control.path['out_by_trace_index'], 'wb') as f:
        pickle.dump(by_trace_index, f, pickle.HIGHEST_PROTOCOL)

    with open(control.path['out_by_trade_date'], 'wb') as f:
        pickle.dump(by_trade_date, f, pickle.HIGHEST_PROTOCOL)

    with open(control.path['out_summary'], 'wb') as f:
        pickle.dump(summary, f, pickle.HIGHEST_PROTOCOL)

    control.timer.lap('wrote output files')

    return None


def write_report_cusip_effectivedatetime_issuepriceids(control):
    'write to stdout'
    # detail line selection criteria
    earliest_date = make_datetime_date('2017-06-23')
    cusips = ('037833AG5',)

    print 'selected cusip -> effectve_datetime -> issuepriceid'

    summary = read_summary(control.arg.issuer)
    for cusip in cusips:
        def select_cusip_date(info):
            return info['cusip'] == cusip and info['effective_date'] >= earliest_date

        selected = filter(select_cusip_date, summary)
        effective_datetimes = sorted(set(map(lambda info: info['effective_datetime'], selected)))
        for effective_datetime in effective_datetimes:
            row_data = filter(lambda info: info['effective_datetime'] == effective_datetime, selected)
            print cusip, effective_datetime,
            for row_datum in row_data:
                print row_datum['issuepriceid'],
            print


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)
    write_report_cusip_effectivedatetime_issuepriceids(control)

    lap('work completed')
    if control.arg.test:
        print 'DISCARD OUTPUT: test'

    # print invocation args
    print control.arg
    print 'done'
    return

# external API
# these entry points all called by other programs


def read_by_trace_index(issuer):
    paths = seven.build.traceinfo(issuer)
    path = paths['out_by_trace_index']
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def read_summary(issuer):
    paths = seven.build.traceinfo(issuer)
    path = paths['out_summary']
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
        datetime

    main(sys.argv)
