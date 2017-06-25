'''create information ("side info") needed to control the build process

Read each ticker file and update the files association information with CUSIPs.

NOTE: This program looks at all trades for the CUSIP and related OTR CUSIPs from the
beginning of time. Thus trades 10 years ago are used to predict next oasspreads today.
That is way too much history. We should fix this when we have a streaming infrastructure.
Most likely, only the last 1000 or so trades are relevant.

INVOCATION
  python features_targets.py {cusip} {effective_date} {--test} {--trace}
where
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 effective_date: YYYY-MM-DD is the date of the trade
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATIONS
  python buildinfo.py ORCL

See build.py for input and output files.

IDEA FOR FUTURE:
'''

from __future__ import division

import argparse
import collections
import copy
import cPickle as pickle
import datetime
import os
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
import seven.feature_makers
import seven.fit_predict_output
import seven.read_csv
import seven.target_maker

pp = pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('issuer', type=seven.arg_type.issuer)
    # parser.add_argument('cusip', type=seven.arg_type.cusip)
    # parser.add_argument('effective_date', type=seven.arg_type.date)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')

    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    path = seven.build.buildinfo(arg.issuer, test=arg.test)
    applied_data_science.dirutility.assure_exists(path['dir_out'])

    return Bunch(
        arg=arg,
        path=path,
        random_seed=random_seed,
        timer=Timer(),
    )


def sort_by_effectivedatetime(df):
    'return new DataFrame in sorted order'
    # Note: the following statement works but generates a SettingWithCopyWarning
    df['effectivedatetime'] = seven.feature_makers.make_effectivedatetime(df)
    result = df.sort_values('effectivedatetime')
    return result


def read_and_transform_trace_prints(issuer, test):
    'return sorted DataFrame containing trace prints for the cusip and all related OTR cusips'
    trace_prints = seven.read_csv.input(
        issuer,
        'trace',
        nrows=1000 if test else None,
    )
    # make sure the trace prints are in non-decreasing datetime order
    sorted_trace_prints = sort_by_effectivedatetime(trace_prints)
    sorted_trace_prints['issuepriceid'] = sorted_trace_prints.index  # make the index one of the values
    return sorted_trace_prints


# def read_pickle(path):
#     'read pickle file, if it exists; return (obj_in_it, err)'
#     if os.path.isfile(path):
#         print 'reading pickle file', path
#         with open(path, 'rb') as f:
#             obj = pickle.load(f)
#         return obj, False
#     else:
#         return None, 'file does not exist'


# def write_pickle(obj, path):
#     with open(path, 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#     print 'wrote pickled object of type %s to path %s' % (type(obj), path)


# def get_trade_date(trace_record):
#     'return date of the trade: datetime.date'
#     return trace_record['effectivedate'].date()


class Accumulator(object):
    def __init__(self, path, empty_obj):
        self.path = path
        # set self.accumulator
        if os.path.isfile(path):
            print 'reading pickle file', path
            with open(path, 'rb') as f:
                self.accumulator = pickle.load(f)
        else:
            self.accumulator = copy.deepcopy(empty_obj)

    def get_cusip(self, trace_record):
        return trace_record['cusip']

    def get_trade_date(self, trace_record):
        'return date of trade:datetime.date'
        return trace_record['effectivedate'].date()

    def write(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.accumulator, f, pickle.HIGHEST_PROTOCOL)
        print 'wrote pickeld object of type %s to path %s' % (type(self.accumulator), self.path)

    def accumulate(self, trace_index, trace_record):
        raise TypeError('a subclass must override me')

    def print_table(self, trace_index, trace_record):
        raise TypeError('a subclass must override me')


class AccumulateIssuers(Accumulator):
    def __init__(self, path, issuer):
        super(AccumulateIssuers, self).__init__(path, collections.defaultdict(set))
        self.issuer = issuer

    def accumulate(self, trace_index, trace_record):
        cusip = self.get_cusip(trace_record)
        self.accumulator[cusip].add(self.issuer)

    def print_table(self):
        print
        print 'cusip -> issuer'
        for cusip in sorted(self.accumulator.keys()):
            cusip_issuers = self.accumulator[cusip]
            if len(cusip_issuers) > 1:
                print 'cusip %s has multiple issuers: %s' % (cusip, cusip_issuers)
                pdb.set_trace()
            else:
                for issuer in cusip_issuers:
                    print cusip, issuer


class AccumulateNTrades(Accumulator):
    def __init__(self, path):
        super(AccumulateNTrades, self).__init__(path, collections.defaultdict(int))

    def accumulate(self, trace_index, trace_record):
        cusip = self.get_cusip(trace_record)
        self.accumulator[cusip] += 1

    def print_table(self):
        print
        print 'cusip -> n_trades'
        for cusip in sorted(self.accumulator.keys()):
            n_trades = self.accumulator[cusip]
            print cusip, n_trades


class AccumulateTraceIndices(Accumulator):
    def __init__(self, path):
        super(AccumulateTraceIndices, self).__init__(path, collections.defaultdict(int))

    def accumulate(self, trace_index, trace_record):
        self.accumulator[trace_index] += 1

    def print_table(self):
        print
        print 'trace_indices occuring more than once'
        for trace_index in sorted(self.accumulator.keys()):
            n = self.accumulator[trace_index]
            if n > 1:
                print trace_index, n


class AccumulateTraceindexTradedate(Accumulator):
    def __init__(self, path):
        super(AccumulateTraceindexTradedate, self).__init__(path, dict())

    def accumulate(self, trace_index, trace_record):
        trade_date = self.get_trade_date(trace_record)
        self.accumulator[trace_index] = trade_date

    def print_table(self):
        print
        print 'trace_index -> trade_date'
        print 'NOT PRINTED (LONG)'


def default_n_trades_value():
    # NOTE: must be defined at top level of this module
    # because it is pickled
    return collections.defaultdict(int)


class AccumulateNTradesByDate(Accumulator):
    def __init__(self, path):
        super(AccumulateNTradesByDate, self).__init__(path, collections.defaultdict(default_n_trades_value))

    def accumulate(self, trace_index, trace_record):
        cusip = self.get_cusip(trace_record)
        trade_date = self.get_trade_date(trace_record)
        self.accumulator[cusip][trade_date] += 1

    def print_table(self):
        print
        print 'cusip -> trade date -> n_trades'
        for cusip in sorted(self.accumulator.keys()):
            n_trades_cusip = self.accumulator[cusip]
            for date in sorted(n_trades_cusip.keys()):
                n = n_trades_cusip[date]
                print cusip, date, n


def do_work(control):
    'accumulate information on the trace prints for the issuer and write that info to the file system'
    def lap():
        'return ellapsed wall clock time:float since previous call to lap()'
        return control.timer.lap('lap', verbose=False)[1]

    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    # input files are for a specific ticker
    trace_prints = read_and_transform_trace_prints(
        control.arg.issuer,
        control.arg.test,
    )
    print 'read and transformed %d trace prints in %.3f wall clock seconds' % (len(trace_prints), lap())

    counter = collections.Counter()

    # accumulate info for cusips
    accumulators = (
        AccumulateIssuers(control.path['out_issuers'], control.arg.issuer),
        AccumulateNTrades(control.path['out_n_trades']),
        AccumulateNTradesByDate(control.path['out_n_trades_by_date']),
        AccumulateTraceIndices(control.path['out_trace_indices']),
        AccumulateTraceindexTradedate(control.path['out_traceindex_tradedate']),
    )
    for trace_index, trace_record in trace_prints.iterrows():
        counter['n trace records read'] += 1
        for accumulator in accumulators:
            accumulator.accumulate(trace_index, trace_record)

    # report counts
    print 'end loop on input in %.3f wall clock seconds' % lap()
    print 'counts'
    for k in sorted(counter.keys()):
        print '%71s: %6d' % (k, counter[k])

    for accumulator in accumulators:
        accumulator.print_table()
        accumulator.write()

    return None


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    # print control
    print control.arg
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
        datetime

    main(sys.argv)
