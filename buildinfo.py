'''create information ("side info") needed to control the build process

Read each ticker file and update the files association information with CUSIPs.

NOTE: This program looks at all trades for the CUSIP and related OTR CUSIPs from the
beginning of time. Thus trades 10 years ago are used to predict next oasspreads today.
That is way too much history. We should fix this when we have a streaming infrastructure.
Most likely, only the last 1000 or so trades are relevant.

INVOCATION
  python buildinfo.py {issuer} {--test} {--trace}
where
 issuer is the symbol for the company (ex: AAPL)
 effective_date: YYYY-MM-DD is the date of the trade
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATIONS
  python buildinfo.py AAPL

See build.py for input and output files.

IDEAS FOR FUTURE:
1. Also scan all the trace files and create mappings across them. For example, it would be good
   to verify that every cusip has exactly one issuer and every issuepriceid occurs once.
'''

from __future__ import division

import argparse
import collections
import copy
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


def read_and_transform_trace_prints(control):
    'return sorted DataFrame containing trace prints for the cusip and all related OTR cusips'
    trace_prints = seven.read_csv.input(
        control.arg.issuer,
        'trace',
        nrows=1000 if control.arg.test else None,
    )
    # make sure the trace prints are in non-decreasing datetime order
    sorted_trace_prints = sort_by_effectivedatetime(trace_prints)
    sorted_trace_prints['issuepriceid'] = sorted_trace_prints.index  # make the index one of the values
    control.timer.lap('read_and_transform_trace_prints')
    return sorted_trace_prints


class Accumulator(object):
    def __init__(self, output_path, initial_accumulator):
        self.output_path = output_path
        self.accumulator = copy.deepcopy(initial_accumulator)

    def get_cusip(self, trace_record):
        return trace_record['cusip']

    def get_trade_date(self, trace_record):
        'return date of trade:datetime.date'
        return trace_record['effectivedate'].date()

    def get_trade_datetime(self, trace_record):
        'return a datetime.datetime'
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
        pass

    def write(self):
        with open(self.output_path, 'wb') as f:
            pickle.dump(self.accumulator, f, pickle.HIGHEST_PROTOCOL)
        print 'wrote picked object of type %s to path %s' % (type(self.accumulator), self.output_path)

    def accumulate(self, trace_index, trace_record):
        raise TypeError('a subclass must override me')

    def print_table(self):
        raise TypeError('a subclass must override me')


class AccumulateCusips(Accumulator):
    'build Dict[cusip, issuer]'
    def __init__(self, output_path):
        super(AccumulateCusips, self).__init__(output_path, set())

    def accumulate(self, trace_index, trace_record):
        cusip = self.get_cusip(trace_record)
        self.accumulator.add(cusip)

    def print_table(self):
        print
        print 'cusips for the issuer identified in the trace print file name'
        print 'has %d entries' % len(self.accumulator)
        printed = 0
        for cusip in sorted(self.accumulator):
            print cusip
            printed += 1
            if printed >= 10:
                print '...'
                break


class AccumulateEffectivedateIssuepriceid(Accumulator):
    def __init__(self, output_path):
        super(AccumulateEffectivedateIssuepriceid, self).__init__(output_path, {})

    def accumulate(self, trace_index, trace_record):
        trade_date = self.get_trade_date(trace_record)
        if trade_date not in self.accumulator:
            self.accumulator[trade_date] = set()
        self.accumulator[trade_date].add(trace_index)

    def print_table(self):
        print
        print 'effectivedate -> set(issuepriceid)'
        print 'has %d rows' % len(self.accumulator)
        printed_trade_date = 0
        for trade_date in sorted(self.accumulator.keys()):
            trace_indices = self.accumulator[trade_date]
            print trade_date,
            printed_index = 0
            for trace_index in sorted(trace_indices):
                print trace_index,
                printed_index += 1
                if printed_index >= 6:
                    break
            if len(trace_indices) > 6:
                print '...'
            else:
                print
            printed_trade_date += 1
            if printed_trade_date >= 10:
                print '...'
                break


class AccumulateIssuepriceidCusip(Accumulator):
    'also determine that all of the trace_index values are distinct'
    def __init__(self, output_path):
        super(AccumulateIssuepriceidCusip, self).__init__(output_path, {})

    def accumulate(self, trace_index, trace_record):
        cusip = self.get_cusip(trace_record)
        assert trace_index not in self.accumulator
        self.accumulator[trace_index] = cusip

    def print_table(self):
        print
        print 'issuepriceid -> cusip'
        print 'has %d rows' % len(self.accumulator)
        printed = 0
        for trace_index in sorted(self.accumulator.keys()):
            cusip = self.accumulator[trace_index]
            print trace_index, cusip
            printed += 1
            if printed >= 10:
                print '...'
                break


class AccumulateIssuepriceidEffectivedate(Accumulator):
    def __init__(self, output_path):
        super(AccumulateIssuepriceidEffectivedate, self).__init__(output_path, {})

    def accumulate(self, trace_index, trace_record):
        trade_date = self.get_trade_date(trace_record)
        self.accumulator[trace_index] = trade_date

    def print_table(self):
        print
        print 'issuepriceid -> effective_date'
        print 'has %d rows' % len(self.accumulator)
        printed = 0
        for trace_index in sorted(self.accumulator.keys()):
            trade_date = self.accumulator[trace_index]
            print trace_index, trade_date
            printed += 1
            if printed >= 10:
                print '...'
                break


class AccumulatorCusipEffectivedatetimeIssuepriceid(Accumulator):
    def __init__(self, output_path):
        'accumulator will have type Dict[cusip, Dict[trade_datetime, Set(issuepriceid)]]'
        super(AccumulatorCusipEffectivedatetimeIssuepriceid, self).__init__(output_path, {})

    def accumulate(self, trace_index, trace_record):
        cusip = self.get_cusip(trace_record)
        trade_datetime = self.get_trade_datetime(trace_record)
        issuepriceid = trace_index
        if cusip not in self.accumulator:
            self.accumulator[cusip] = {}
        if trade_datetime not in self.accumulator[cusip]:
            self.accumulator[cusip][trade_datetime] = set()
        self.accumulator[cusip][trade_datetime].add(issuepriceid)

    def print_table(self):
        first_date = datetime.date(2017, 6, 26)
        interesting_cusips = ('037833AG5',)

        def report(title, f):
            'apply function f to each (cusip, trade_datetime, set(issuepriceid) that is selected'
            print
            print title
            print 'for dates startings %s' % first_date
            print 'for cusips in %s' % interesting_cusips
            print
            for cusip in sorted(self.accumulator.keys()):
                if cusip not in interesting_cusips:
                    continue
                d = self.accumulator[cusip]
                for trade_datetime in sorted(d.keys()):
                    if trade_datetime.date() < first_date:
                        continue
                    f(cusip, trade_datetime, d[trade_datetime])
            print

        def lines1(cusip, trade_datetime, issuepriceids):
            print cusip, trade_datetime,
            for issuepriceid in issuepriceids:
                print issuepriceid,
            print

        report(
            'cusip -> effective_datetime -> issuepriceid',
            lines1,
        )

        def lines2(cusip, trade_datetime, issuepriceids):
            if len(issuepriceids) > 2:
                print cusip, trade_datetime, len(issuepriceids)

        report(
            '(cusip -> effective_date -> n_trades) with more than one trade at that time',
            lines2,
        )


def do_work(control):
    'accumulate information on the trace prints for the issuer and write that info to the file system'
    def lap():
        'return ellapsed wall clock time:float since previous call to lap()'
        return control.timer.lap('lap', verbose=False)[1]

    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    # input files are for a specific ticker
    trace_prints = read_and_transform_trace_prints(control)

    counter = collections.Counter()

    # accumulate info for cusips
    accumulators = (
        AccumulatorCusipEffectivedatetimeIssuepriceid(control.path['out_cusip_effectivedatetime_issuepriceids']),
        AccumulateCusips(control.path['out_cusips']),
        AccumulateEffectivedateIssuepriceid(control.path['out_effectivedate_issuepriceid']),
        AccumulateIssuepriceidCusip(control.path['out_issuepriceid_cusip']),
        AccumulateIssuepriceidEffectivedate(control.path['out_issuepriceid_effectivedate']),
    )
    for trace_index, trace_record in trace_prints.iterrows():
        counter['n trace records read'] += 1
        for accumulator in accumulators:
            accumulator.accumulate(trace_index, trace_record)
    control.timer.lap('accumulated all info')

    # report counts
    print 'end loop on input in %.3f wall clock seconds' % lap()
    print 'counts'
    for k in sorted(counter.keys()):
        print '%71s: %6d' % (k, counter[k])

    for accumulator in accumulators:
        accumulator.print_table()
        accumulator.write()

    control.timer.lap('wrote output files')

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
