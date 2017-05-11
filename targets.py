'''create targets sets for all CUSIPS in a ticker file

INVOCATION
  python targets.py {ticker} [--cusip CUSIP] [--test] [--trace]

where
 {ticker} is a ticker; ex; orcl
 --cusip CUSIP means to create the targets only for the specified CUSIP
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python targets.py orcl --cusip 68389XAS4 --test
 python targets.py orcl

see build.py for inputs and outputs
'''

from __future__ import division

import argparse
import collections
import os
import numpy as np
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import applied_data_science.dirutility

from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.arg_type as arg_type
import seven.feature_makers as feature_makers
# from seven.FeatureMakers import FeatureMakerTradeId
import seven.read_csv as read_csv

import build


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', type=arg_type.ticker)
    parser.add_argument('--cusip', action='store', type=arg_type.cusip)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = build.targets(arg.ticker, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=Timer(),
    )


def do_work(control):
    'write target values for each trade in the ticker file'
    def validate(df):
        'Raise if a problem'
        assert (df.ticker == control.arg.ticker.upper()).all()

    def append_info(field_id, trace_record, d):
        'mutate d by appending certain info from the trace_record'
        def field_name(s):
            return 'info_%s_%s' % (field_id, s)

        for info_field_name in ('effectivedatetime', 'quantity', 'oasspread', 'trade_type'):
            d[field_name(info_field_name)].append(trace_record[info_field_name])

    # BODY STARTS HERE
    # read and transform the input ticker file
    # NOTE: if usecols is supplied, then the file is not read correctly
    df_trace = read_csv.input(
        control.arg.ticker,
        'trace',
        nrows=10 if control.arg.test else None,
    )
    print 'read %d input trades' % len(df_trace)
    cusips = set(df_trace.cusip)
    print 'containing %d CUSIPS' % len(cusips)
    validate(df_trace)
    df_trace['effectivedatetime'] = feature_makers.make_effectivedatetime(df_trace)
    del df_trace['effectivedate']
    del df_trace['effectivetime']
    del df_trace['ticker']

    # create a result file for each CUSIP in the input ticker file
    for i, cusip in enumerate(cusips):
        print 'cusip %s %d of %d' % (cusip, i + 1, len(cusips))
        if control.arg.cusip is not None:
            if cusip != control.arg.cusip:
                print 'skipping', cusip
                continue
        mask = df_trace.cusip == cusip
        df_trace_cusip_unsorted = df_trace.loc[mask]
        df_trace_cusip_sorted = df_trace_cusip_unsorted.sort_values(by='effectivedatetime')

        # build up information to create a date frame
        d = collections.defaultdict(list)  # accumulate columns of new DataFrame here
        indices = []
        missing_oasspreads = set()  # trace_indices that have missing oasspread values
        for this_integer_location in xrange(0, len(df_trace_cusip_sorted) - 1, 1):
            # don't create targets for the last trade, as there is nothing to predict for it
            this_trace_index = df_trace_cusip_sorted.index[this_integer_location]
            this_trace_record = df_trace_cusip_sorted.iloc[this_integer_location]
            indices.append(this_trace_index)
            d['id_trace_index'].append(this_trace_index)
            append_info('this', this_trace_record, d)  # mutate d

            # determine target oasspread for each trade type by examining future trades
            future_trace_print_found = False
            next_oasspreads = {}
            for next_integer_location in xrange(this_integer_location + 1, len(df_trace_cusip_sorted), 1):
                next_trace_index = df_trace_cusip_sorted.index[next_integer_location]
                next_trace_record = df_trace_cusip_sorted.iloc[next_integer_location]
                if not future_trace_print_found:
                    # this info is needed to measure the accuracy
                    append_info('subsequent', next_trace_record, d)  # mutate d
                    future_trace_print_found = True
                next_trade_type = next_trace_record['trade_type']
                if next_trade_type not in next_oasspreads:
                    oasspread = next_trace_record['oasspread']
                    if np.isnan(oasspread):
                        missing_oasspreads.add(next_trace_index)
                    next_oasspreads[next_trade_type] = oasspread
                if len(next_oasspreads) == 3:
                    break  # have gathered oasspreads for 'B', 'D', and 'S' trade types
            if future_trace_print_found:
                # we found subsequent records
                d['target_next_oasspread_B'].append(next_oasspreads.get('B', np.nan))
                d['target_next_oasspread_D'].append(next_oasspreads.get('D', np.nan))
                d['target_next_oasspread_S'].append(next_oasspreads.get('S', np.nan))

        if len(missing_oasspreads) == 0:
            print 'found no missing oasspreads'
        else:
            print 'found %d missing oasspreads for these trace indices' % len(missing_oasspreads)
            for index in missing_oasspreads:
                print ' ', index

        # check lengths
        len_index = len(indices)
        print len_index
        for k, v in d.iteritems():
            print k, len(v)
            assert len_index == len(v)

        result = pd.DataFrame(
            data=d,
            index=indices,
        )
        print 'cusip %s len result %d' % (cusip, len(result))
        path = os.path.join(control.path['dir_out'], cusip + '.csv')
        result.to_csv(path)
        print 'wrote %d records to %s' % (len(result), path)
    return


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    print control
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflake warnings for debugging tools
        pdb.set_trace()
        pprint()

    main(sys.argv)
