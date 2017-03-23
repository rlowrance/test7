'''create targets sets for all CUSIPS in a ticker file

INVOCATION
  python targets.py {ticker} [--cusip CUSIP] [--test] [--trace]

where
 {ticker}.csv is a CSV file in MidPredictors/data
 --cusip CUSIP means to create the targets only for the specified CUSIP
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python targets.py orcl --cusip 68389XAS4 --test
 python targets.py orcl

INPUTS
 MidPredictor/{ticker}.csv

where
 {cusip} is identified in the {trade-print-filename}

OUTPUTS
 WORKING/targets/{ticker}-{cusip}.csv: columns: next_B_price, next_D_price, next_S_price index:<from ticker>
   One file for each cusip in the ticker
 WORKING/tagets/0log-{ticker}.txt

'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
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
import seven
import seven.models
import seven.path


class Doit(object):
    def __init__(self, ticker, test=False, me='targets'):
        self.ticker = ticker
        self.me = me
        self.test = test
        # define directories
        working = seven.path.working()
        midpredictor = seven.path.midpredictor_data()
        out_dir = os.path.join(
            working,
            '%s-%s' % (me, ticker) + ('-test' if test else '')
        )
        with open(os.path.join(working, 'cusips', ticker + '.pickle'), 'r') as f:
            self.cusips = pickle.load(f).keys()
        # path to files abd durecties
        self.in_ticker = os.path.join(midpredictor, '%s.csv' % ticker)

        self.out_dir = out_dir
        self.out_targets = {
            os.path.join(working, me, '%s-%s.csv' % (ticker, cusip))
            for cusip in self.cusips
        }
        self.out_log = os.path.join(out_dir, '0log.txt')

        # used by Doit tasks
        self.actions = [
            'python %s.py %s' % (me, ticker)
        ]
        self.targets = list(self.out_targets)
        self.targets.append(self.out_log)
        self.file_dep = [
            self.me + '.py',
            self.in_ticker,
        ]

    def __str__(self):
        for k, v in self.__dict__.iteritems():
            print 'doit.%s = %s' % (k, v)
        return self.__repr__()


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', type=arg_type.ticker)
    parser.add_argument('--cusip', action='store', type=arg_type.cusip)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])
    arg.me = parser.prog.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    doit = Doit(arg.ticker)
    applied_data_science.dirutility.assure_exists(doit.out_dir)

    return Bunch(
        arg=arg,
        doit=doit,
        random_seed=random_seed,
        timer=Timer(),
    )


def do_work(control):
    'write target values for each trade in the ticker file'
    def compare(next_spreads, last_spreads, trade_type, f):
        if trade_type in next_spreads and trade_type in last_spreads:
            return f(next_spreads[trade_type], last_spreads[trade_type])
        else:
            return False

    def decreased(next_spreads, last_spreads, trade_type):
        return compare(next_spreads, last_spreads, trade_type, lambda next, last: next < last)

    def increased(next_spreads, last_spread, trade_type):
        return compare(next_spreads, last_spreads, trade_type, lambda next, last: next > last)

    def make_next_spreads(df, index):
        'return (Dict of next OAS spreads for each trade_type, 2 x list of indices of trades without oasspreads'
        mask = df.effectivedatetime > df.loc[index].effectivedatetime
        after_current_trade = df.loc[mask]
        after_current_trade_sorted = after_current_trade.sort_values(by='effectivedatetime')
        result = {}
        have_price_and_no_spread = []
        have_no_price_and_no_spread = []
        for index, row in after_current_trade_sorted.iterrows():
            if np.isnan(row.oasspread):
                if np.isnan(row.price):
                    print 'missing price and oassspread', index
                    have_no_price_and_no_spread.append(index)
                else:
                    have_price_and_no_spread.append(index)
            elif row.trade_type not in result:
                # first oasspread for the row.trade_type
                result[row.trade_type] = row.oasspread
            else:
                # have already seen a next spread for the row.trade_type
                pass
            if len(result) == 3:
                # stop when we have the next oasspread for a D, B, and S trade
                break
        return result, have_price_and_no_spread, have_no_price_and_no_spread

    def validate(df):
        'Raise if a problem'
        assert (df.ticker == control.arg.ticker.upper()).all()

    # BODY STARTS HERE
    # read and transform the input ticker file
    # NOTE: if usecols is supplied, then the file is not read correctly
    df_ticker = seven.models.read_csv(
        control.doit.in_ticker,
        nrows=10 if control.arg.test else None,
        parse_dates=['effectivedate', 'effectivetime'],
        # usecols=['cusip', 'price', 'effectivedate', 'effectivetime', 'ticker', 'trade_type'],
    )
    print 'read %d input trades' % len(df_ticker)
    cusips = set(df_ticker.cusip)
    print 'containing %d CUSIPS' % len(cusips)
    validate(df_ticker)
    df_ticker['effectivedatetime'] = seven.models.make_effectivedatetime(df_ticker)
    del df_ticker['effectivedate']
    del df_ticker['effectivetime']
    del df_ticker['ticker']

    # create a result file for each CUSIP in the input ticker file
    for i, cusip in enumerate(cusips):
        print 'cusip %s %d of %d' % (cusip, i + 1, len(cusips))
        if False and cusip == '68389XAS4':
            print 'found', cusip
            pdb.set_trace()
        if control.arg.cusip is not None:
            if cusip != control.arg.cusip:
                print 'skipping', cusip
                continue
        mask = df_ticker.cusip == cusip
        df_cusip_unsorted = df_ticker.loc[mask]
        df_cusip = df_cusip_unsorted.sort_values(by='effectivedatetime')
        last_spreads = {}  # build up the oasspreads of the last trades
        d = collections.defaultdict(list)  # accumulate columns of new DataFrame here
        indices = []
        all_have_no_price_and_no_spread = set()  # items are indices
        all_have_price_and_no_spread = set()     # items are indices
        for index, row in df_cusip.iterrows():
            last_spreads[row.trade_type] = row.oasspread
            next_spreads, have_price_and_no_spread, have_no_price_and_no_spread = make_next_spreads(df_cusip, index)
            all_have_price_and_no_spread.update(have_price_and_no_spread)
            all_have_no_price_and_no_spread.update(have_no_price_and_no_spread)
            # all these values are possible NaN
            d['oasspread_D'].append(next_spreads.get('D', np.nan))
            d['oasspread_S'].append(next_spreads.get('S', np.nan))
            d['B_spread_increased'].append(increased(next_spreads, last_spreads, 'B'))
            d['B_spread_decreased'].append(decreased(next_spreads, last_spreads, 'B'))
            d['D_spread_increased'].append(increased(next_spreads, last_spreads, 'D'))
            d['D_spread_decreased'].append(decreased(next_spreads, last_spreads, 'D'))
            d['S_spread_increased'].append(increased(next_spreads, last_spreads, 'S'))
            d['S_spread_decreased'].append(decreased(next_spreads, last_spreads, 'S'))
            indices.append(index)
        print 'cusip %s; fraction with price and no spread: %f' % (
            cusip,
            len(all_have_price_and_no_spread) * 1.0 / len(df_cusip),
        )
        print all_have_price_and_no_spread
        print 'cusip %s; fraction with no price and no spread: %f' % (
            cusip,
            len(all_have_no_price_and_no_spread) * 1.0 / len(df_cusip),
        )
        print all_have_no_price_and_no_spread

        result = pd.DataFrame(
            data=d,
            index=indices,
        )
        print 'cusip %s len result %d' % (cusip, len(result))
        path = os.path.join(control.doit.out_dir, '%s-%s.csv' % (control.arg.ticker, cusip))
        result.to_csv(path)
        print 'wrote %d records to %s' % (len(result), path)
    return


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.doit.out_log)  # now print statements also write to the log file
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
