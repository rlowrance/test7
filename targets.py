'''create targets sets for all CUSIPS in a ticker file

INVOCATION
  python targets.py {ticker} [--cusip CUSIP] [--test] [--trace]

where
 {ticker}.csv is a CSV file in MidPredictors/data
 --cusip CUSIP means to create the targets only for the specified CUSIP
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python targets.py orcl.csv

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
import cPickle as pickle
import datetime
import os
import numpy as np
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import arg_type
from Bunch import Bunch
import dirutility
from Logger import Logger
import models
import seven
import seven.path
from Timer import Timer


class Doit(object):
    def __init__(self, ticker, test=False, me='targets'):
        self.ticker = ticker
        self.me = me
        self.test = test
        # define directories
        working = seven.path.working()
        midpredictor = seven.path.midpredictor_data()
        out_dir = os.path.join(working, me + ('-test' if test else ''))
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
        self.targets = self.out_targets.copy().add(self.out_log)
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
    parser.add_argument('ticker')
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
    dirutility.assure_exists(doit.out_dir)

    return Bunch(
        arg=arg,
        doit=doit,
        random_seed=random_seed,
        timer=Timer(),
    )


def do_work(control):
    'write order imbalance for each trade in the input file'
    def next_prices(df, index):
        'return Dict of next prices for each trade_type'
        mask = df.effectivedatetime > df.loc[index].effectivedatetime
        after_current_trade = df.loc[mask]
        result = {}
        for index, row in after_current_trade.iterrows():
            if row.trade_type not in result:
                result[row.trade_type] = row.price
            if len(result) == 3:
                # stop when we have a D, B, and S trade
                break
        return result

    def validate(df):
        'Raise if a problem'
        assert (df.ticker == control.arg.ticker.upper()).all()

    # BODY STARTS HERE
    # read and transform the input ticker file
    # NOTE: if usecols is supplied, then the file is not read correctly
    df_ticker = models.read_csv(
        control.doit.in_ticker,
        parse_dates=['effectivedate', 'effectivetime'],
        # usecols=['cusip', 'price', 'effectivedate', 'effectivetime', 'ticker', 'trade_type'],
    )
    print 'read %d input trades' % len(df_ticker)
    cusips = set(df_ticker.cusip)
    print 'containing %d CUSIPS' % len(cusips)
    validate(df_ticker)
    df_ticker['effectivedatetime'] = models.make_effectivedatetime(df_ticker)
    del df_ticker['effectivedate']
    del df_ticker['effectivetime']
    del df_ticker['ticker']

    # create a result file for each CUSIP in the input ticker file
    for i, cusip in enumerate(cusips):
        print 'cusip %s %d of %d' % (cusip, i + 1, len(cusips))
        if control.arg.cusip is not None:
            if cusip != control.arg.cusip:
                print 'skipping', cusip
                continue
        mask = df_ticker.cusip == cusip
        df_cusip_unsorted = df_ticker.loc[mask]
        df_cusip = df_cusip_unsorted.sort_values(by='effectivedatetime')
        result = pd.DataFrame(
            columns=['next_price_B', 'next_price_D', 'next_price_S'],
            index=df_cusip.index,
        )  # pre-allocate for speed
        for index, row in df_cusip.iterrows():
            prices = next_prices(df_cusip, index)
            next_row = (prices.get('B', np.nan), prices.get('D', np.nan), prices.get('S', np.nan))
            result.loc[index] = next_row
        print 'cusip %s len result %d' % (cusip, len(result))
        path = os.path.join(control.doit.out_dir, '%s-%s.csv' % (control.arg.ticker, cusip))
        print 'writing to', path
        result.to_csv(path)
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
