'''create feature sets for a CUSIP

INVOCATION
  python features.py {ticker}.csv [--cusip CUSIP] [--test] [--trace]

where
 {ticker}.csv is a CSV file in MidPredictors/data
 --cusip CUSIP means to create the targets only for the specified CUSIP
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python features.py orcl
 python features.py orcl --cusip 68389XAS4

INPUTS
 MidPredictor/{ticker}.csv

OUTPUTS
 WORKING/features/{ticker}-{cusip}.csv
where
 {cusip} is in the {ticker} file
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import datetime
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import arg_type
from Bunch import Bunch
import dirutility
from Logger import Logger
import seven.models as models
import seven
from seven.OrderImbalance4 import OrderImbalance4
import seven.path
from Timer import Timer


class Doit(object):
    def __init__(self, ticker, test=False, me='features'):
        self.ticker = ticker
        self.me = me
        self.test = test
        # define directories
        midpredictor = seven.path.midpredictor_data()
        working = seven.path.working()
        out_dir = os.path.join(working, me + ('-test' if test else ''))
        # read in CUSIPs for the ticker
        with open(os.path.join(working, 'cusips', ticker + '.pickle'), 'r') as f:
            self.cusips = pickle.load(f).keys()
        # path to files abd durecties
        self.in_ticker_filename = os.path.join(midpredictor, ticker + '.csv')
        self.out_cusips = [
            os.path.join(working, me, '%s-%s.csv' % (ticker, cusip))
            for cusip in self.cusips
        ]
        self.out_dir = out_dir
        self.out_log = os.path.join(out_dir, '0log.txt')
        # used by Doit tasks
        self.actions = [
            'python %s.py %s' % (me, ticker)
        ]
        self.targets = [
            self.out_cusips,
            self.out_log,
        ]
        self.file_dep = [
            self.me + '.py',
            self.in_ticker_filename,
        ]


def make_control(argv):
    'return a Bunch'

    print argv
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

    doit = Doit(arg.ticker, me=arg.me)
    dirutility.assure_exists(doit.out_dir)

    dir_working = seven.path.working()
    if arg.test:
        dir_out = os.path.join(dir_working, arg.me + '-test')
    else:
        dir_out = os.path.join(dir_working, arg.me)
    dirutility.assure_exists(dir_out)

    return Bunch(
        arg=arg,
        doit=doit,
        random_seed=random_seed,
        timer=Timer(),
    )


month_number = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}


def make_next_price(trade_type, datetime, index, df):
    mask = df.datetime > datetime
    after_current_trade = df.loc[mask]
    # sorted = after_current_trade.sort_values('datetime')
    for index, row in after_current_trade.iterrows():
        if row.trade_type == trade_type:
            return row.trade_price
    return None


def make_days_to_maturity(maturity_date, datetime):
    'return float'
    seconds_per_day = 24 * 60 * 60
    diff = maturity_date - datetime
    days = diff.days
    seconds = diff.seconds
    result = days + seconds / seconds_per_day
    return result


def to_datetime_time(s):
    'convert str like HH:MM:SS to datetime.time value'
    hour, minute, second = s.split(':')
    return datetime.time(
        int(hour),
        int(minute),
        int(second),
    )


def test_equal(a, b):
    'trace if a != b'
    if a != b:
        print 'not equal'
        print a
        print b
        pdb.set_trace()


def test_true(b):
    if not b:
        print 'not true'
        print b
        pdb.set_trace()


class Skipped(object):
    def __init__(self):
        self.counter = collections.Counter()
        self.n_skipped = 0

    def skip(self, reason):
        self.counter[reason] += 1
        self.n_skipped += 1


class Context(object):
    'accumulate running info for trades for a specific CUSIP'
    def __init__(self, lookback=None, typical_bid_offer=None, proximity_cutoff=None):
        self.order_imbalance4_object = OrderImbalance4(
            lookback=lookback,
            typical_bid_offer=typical_bid_offer,
            proximity_cutoff=proximity_cutoff,
        )
        self.order_imbalance4 = None
        self.prior_price_B = None
        self.prior_price_D = None
        self.prior_price_S = None
        self.prior_quantity_B = None
        self.prior_quantity_D = None
        self.prior_quantity_S = None

    def update(self, trade):
        self.order_imbalance4 = self.order_imbalance4_object.imbalance(
            trade_type=trade.trade_type,
            trade_quantity=trade.quantity,
            trade_price=trade.price,
        )
        if trade.trade_type == 'B':
            self.prior_price_B = trade.price
            self.prior_quantity_B = trade.quantity
        elif trade.trade_type == 'D':
            self.prior_price_D = trade.price
            self.prior_quantity_D = trade.quantity
        elif trade.trade_type == 'S':
            self.prior_price_S = trade.price
            self.prior_quantity_S = trade.quantity
        else:
            print trade
            print 'unknown trade_type', trade.trade_type
            pdb.set_trace()

    def missing_any_historic_price(self):
        return (
            self.prior_price_B is None or
            self.prior_price_D is None or
            self.prior_price_S is None
        )


def do_work(control):
    'write order imbalance for each trade in the input file
    def validate(df):
        assert (df.ticker == control.arg.ticker.upper()).all()

    # BODY STARTS HERE
    verbose = False
    df_trades = models.read_csv(
        control.doit.in_ticker_filename,
        parse_dates=['maturity', 'effectivedate', 'effectivetime'],
    )
    validate(df_trades)
    df['effectivedatetime'] = models.make_effectivedatetime(df_trades)

    result = {}   # Dict[cusip, pd.DataFrame]
    context = {}  # Dict[cusip, Context]
    order_imbalance4_hps = {
        'lookback': 10,
        'typical_bid_offer': 2,
        'proximity_cutoff': 20,
    }
    print 'creating feaures'
    count = 0
    for index, trade in df_trades.iterrows():
        count += 1
        if count % 1000 == 1:
            print 'count %d of %d' % (count, len(df_trades))
        if verbose:
            print 'index', index, trade.cusip, trade.trade_type
        cusip = trade.cusip
        if control.arg.cusip is not None:
            if cusip != control.arg.cusip:
                continue
        if cusip not in context:
            context[cusip] = Context(
                lookback=order_imbalance4_hps['lookback'],
                typical_bid_offer=order_imbalance4_hps['typical_bid_offer'],
                proximity_cutoff=order_imbalance4_hps['proximity_cutoff'],
            )
            result[cusip] = pd.DataFrame()
        cusip_context = context[cusip]
        cusip_context.update(trade)
        if cusip_context.missing_any_historic_price():
            print 'missing some historic prices', index, cusip
            continue
        next_row = models.make_features_dict(
            coupon=float(trade.coupon),
            cusip=trade.cusip,
            days_to_maturity=make_days_to_maturity(trade.maturity, trade.effectivedate),
            effectivedatetime=trade.effectivedatetime,
            order_imbalance4=cusip_context.order_imbalance4,
            prior_price_B=cusip_context.prior_price_B,
            prior_price_D=cusip_context.prior_price_D,
            prior_price_S=cusip_context.prior_price_S,
            prior_quantity_B=cusip_context.prior_quantity_B,
            prior_quantity_D=cusip_context.prior_quantity_D,
            prior_quantity_S=cusip_context.prior_quantity_S,
            trade_price=trade.price,
            trade_quantity=trade.quantity,
            trade_type_is_B=1 if trade.trade_type == 'B' else 0,
            trade_type_is_D=1 if trade.trade_type == 'D' else 0,
            trade_type_is_S=1 if trade.trade_type == 'S' else 0,
        )
        result[cusip] = result[cusip].append(
            pd.DataFrame(next_row, index=[index]),
            verify_integrity=True,
        )
    print 'writing result files'
    for cusip, df in result.iteritems():
        filename = '%s-%s.csv' % (control.arg.ticker, cusip)
        path = os.path.join(control.doit.out_dir, filename)
        df.to_csv(path)
        print 'wrote %d records to %s' % (len(df), filename)
    return None


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
