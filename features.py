'''create feature sets for a CUSIP

INVOCATION
  python features.py {ticker}.csv [--cusip CUSIP] [--test] [--trace]

where
 {ticker}.csv is a CSV file in MidPredictors/data
 --cusip CUSIP means to create the targets only for the specified CUSIP
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python features.py orcl.csv

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
import seven
from seven.OrderImbalance4 import OrderImbalance4
import seven.path
from Timer import Timer


def make_control(argv):
    'return a Bunch'

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker_filename', type=arg_type.filename_csv)
    parser.add_argument('--cusip', action='store', type=arg_type.cusip)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])
    arg.me = parser.prog.split('.')[0]
    arg.ticker = arg.ticker_filename.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    dir_working = seven.path.working()
    if arg.test:
        dir_out = os.path.join(dir_working, arg.me + '-test')
    else:
        dir_out = os.path.join(dir_working, arg.me)
    dirutility.assure_exists(dir_out)

    return Bunch(
        arg=arg,
        path_in_ticker_filename=os.path.join(seven.path.midpredictor_data(), arg.ticker_filename),
        path_out_result=dir_out,  # file {cusip}.csv is created here
        path_out_log=os.path.join(dir_out, '0log-' + arg.ticker + '.txt'),
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
        self.last_B_price = None
        self.last_D_price = None
        self.last_S_price = None

    def update(self, trade):
        self.order_imbalance4 = self.order_imbalance4_object.imbalance(
            trade_type=trade.trade_type,
            trade_quantity=trade.quantity,
            trade_price=trade.price,
        )
        if trade.trade_type == 'B':
            self.last_B_price = trade.price
        elif trade.trade_type == 'D':
            self.last_D_price = trade.price
        elif trade.trade_type == 'S':
            self.last_S_price = trade.price
        else:
            print trade
            print 'unknown trade_type', trade.trade_type
            pdb.set_trace()

    def missing_any_historic_price(self):
        return (
            self.last_B_price is None or
            self.last_D_price is None or
            self.last_S_price is None
        )


def do_work(control):
    'write order imbalance for each trade in the input file'
    def read_csv(path, parse_dates=None):
        df = pd.read_csv(
            path,
            index_col=0,
            nrows=100 if control.arg.test else None,
            usecols=None,
            low_memory=False,
            parse_dates=parse_dates,
        )
        print 'read %d rows from file %s' % (len(df), path)
        print df.columns
        return df

    def add_effectivedatetime(df):
        'create new column that combines the effectivedate and effective time'
        values = []
        for the_date, the_time in zip(df.effectivedate, df.effectivetime):
            values.append(datetime.datetime(
                the_date.year,
                the_date.month,
                the_date.day,
                the_time.hour,
                the_time.minute,
                the_time.second,
            ))
        df['effectivedatetime'] = pd.Series(values, index=df.index)

    def validate(df):
        assert (df.ticker == control.arg.ticker.upper()).all()

    # BODY STARTS HERE
    verbose = False
    df_trades = read_csv(
        control.path_in_ticker_filename,
        parse_dates=['maturity', 'effectivedate', 'effectivetime'],
    )
    validate(df_trades)
    add_effectivedatetime(df_trades)

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
            continue
        next_row = {  # create at least the features in models.features
            'ticker_file_index': index,
            'effectivedatetime': trade.effectivedatetime,
            'trade_type': trade.trade_type,
            'trade_quantity': trade.quantity,
            'trade_price': trade.price,
            'last_B_price': cusip_context.last_B_price,
            'last_D_price': cusip_context.last_D_price,
            'last_S_price': cusip_context.last_S_price,
            'coupon': float(trade.coupon),
            'days_to_maturity': make_days_to_maturity(trade.maturity, trade.effectivedate),
            'order_imbalance': cusip_context.order_imbalance4,
        }
        result[cusip] = result[cusip].append(
            pd.DataFrame(next_row, index=[index]),
            verify_integrity=True,
        )
        if verbose or len(result[cusip]) % 100 == 0:
            print index, cusip, len(result[cusip])
    print 'writing result files'
    for cusip, df in result.iteritems():
        filename = '%s-%s.csv' % (control.arg.ticker, cusip)
        path = os.path.join(control.path_out_result, filename)
        df.to_csv(path)
        print 'wrote %d records to %s' % (len(df), filename)
    return None


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path_out_log)  # now print statements also write to the log file
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
