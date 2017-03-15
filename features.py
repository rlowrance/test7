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
 WORKING/features-{ticker}/{ticker}-{cusip}.csv
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

import applied_data_science
import applied_data_science.dirutility

from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.arg_type as arg_type
import seven.models as models
import seven
from seven.OrderImbalance4 import OrderImbalance4
import seven.path


class Doit(object):
    def __init__(self, ticker, test=False, me='features'):
        self.ticker = ticker
        self.me = me
        self.test = test
        # define directories
        midpredictor = seven.path.midpredictor_data()
        working = seven.path.working()
        out_dir = os.path.join(working, '%s-%s' % (me, ticker) + ('-test' if test else ''))
        # read in CUSIPs for the ticker
        with open(os.path.join(working, 'cusips', ticker + '.pickle'), 'r') as f:
            self.cusips = pickle.load(f).keys()
        # path to files abd durecties
        self.in_security_master = os.path.join(midpredictor, ticker + '_sec_master.csv')
        self.in_ticker_filename = os.path.join(midpredictor, ticker + '.csv')
        self.out_cusips = [
            os.path.join(out_dir, '%s-%s.csv' % (ticker, cusip))
            for cusip in self.cusips
        ]
        self.out_dir = out_dir
        self.out_log = os.path.join(out_dir, '0log.txt')
        # used by Doit tasks
        self.actions = [
            'python %s.py %s' % (me, ticker)
        ]
        self.targets = [self.out_log]
        for cusip in self.out_cusips:
            self.targets.append(cusip)
        self.file_dep = [
            self.me + '.py',
            self.in_security_master,
            self.in_ticker_filename,
        ]

    def __str__(self):
        for k, v in self.__dict__.iteritems():
            print 'doit.%s = %s' % (k, v)
        return self.__repr__()


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
    applied_data_science.dirutility.assure_exists(doit.out_dir)

    dir_working = seven.path.working()
    if arg.test:
        dir_out = os.path.join(dir_working, arg.me + '-test')
    else:
        dir_out = os.path.join(dir_working, arg.me)
    applied_data_science.dirutility.assure_exists(dir_out)

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
    'write order imbalance for each trade in the input file'
    def validate_trades(df):
        assert (df.ticker == control.arg.ticker.upper()).all()

    def get_security_master(cusip, field_name):
        'return value in the security master'
        assert field_name in df_security_master.columns
        relevant = df_security_master.loc[df_security_master.cusip == cusip]
        if len(relevant) != 1:
            print 'did not find exactly one row for the cusip'
            print cusip
            print df_security_master.cusip
            pdb.set_trace()
        return relevant[field_name][0]

    def months_from_until(a, b):
        'return months from date a to date b'
        delta_days = (b - a).days
        return delta_days / 30.0

    def skipping(cusip, index, msg):
        print 'skipping cusip %s index %s: %s' % (cusip, index, msg)

    # BODY STARTS HERE
    verbose = False

    # read {ticker}.csv
    df_trades = models.read_csv(
        control.doit.in_ticker_filename,
        parse_dates=['maturity', 'effectivedate', 'effectivetime'],
        nrows=1000 if control.arg.test else None,
        verbose=True,
    )
    validate_trades(df_trades)
    df_trades['effectivedatetime'] = models.make_effectivedatetime(df_trades)

    df_security_master = models.read_csv(
        control.doit.in_security_master,
        parse_dates=['issue_date', 'maturity_date'],
        verbose=True,
    )
    print len(df_security_master)
    df_security_master['cusip'] = df_security_master.index

    result = {}   # Dict[cusip, pd.DataFrame]
    context = {}  # Dict[cusip, Context]
    order_imbalance4_hps = {
        'lookback': 10,
        'typical_bid_offer': 2,
        'proximity_cutoff': 20,
    }
    print 'creating feaures'
    count = 0
    cusips_not_in_security_master = set()
    maturity_dates_bad = set()
    for index, trade in df_trades.iterrows():
        count += 1
        if count % 1000 == 1:
            print 'trade count %d of %d' % (count, len(df_trades))
        if verbose:
            print 'index', index, trade.cusip, trade.trade_type
        cusip = trade.cusip
        if control.arg.cusip is not None:
            if cusip != control.arg.cusip:
                print 'skipping cusip %s, because of --cusip arg' % cusip
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
            skipping(cusip, index, 'missing some historic prices')
            continue
        if cusip not in df_security_master.cusip:
            if cusip not in cusips_not_in_security_master:
                skipping(cusip, index, 'not in security master')
                cusips_not_in_security_master.add(cusip)
            continue
        # features from the security master file
        # TODO: receive version with an effectivedate, then add these features:
        #  amount_outstanding
        #  fraction_of_issue_outstanding
        # TODO: receive updated version of security master, then add these features
        #  is_puttable
        amount_issued = get_security_master(cusip, 'issue_amount')
        collateral_type = get_security_master(cusip, 'COLLAT_TYP')
        assert collateral_type in ('SR UNSECURED',)
        is_callable = get_security_master(cusip, 'is_callable')
        maturity_date = get_security_master(cusip, 'maturity_date')
        coupon_type = get_security_master(cusip, 'CPN_TYP')
        assert coupon_type in ('FIXED', 'FLOATING')
        months_to_maturity = months_from_until(trade.effectivedate, maturity_date)
        if months_to_maturity < 0:
            if maturity_date not in maturity_dates_bad:
                skipping(
                    cusip,
                    index,
                    'negative months to maturity; trade.effectivedate %s maturity_date %s' % (
                        trade.effectivedate,
                        maturity_date,
                    ),
                )
                maturity_dates_bad.add(maturity_date)
            continue
        next_row = models.make_features_dict(
            id_cusip=trade.cusip,
            id_effectivedatetime=trade.effectivedatetime,
            # features from df_security_master
            amount_issued=amount_issued,
            collateral_type_is_sr_unsecured=collateral_type == 'SR UNSECURED',
            coupon_is_fixed=coupon_type == 'FIXED',
            coupon_is_floating=coupon_type == 'FLOATING',
            coupon_current=get_security_master(cusip, 'curr_cpn'),
            is_callable=is_callable == 'TRUE',
            months_to_maturity=months_to_maturity,
            # features from df_trades
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
    print 'unusual conditions'
    print '%d cusips not in security master' % len(cusips_not_in_security_master)
    for missing_cusip in cusips_not_in_security_master:
        print ' ', missing_cusip
    print '%d maturity dates in the security master were bad' % len(maturity_dates_bad)
    for bad_maturity_date in maturity_dates_bad:
        print ' ', bad_maturity_date
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
