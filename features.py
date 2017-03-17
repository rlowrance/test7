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

FEATURES CREATED BY INPUT FILE AND NEXT STEPS (where needed)
  {ticker}.csv: trade info by effectivedatetime for many CUSIPs
    oasspread
    order_imbalance4
    prior_oasspread_{trade_type}
    prior_quantity_{trade_type}
    quantity
    trade_type_is_{trade_type}

for
 trade_type in {B, D, S}
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import datetime
import numbers
import numpy as np
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
from seven.FeatureMakers import FeatureMakerTicker, DeltaPrice
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
        # path to files and durecties
        self.in_ticker_equity_ohlc = os.path.join(midpredictor, ticker + '_equity_ohlc.csv')
        self.in_security_master = os.path.join(midpredictor, ticker + '_sec_master.csv')
        self.in_spx_equity_ohlc = os.path.join(midpredictor, 'spx_equity_ohlc.csv')
        self.in_ticker = os.path.join(midpredictor, ticker + '.csv')
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
            self.in_ticker_equity_ohlc,
            self.in_security_master,
            self.in_spx_equity_ohlc,
            self.in_ticker,
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


def append_feature(d, cusip, feature_name, feature_value):
        if cusip not in d:
            d[cusip] = collections.defaultdict(list)
        if np.isnan(feature_value):
            print 'error: feature %s is NaN' % feature_name
            pdb.set_trace()
        if feature_value is None:
            print 'error: feature %s is None' % feature_name
            pdb.set_trace()
        if not isinstance(feature_value, numbers.Number):
            print 'error: feature %s is %s of type %s, which is not a number' % (
                feature_name,
                type(feature_name),
                feature_value,
            )
            pdb.set_trace()
        if not isinstance(feature_name, str):
            print 'error: feature name is %s, which is a %s, not a string' % (
                feature_name,
                type(feature_name),
            )
        d[cusip][feature_name].append(feature_value)


def append_features(d, cusip, input_record, feature_maker_class):
    'mutate d:Dict to include features possibly created by make_class.make_features(cusip, input_record)'
    new_features = feature_maker_class.make_features(cusip, input_record)
    if new_features is not None:
        for feature_name, feature_value in new_features.iteritems():
            append_feature(d, cusip, feature_name, feature_value)
    return new_features


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
    df_ticker = models.read_csv(
        control.doit.in_ticker,
        parse_dates=['maturity', 'effectivedate', 'effectivetime'],
        nrows=1000 if control.arg.test else None,
        verbose=True,
    )
    validate_trades(df_ticker)
    df_ticker['effectivedatetime'] = models.make_effectivedatetime(df_ticker)

    df_security_master = models.read_csv(
        control.doit.in_security_master,
        parse_dates=['issue_date', 'maturity_date'],
        verbose=True,
    )
    print len(df_security_master)
    df_security_master['cusip'] = df_security_master.index

    def read_equity_ohlc(path):
        return models.read_csv(
            path,
            parse_dates=[0],
            verbose=True
        )

    delta_price = DeltaPrice(
        df_ticker=read_equity_ohlc(control.doit.in_ticker_equity_ohlc),
        df_spx=read_equity_ohlc(control.doit.in_spx_equity_ohlc),
    )

    print 'creating feaures'
    count = 0
    cusips_not_in_security_master = set()
    maturity_dates_bad = set()
    d = {}  # Dict[cusip, Dict[feature_name, feature_values]], maintained by append_features()
    feature_maker_ticker = FeatureMakerTicker(
        order_imbalance4_hps={
            'lookback': 10,
            'typical_bid_offer': 2,
            'proximity_cutoff': 20,
        },
    )
    all_feature_makers = (
        feature_maker_ticker,
    )
    for index, trade in df_ticker.iterrows():
        count += 1
        if count % 1000 == 1:
            print 'trade count %d of %d' % (count, len(df_ticker))
        if verbose:
            print 'index', index, trade.cusip, trade.trade_type
        cusip = trade.cusip
        if control.arg.cusip is not None:
            if cusip != control.arg.cusip:
                print 'skipping cusip %s, because of --cusip arg' % cusip
                continue
        appended = append_features(d, cusip, trade, feature_maker_ticker)
        if appended is not None:
            append_feature(d, cusip, 'id_index', index)
            append_feature(d, cusip, 'id_cusip', trade.cusip)
            append_feature(d, cusip, 'id_effectivedatetime', trade.effectivedatetime)
        # what about trades that make_ticker creates but some other maker does not?
        continue
        # OLD BELOW ME
        if cusip not in context_cusip:
            context_cusip[cusip] = TickerContextCusip(
                lookback=order_imbalance4_hps['lookback'],
                typical_bid_offer=order_imbalance4_hps['typical_bid_offer'],
                proximity_cutoff=order_imbalance4_hps['proximity_cutoff'],
            )
            result[cusip] = pd.DataFrame()
        cusip_context = context_cusip[cusip]
        cusip_context.update(trade)
        if cusip_context.missing_any_historic_oasspread():
            skipping(cusip, index, 'missing some historic prices')
            continue
        if cusip not in df_security_master.cusip:
            if cusip not in cusips_not_in_security_master:
                skipping(cusip, index, 'not in security master')
                cusips_not_in_security_master.add(cusip)
            continue
        # features from {ticker}_equity_ohlc and spx_equity_ohlc
        trade_date = trade.effectivedate.date()
        if not delta_price.is_available(trade_date):
            skipping(cusip, index, 'delta_price not available for %s' % trade_date)
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
        assert coupon_type in ('FIXED', 'FLOATING')  # TODO: add stepup and possibly others
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
            # features from {ticker}_equity_ohlc and spx_equity_ohlc
            price_delta_ratio_1_day=delta_price.ratio(trade_date, 1),
            price_delta_ratio_2_days=delta_price.ratio(trade_date, 2),
            price_delta_ratio_3_days=delta_price.ratio(trade_date, 3),
            price_delta_ratio_week=delta_price.ratio(trade_date, 5),
            price_delta_ratio_month=delta_price.ratio(trade_date, 20),
            # features from {ticker}_security_master
            amount_issued=amount_issued,
            collateral_type_is_sr_unsecured=collateral_type == 'SR UNSECURED',
            coupon_is_fixed=coupon_type == 'FIXED',
            coupon_is_floating=coupon_type == 'FLOATING',
            coupon_current=get_security_master(cusip, 'curr_cpn'),
            is_callable=is_callable == 'TRUE',
            months_to_maturity=months_to_maturity,
            # features from {ticker}.csv
            order_imbalance4=cusip_context.order_imbalance4,
            prior_oasspread_B=cusip_context.prior_oasspread_B,
            prior_oasspread_D=cusip_context.prior_oasspread_D,
            prior_oasspread_S=cusip_context.prior_oasspread_S,
            prior_quantity_B=cusip_context.prior_quantity_B,
            prior_quantity_D=cusip_context.prior_quantity_D,
            prior_quantity_S=cusip_context.prior_quantity_S,
            trade_oasspread=trade.oasspread,
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
    # maybe drop records that have any NaN value
    for cusip in d.keys():
        df = pd.DataFrame(
            data=d[cusip],
            index=d[cusip]['index']
        )
        filename = '%s-%s.csv' % (control.arg.ticker, cusip)
        path = os.path.join(control.doit.out_dir, filename)
        df.to_csv(path)
        print 'wrote %d records to %s' % (len(df), filename)
    print 'reasons records skipped'
    for maker in (feature_maker_ticker,):
        for reason, count in maker.skipped_reasons.iteritems():
            print maker.input_file, reason, count
    print 'number of feature records created (assuming no interactions with other input files)'
    for make in all_feature_makers:
        print maker.input_file, maker.count_created

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
    # print control
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflake warnings for debugging tools
        pdb.set_trace()
        pprint()

    main(sys.argv)
