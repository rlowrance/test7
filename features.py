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
 python features.py orcl --cusip 68389XAS4  --test
 python features.py orcl --cusip 68389XAP0  --test

INPUTS
 MidPredictor/{ticker}.csv

OUTPUTS
 WORKING/features-{ticker}/{ticker}-{cusip}.csv
where
 {cusip} is in the {ticker} file

FEATURES CREATED BY INPUT FILE AND NEXT STEPS (where needed)
  comparables_{ticker}.csv    TODO GC to provide file with time stamps and related CUSIPs
                              then RL to create features
    is_on_the_run                    0 or 1
    oftr_{oftr_cardinal}-{feature}   missing if bond is on the run; redefine feature sets when missing
    <other features to be designed by GC>

  {ticker}.csv: trade info by effectivedatetime for many CUSIPs in the file
    oasspread
    order_imbalance4
    prior_oasspread_{trade_type}
    prior_quantity_{trade_type}
    quantity
    trade_type_is_{trade_type}

  {ticker}_equity_ohlc.csv and spx_equity_ohlc.csv
    price_delta_ratio_back_days_{back}  (ticker price delta / market price delta) for {back} market days
    price_delta_ratio_back_hours_1      same ratio, back one hour; TODO: create after GC provides data

  {ticker}_fund.csv  TODO GC to provide file with usable dates (or decode current dates)
    debt_to_market_cap
    debt_to_ltm_ebitda
    interest_coverage
    <other features specified in the google doc>

  {ticker}_sec_master.csv   GC to update file
    amount_issued
    collateral_type_is_sr_unsecured
    coupon_is_fixed
    coupon_is_floating
    coupon_current
    fraction_still_outstanding  RL to add once file provided by GC; should be in separate file
    is_callable
    is_puttable              TODO add once file is updated by GC (note is in {ticker}_comparables_sec_master)
    months_to_maturity

for
 back in {1, 2, ..., 30}
 {feature}                         feature of a corresponding on-the-run bond
 oftr_cardinal in {1, 2, 3}        if bond is off the run, corresponding CUSIPs ordinal rank
                                   where rank == 1 is the most similar on-the-run CUSIP
 trade_type in {B, D, S}
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import numbers
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
from seven.FeatureMakers import FeatureMakerOhlc
from seven.FeatureMakers import FeatureMakerSecurityMaster
from seven.FeatureMakers import FeatureMakerTicker
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


def append(d, cusip, features):
    if cusip not in d:
        d[cusip] = collections.defaultdict(list)
    for feature_name, feature_value in features.iteritems():
        d[cusip][feature_name].append(feature_value)


def combine_features(features):
    'return dict containing all the features in the list features: List{Dict[feature_name, feature_value]}'
    # check for duplicate feature names
    # check for numeric feature_values
    result = {}
    for feature in features:
        print 'feature', feature
        for feature_name, feature_value in feature.iteritems():
            print feature_name, feature_value
            if feature_name in result:
                print 'error: duplicate feature name:', feature_name, 'with value', feature_value
                pdb.set_trace()
            if not isinstance(feature_value, numbers.Number):
                print 'error: feature value is not a number:', feature_name, feature_value
            result[feature_name] = feature_value
    return result


def do_work(control):
    'write order imbalance for each trade in the input file'
    def validate_trades(df):
        assert (df.ticker == control.arg.ticker.upper()).all()

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

    print 'creating feaures'
    count = 0
    cusips_not_in_security_master = set()
    maturity_dates_bad = set()

    def read_equity_ohlc(path):
        return models.read_csv(
            path,
            parse_dates=[0],
            verbose=True,
        )

    feature_maker_ohlc = FeatureMakerOhlc(
        df_ticker=read_equity_ohlc(control.doit.in_ticker_equity_ohlc),
        df_spx=read_equity_ohlc(control.doit.in_spx_equity_ohlc),
        verbose=True,
    )
    feature_maker_security_master = FeatureMakerSecurityMaster(
        df=models.read_csv(
            control.doit.in_security_master,
            parse_dates=['issue_date', 'maturity_date'],
            verbose=True,
        )
    )
    feature_maker_ticker = FeatureMakerTicker(
        order_imbalance4_hps={
            'lookback': 10,
            'typical_bid_offer': 2,
            'proximity_cutoff': 20,
        },
    )
    all_feature_makers = (
        feature_maker_ohlc,
        feature_maker_security_master,
        feature_maker_ticker,
    )
    d = {}  # Dict[cusip, Dict[feature_name, feature_values]], maintained by append_features()
    for index, trade in df_ticker.iterrows():
        # maybe mutate d based on index and trade
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
        features_made = []
        stopped_early = False
        for feature_maker in all_feature_makers:
            features = feature_maker.make_features(cusip, trade)
            if features is None:
                print 'no features for', index, cusip, feature_maker.input_file_name
                stopped_early = True
                continue
            else:
                features_made.append(features)
        if stopped_early:
            continue
        row_id = {
            'id_index': index,
            'id_cusip': cusip,
            'id_effectivedatetime': trade.effectivedatetime,
        }
        features_made.append(row_id)
        all_features = combine_features(features_made)
        append(d, cusip, all_features)

    print 'writing result files'
    for cusip in d.keys():
        # check that length of values is the same
        print 'feature names, # records (should be same for all)'
        for k, v in d[cusip].iteritems():
            print ' ', k, len(v)
        df = pd.DataFrame(
            data=d[cusip],
            index=d[cusip]['id_index'],
        )
        filename = '%s-%s.csv' % (control.arg.ticker, cusip)
        path = os.path.join(control.doit.out_dir, filename)
        df.to_csv(path)
        print 'wrote %d records to %s' % (len(df), filename)
    print 'reasons records skipped, # times this reason was used'
    for maker in (feature_maker_ticker,):
        for reason, count in maker.skipped_reasons.iteritems():
            print ' ', maker.input_file_name, reason, count
    print 'number of feature records created (assuming no interactions with other input files)'
    for maker in all_feature_makers:
        print ' ', maker.input_file_name, maker.count_created

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
