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
 WORKING/features-{ticker}/{cusip}.csv
where
 {cusip} is in the {ticker} file

FEATURES CREATED BY INPUT FILE AND NEXT STEPS (where needed)
  identifiers (not features) all from the {ticker}.csv file
    id_index              index from the {ticker}.csv file; this uniquely identifies the ticker_record
    id_cusip              cusip
    id_effectivedatetime  date & time of the ticker_record

  comparables_{ticker}.csv    TODO GC to provide file with time stamps and related CUSIPs
                              then RL to create features
    is_on_the_run                    0 or 1
    oftr_{oftr_cardinal}-{feature}   missing if bond is on the run; redefine feature sets when missing
    <other features to be designed by GC>

  {ticker}.csv: ticker_record info by effectivedatetime for many CUSIPs in the file
    oasspread
    order_imbalance4
    oasspread_ ticker_record_type}
    quantity_ ticker_record_type}
    quantity
 ticker_record_type_is_ ticker_record_type}

  {ticker}_equity_ohlc.csv and spx_equity_ohlc.csv
    price_delta_ratio_back_days_{days}    (ticker price delta / market price delta) for {back} market days
    price_delta_ratio_back_hours_{hours}  same ratio, back one hour; TODO: create after GC provides data

  {ticker}_fund.csv
    debt_to_market_cap
    debt_to_ltm_ebitda
    gross_leverage
    interest_coverage
    (interest_expense_size)           TODO: add once appears in file
    ltm_ebitda_size                   NOTE: Googe doc asks for last 12 months EBITDA
    (operating_income_size)           TODO: add once appears in file

  {ticker}_sec_master.csv   GC to update file
    amount_issued
    collateral_type_is_sr_unsecured
    coupon_is_fixed
    coupon_is_floating
    coupon_current
    fraction_still_outstanding  RL to add once file provided by GC; should be in separate file
    is_callable
    is_puttable              TODO add once file is updated by GC (note is in {ticker}_comparables_sec_master)
    months_to_maturity_size

for
 days in {1, 2, ..., 30}
 {feature}                         feature of a corresponding on-the-run bond
 hours in {1, 2, ..., 8}
 oftr_cardinal in {1, 2, 3}        if bond is off the run, corresponding CUSIPs ordinal rank
                                   where rank == 1 is the most similar on-the-run CUSIP
 ticker_record_type in {B, D, S}
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import applied_data_science
import applied_data_science.dirutility
import applied_data_science.timeseries as timeseries

from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.arg_type as arg_type
import seven.models as models
import seven
from seven.feature_makers import FeatureMakerFund
from seven.feature_makers import FeatureMakerNodupeTraceTickerOtr
from seven.feature_makers import FeatureMakerOhlc
from seven.feature_makers import FeatureMakerSecurityMaster
from seven.feature_makers import FeatureMakerTicker
from seven.feature_makers import FeatureMakerTradeId
import seven.path


class Doit(object):
    def __init__(self, ticker, test=False, me='features'):
        def make_outfile_name(cusip):
            return '%s.csv' % cusip

        self.make_outfile_name = make_outfile_name

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
        self.in_fund = os.path.join(midpredictor, ticker + '_fund.csv')
        self.in_security_master = os.path.join(midpredictor, ticker + '_sec_master.csv')
        self.in_spx_equity_ohlc = os.path.join(midpredictor, 'spx_equity_ohlc.csv')
        self.in_ticker = os.path.join(midpredictor, ticker + '.csv')
        self.in_ticker_equity_ohlc = os.path.join(midpredictor, ticker + '_equity_ohlc.csv')
        self.out_cusips = [
            os.path.join(out_dir, self.make_outfile_name(cusip))
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
            self.in_fund,
            self.in_security_master,
            self.in_spx_equity_ohlc,
            self.in_ticker,
            self.in_ticker_equity_ohlc,
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


class NodupeTraceTickerOtr(object):
    def __init__(self, df):
        pdb.set_trace()
        super(FeatureMakerNodupeTraceTickerOtr, self).__init__('nodup-trace-ticker-otr')
        self.otr_cusip = {}
        for index, record in df.iterrows():
            original_index = record.originalsequencenumber
            otr_cusip = record.bechmarkcusip
            self.otr[original_index] = otr_cusip

    def otr(self, ticker_index):
        pdb.set_trace()
        return self.otr[ticker_index]


def do_work(control):
    'write order imbalance for each ticker_record in the input file'
    def validate_ticker_records(df):
        assert (df.ticker == control.arg.ticker.upper()).all()

    # BODY STARTS HERE
    otr = Otr(
        df=pd.read_csv(
            control.doit.in_nodup_trace_ticker_otr,
            low_memory=False,
        )
    )
    # read {ticker}.csv
    df_ticker = models.read_csv(
        control.doit.in_ticker,
        parse_dates=['maturity', 'effectivedate', 'effectivetime'],
        nrows=1000 if control.arg.test else None,
        verbose=True,
    )
    validate_ticker_records(df_ticker)
    df_ticker['effectivedatetime'] = models.make_effectivedatetime(df_ticker)

    print 'creating feaures'
    count = 0

    def read_equity_ohlc(path):
        return models.read_csv(
            path,
            parse_dates=[0],
            verbose=True,
        )

    feature_maker_fund = FeatureMakerFund(
        df_fund=pd.read_csv(
            control.doit.in_fund,
            low_memory=False,
        ),
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
    feature_maker_id = FeatureMakerTradeId()
    all_feature_makers = (
        feature_maker_fund,
        feature_maker_id,
        feature_maker_ohlc,
        feature_maker_security_master,
        feature_maker_ticker,
    )

    def master_file_records(df_ticker):
        for ticker_index, ticker_record in df_ticker.iterrows():
            yield ticker_index, ticker_record

    def selected(ticker_index, ticker_record):
        if control.arg.cusip is not None:
            if ticker_record.cusip == control.arg.cusip:
                return (True, None)
            else:
                return (False, 'cusip was %s not the looked for %s' % (ticker_record.cusip, control.arg.cusip))
        else:
            return (True, None)

    d = {}  # Dict[cusip, Dict[feature_name, feature_values]], maintained by append_features()
    create_features = timeseries.CreateFeatures()
    for features, ticker_index, ticker_record in create_features.create(
        feature_makers=all_feature_makers,
        master_file_records=master_file_records(df_ticker),
        selected=selected,
        verbose=False,
    ):
        cusip = ticker_record.cusip
        if cusip not in d:
            d[cusip] = collections.defaultdict(list)
        for feature_name, feature_value in features.iteritems():
            d[cusip][feature_name].append(feature_value)
        otr[ticker_index] = get_otr_cusip(ticker_index)
    print 'read %d ticker records across %d cusips' % (create_features.n_input_records, len(d))
    print 'wrote %d feature records' % create_features.n_output_records
    print 'writing output files'
    for cusip in d.keys():
        # check that length of values is the same
        print 'feature names, # records (should be same for all)'
        common_length = None
        for feature_name in sorted(d[cusip].keys()):
            v = d[cusip][feature_name]
            print ' %45s %6d' % (feature_name, len(v))
            if common_length is None:
                common_length = len(v)
            else:
                assert len(v) == common_length
        df = pd.DataFrame(
            data=d[cusip],
            index=d[cusip]['id_ticker_index'],
        )
        filename = control.doit.make_outfile_name(cusip)
        path = os.path.join(control.doit.out_dir, filename)
        df.to_csv(path)
        print 'wrote %d records to %s' % (len(df), filename)
    print 'reasons records were skipped'
    for msg, count in create_features.skipped.iteritems():
        print '%50s: %d' % (msg, count)
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
