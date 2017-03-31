'''create feature sets for a CUSIP

INVOCATION
  python features.py {ticker}.csv [--cusips CUSIP ...] [--test] [--trace]

where
 {ticker}.csv is a CSV file in MidPredictors/data
 --cusips CUSIP1 CUSIP2 ... means to create the targets only for the specified CUSIPs
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python features.py orcl
 python features.py orcl --cusips 68389XAS4 --test   (NOTE: does not create output)
 python features.py orcl --cusips 68389XAC9 --test   (creates 137 feature output records )

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
    id_cusip_otr          related on-the-run cusip
    id_effectivedatetime  date & time of the ticker_record

  comparables_{ticker}.csv    TODO GC to provide file with time stamps and related CUSIPs
                              then RL to create features
    is_on_the_run                    0 or 1
    oftr_{oftr_cardinal}-{feature}   missing if bond is on the run; redefine feature sets when missing
    <other features to be designed by GC>

  nodupe_trace_{ticker}_otr.csv: ticker_record info with unique key issuepriceid  for all CUSIPs in the file
    p_oasspread_size
    p_order_imbalance4
    p_oasspread_{bds}_size
    p_quantity_{bds}_size
    p_quantity_size
    p_trade_type_is_{bds}
    where {bds} is one of {'B', 'D', 'S'}
    NOTE: this file contains cusip1, which is the cusip for the on-the-run bond that most trades like
          the bond identified in the field cusip

  {ticker}_etf_{index_name}.csv
    etf_weight_{index_name}_size  number in [0, 1]
    where {index_name} is one of {'agg', 'lqd'}

  {ticker}_equity_ohlc.csv and spx_equity_ohlc.csv
    price_delta_ratio_back_days_{days}    (ticker price delta / market price delta) for {back} market days
    price_delta_ratio_back_hours_{hours}  same ratio, back one hour; TODO: create after GC provides data

  {ticker}_fund.csv
    debt_to_market_cap
    debt_to_ltm_ebitda
    gross_leverage
    interest_coverage
    (interest_expense_size)           TODO: add once appears in file
    ltm_ebitda_size
    (operating_income_size)           TODO: add once appears in file

  {ticker}_and_comps_sec_master.csv   GC to update file
    amount_issued_size
    collateral_type_is_sr_unsecured   TODO: add, once this feature is in the file
    coupon_current_size
    coupon_is_fixed_rate
    coupon_is_floating_fate
    fraction_still_outstanding       TODO: add once file provided by GC; should be in separate file
    is_callable
    is_puttable
    months_to_maturity_size

 nodupe_trace_{ticker}_ltr.csv
    recreate all the features above (excluding the identifiers), prefixed by '2_'

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
import copy
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
import seven
import seven.feature_makers as feature_makers
from seven.feature_makers import FeatureMakerEtf
from seven.feature_makers import FeatureMakerFund
from seven.feature_makers import FeatureMakerOhlc
# from seven.feature_makers import FeatureMakerOtr
from seven.feature_makers import FeatureMakerSecurityMaster
from seven.feature_makers import FeatureMakerTrace
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
        self.in_etf_agg = os.path.join(midpredictor, 'etf', ticker + '_etf_agg.csv')
        self.in_etf_lqd = os.path.join(midpredictor, 'etf', ticker + '_etf_lqd.csv')
        self.in_fund = os.path.join(midpredictor, 'fundamentals', ticker + '_fund.csv')
        self.in_security_master = os.path.join(midpredictor, 'secmaster', ticker + '_and_comps_sec_master.csv')
        self.in_spx_equity_ohlc = os.path.join(midpredictor, 'tmp-todelete', 'spx_equity_ohlc.csv')
        self.in_ticker_equity_ohlc = os.path.join(midpredictor, 'tmp-todelete', ticker + '_equity_ohlc.csv')
        self.in_trace = os.path.join(midpredictor, 'trace', 'nodupe_trace_%s_otr.csv' % ticker)
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
            self.in_etf_agg,
            self.in_etf_lqd,
            self.in_fund,
            self.in_security_master,
            self.in_spx_equity_ohlc,
            self.in_ticker_equity_ohlc,
            self.in_trace,
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
    parser.add_argument('--cusips', action='store', nargs='*', type=arg_type.cusip)
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


def read_csv(path, date_columns=None, usecols=None, index_col=0, nrows=None, parse_dates=None, verbose=True):
    if index_col is not None and usecols is not None:
        print 'cannot read both the index column and specific columns'
        print 'possibly a bug in scikit-learn'
        pdb.set_trace()
    df = pd.read_csv(
        path,
        index_col=index_col,
        nrows=nrows,
        usecols=usecols,
        low_memory=False,
        parse_dates=parse_dates,
    )
    if verbose:
        print 'read %d rows from file %s' % (len(df), path)
        print df.columns
    return df


def do_work(control):
    'write order imbalance for each ticker_record in the input file'
    def validate_trace_records(df):
        assert (df.ticker == control.arg.ticker.upper()).all()
        unique_ids = set()
        for index, row in df.iterrows():
            if index in unique_ids:
                print 'error: found non-unique id/index', index
                pdb.set_trace()
            unique_ids.add(index)

    def read_equity_ohlc(path):
        return read_csv(
            path,
            parse_dates=[0],
        )

    # BODY STARTS HERE

    # read {ticker}.csv
    trade_print_identifier_column_name = 'issuepriceid'
    df_trace = read_csv(
        control.doit.in_trace,
        index_col=trade_print_identifier_column_name,
        nrows=1000 if control.arg.test else None,
        parse_dates=['maturity', 'effectivedate', 'effectivetime', ],
    )
    validate_trace_records(df_trace)
    df_trace['effectivedatetime'] = feature_makers.make_effectivedatetime(df_trace)

    feature_maker_etf_agg = FeatureMakerEtf(
        df=read_csv(
            control.doit.in_etf_agg,
            parse_dates=[0],
        ),
        name='agg',
    )
    feature_maker_etf_lqd = FeatureMakerEtf(
        df=read_csv(
            control.doit.in_etf_lqd,
            parse_dates=[0],
        ),
        name='lqd',
    )
    feature_maker_fund = FeatureMakerFund(
        df_fund=read_csv(
            control.doit.in_fund,
        ),
    )
    feature_maker_id = FeatureMakerTradeId()
    feature_maker_ohlc = FeatureMakerOhlc(
        df_ticker=read_equity_ohlc(control.doit.in_ticker_equity_ohlc),
        df_spx=read_equity_ohlc(control.doit.in_spx_equity_ohlc),
        verbose=True,
    )
    feature_maker_security_master = FeatureMakerSecurityMaster(
        df=read_csv(
            control.doit.in_security_master,
            parse_dates=['issue_date', 'maturity_date'],
            verbose=True,
        )
    )
    feature_maker_trace = FeatureMakerTrace(
        order_imbalance4_hps={
            'lookback': 10,
            'typical_bid_offer': 2,
            'proximity_cutoff': 20,
        },
    )
    all_feature_makers = (
        feature_maker_etf_agg,
        feature_maker_etf_lqd,
        feature_maker_fund,
        feature_maker_id,
        feature_maker_ohlc,
        feature_maker_security_master,
        feature_maker_trace,
    )

    def master_file_records(df):
        for ticker_index, ticker_record in df.iterrows():
            yield ticker_index, ticker_record

    def selected(index, record):
        'select records designated by control.arg.cusips, by returning (use_record, error_message)'
        if control.arg.cusips is not None:
            if record.cusip in control.arg.cusips:
                return (True, None)
            else:
                return (False, 'cusip was %s, not in the looked for %s' % (record.cusip, control.arg.cusips))
        else:
            return (True, None)

    # pass 1: create features for the trades, create d:Dict
    d = {}  # Dict[cusip, Dict[feature_name, feature_values:List[Number]]], maintained by append_features()
    create_features = timeseries.CreateFeatures()
    for features, ticker_index, ticker_record in create_features.create(
        feature_makers=all_feature_makers,
        master_file_records=master_file_records(df_trace),
        selected=selected,
        verbose=True,
    ):
        cusip = ticker_record.cusip
        if cusip not in d:
            d[cusip] = collections.defaultdict(list)
        for feature_name, feature_value in features.iteritems():
            if feature_name == 'issuepriceid':
                print 'found', feature_name
                pdb.set_trace()
            d[cusip][feature_name].append(feature_value)
        # otr[ticker_index] = get_otr_cusip(ticker_index)

    # pass 2: append features from the on-the-run bonds, create dd:Dict
    def is_otr_feature_name(feature_name):
        return not feature_name.startswith('id_')

    def otr_feature_name(off_the_run_feature_name):
        return 'otr1_' + off_the_run_feature_name

    def extend_feature_columns(existing_feature_name_values, new_feature_name_values, feature_name_prefix):
        'return new features dict incorporating both existing and new names and values'
        result = {}
        for existing_name, existing_values in existing_feature_name_values.iteritems():
            if existing_name.startswith('p_'):
                result[existing_name] = existing_values
                result[feature_name_prefix + existing_name[2:]] = new_feature_name_values[feature_name]
            elif existing_name.startswith('id_'):
                result[existing_name] = existing_values
            else:
                print 'error: unexpected feature_name', existing_name
                pdb.set_trace()
        return result

    # NOTE: it's possible that the on-the-run and off-the-run CUSIPs are identical, in which case,
    # we have duplicate feature values. That isn't a problem for the decision tree-based methods,
    # which will randomly choose one of the duplicate columns
    dd = copy.deepcopy(d)
    # add new columns to each cusip's dictionary
    for cusip in d.keys():
        cusip1 = feature_maker_trace.otr1[cusip]
        dd[cusip] = extend_feature_columns(dd[cusip], dd[cusip1], 'otr1_')

    # report and write output file
    print 'read %d ticker records across %d cusips' % (create_features.n_input_records, len(dd))
    print 'created %d feature records across all cusips' % create_features.n_output_records
    print 'writing output files'
    for cusip in dd.keys():
        # check that length of values is the same
        print 'feature names, # records (should be same for all)'
        common_length = None
        for feature_name in sorted(dd[cusip].keys()):
            v = dd[cusip][feature_name]
            print ' %45s %6d' % (feature_name, len(v))
            if common_length is None:
                common_length = len(v)
            else:
                assert len(v) == common_length
        df = pd.DataFrame(
            data=dd[cusip],
            index=dd[cusip]['id_ticker_index'],
        )
        filename = control.doit.make_outfile_name(cusip)
        path = os.path.join(control.doit.out_dir, filename)
        df_sorted = df.sort_index(axis=1)
        df_sorted.to_csv(path)
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
    # print control
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflake warnings for debugging tools
        pdb.set_trace()
        pprint()

    main(sys.argv)
