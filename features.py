'''create feature sets for a CUSIP

INVOCATION
  python features.py {ticker} [--cusips CUSIP ...] [--test] [--trace]

where
 {ticker} is a CSV file in MidPredictors/data
 --cusips CUSIP1 CUSIP2 ... means to create the targets only for the specified CUSIPs
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION THAT SHOULD SUCCEED
 python features.py orcl
 python features.py orcl --cusips 68389XAM7 --test

EXMPLES OF INVOCATIONS THAT SHOULD FAIL
 python features.py orcl --cusips 68389XAS4 --test   (NOTE: does not create output)
 python features.py orcl --cusips 68389XAC9 68389XAH8 68389XAE5 68402LAC8 --test
   (generates errors, because 68402LAC8 is not in the security master)
 python features.py orcl --cusips 68389XAC9 68389XAH8 68389XAE5 --test
   (generate KeyError: '68402LAC8')

INPUTS
 see the features created description below and the file build.py

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
import seven.feature_makers as feature_makers
import seven.read_csv as read_csv

from seven.feature_makers import FeatureMakerEtf
from seven.feature_makers import FeatureMakerFund
from seven.feature_makers import FeatureMakerOhlc
from seven.feature_makers import FeatureMakerSecurityMaster
from seven.feature_makers import FeatureMakerTrace
from seven.feature_makers import FeatureMakerTradeId

import build
pp = pprint


def make_control(argv):
    'return a Bunch'

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker')
    parser.add_argument('--cusips', action='store', nargs='*', type=arg_type.cusip)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = build.features(arg.ticker, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=Timer(),
    )


def pass1(df_trace, all_feature_makers, control, cusip_counter, create_features):
    'return Dict[cusip, Dataframe] containing just the features for the cusip itself'
    # pass2() adds in the features for the first related on-the-run cusip
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

    class AccumulateErrors(object):
        def __init__(self):
            self.skipped_by_selected = collections.Counter()
            self.skipped_by_feature_maker_name = collections.defaultdict(collections.Counter)
            self.skipped_by_cusip = collections.defaultdict(collections.Counter)

        def report_skipped_master_record(self, index, master_record, maybe_feature_maker, msg):
            if maybe_feature_maker is None:
                self.skipped_by_selected[msg] += 1
            else:
                self.skipped_by_feature_maker_name[maybe_feature_maker.name][msg] += 1
                cusip = master_record['cusip']
                key = (maybe_feature_maker.name, msg)
                self.skipped_by_cusip[cusip][key] += 1

        def print_reports(self, cusip_counterOLD):
            print 'records skipped by the master record selector'
            for msg, count in self.skipped_by_selected.iteritems():
                print msg, count

            print 'records skipped by a feature maker'
            for feature_maker_name, counter in sorted(self.skipped_by_feature_maker_name.iteritems()):
                print 'skipped by feature maker', feature_maker_name
                total = 0
                for msg, count in sorted(counter.iteritems()):
                    print ' %50s %d' % (msg, count)
                    total += count
                print ' total skipped', total

            n = 5
            print 'first % records skipped by a feature more for a cusip' % n
            for cusip, counter in sorted(self.skipped_by_cusip.iteritems()):
                print 'skipped for cusip', cusip
                total = 0
                for key, count in counter.items()[:n]:
                    name, msg = key
                    print ' %20s %50s %d' % (name, msg, count)
                if len(counter) > n:
                    print ' ', '... %d more not printed' % (len(counter) - n)

            print 'count of records skipped for cusip'
            for cusip, counter in self.skipped_by_cusip.iteritems():
                print 'for cusip %s, skipped %d records of %d' % (
                    cusip,
                    sum(counter.values()),
                    cusip_counter[cusip],
                )

    d = {}  # Dict[cusip, Dict[feature_name, feature_values:List[Number]]], maintained by append_features()
    accumulate_errors = AccumulateErrors()
    for features, ticker_index, ticker_record in create_features.create(
        feature_makers=all_feature_makers,
        master_file_records=master_file_records(df_trace),
        selected=selected,
        report_skipped_master_record=accumulate_errors.report_skipped_master_record,
        verbose=True,
    ):
        cusip = ticker_record.cusip
        cusip_counter[cusip] += 1
        if cusip not in d:
            d[cusip] = collections.defaultdict(list)
        for feature_name, feature_value in features.iteritems():
            d[cusip][feature_name].append(feature_value)

    print 'errors found in pass 1'
    accumulate_errors.print_reports(cusip_counter)
    print 'pass 1 created features for %d cusips' % len(d)

    # create the pass 1 data frames
    pass1_dataframes = {}
    for cusip in d.keys():
        df = pd.DataFrame(
            data=d[cusip],
            index=d[cusip]['id_ticker_index'],
        )
        pass1_dataframes[cusip] = df

    return pass1_dataframes


def pass2(dataframes):
    'return Dict[cusip, Dataframe] by joining cusip1 features to each cusip'
    def make_new_feature_values(cusip_df, feature_name, cusip):
        cusip1s = set(cusip_df[feature_column_cusip1])
        print 'pass2', cusip, feature_name, cusip1s
        new_series = pd.Series(
            data=0.0,  # NOTE: every feature is numeric
            index=cusip_df.index,
        )
        sum_masks = 0
        for cusip1 in cusip1s:
            # the cusip1 values are mutually exclusive and collective exhaustive for the index values
            mask = cusip_df[feature_column_cusip1] == cusip1
            new_series.loc[mask] += cusip_df[feature_name].loc[mask]  # each index value will be changed once
            sum_masks += sum(mask)
        if sum_masks != len(cusip_df):
            print 'error: did not rebuild every index value', sum_masks, len(cusip_df)
            pdb.set_trace()
        return new_series

    def make_new_feature_name(old_feature_name):
        return prefix_new_feature_name + old_feature_name[len(prefix_old_feature_name):]

    result = {}
    prefix_id = 'id_'
    prefix_old_feature_name = 'p_'
    prefix_new_feature_name = 'otr1_'
    feature_column_cusip = 'id_cusip'
    feature_column_cusip1 = 'id_cusip1'
    for cusip, cusip_df in dataframes.iteritems():
        assert (cusip_df[feature_column_cusip] == cusip).all()
        new_df = pd.DataFrame(index=cusip_df.index)
        for feature_name in cusip_df.columns:
            if feature_name.startswith(prefix_id):
                new_df[feature_name] = cusip_df[feature_name]
            else:
                assert feature_name.startswith(prefix_old_feature_name)
                new_df[feature_name] = cusip_df[feature_name]  # copy the existing cusip feature
                new_df[make_new_feature_name(feature_name)] = make_new_feature_values(
                    cusip_df,
                    feature_name,
                    cusip,
                )
        result[cusip] = new_df
    return result


def do_work(control):
    'write order imbalance for each ticker_record in the input file'
    def validate_trace_records(df):
        'return Set(all_cusips), Set(all_isins) in the trace records; stop if an error is found'
        def expected_ticker(df):
            assert (df.ticker == control.arg.ticker.upper()).all()

        def unique_index_values(df):
            unique_ids = set()
            for index, row in df.iterrows():
                if index in unique_ids:
                    print 'error: found non-unique id/index', index
                    pdb.set_trace()
                unique_ids.add(index)

        def all_cusip1_values_present(df):
            'return set(all_cusips), set(all_isin)'
            all_cusips = set()
            all_isins = set()
            cusip_counter = collections.Counter()
            otr_cusips = collections.defaultdict(collections.Counter)
            for index, record in df.iterrows():
                cusip = record['cusip']
                cusip1 = record['cusip1']
                cusip_counter[cusip] += 1
                all_cusips.add(cusip)
                all_cusips.add(cusip1)
                all_isins.add(record['isin'])
                otr_cusips[cusip][cusip1] += 1
            for cusip in sorted(otr_cusips.keys()):
                print cusip, otr_cusips[cusip]
            n_missing = 0
            for cusip, ort_cusips in otr_cusips.iteritems():
                for otr_cusip, n_occurrences in ort_cusips.iteritems():
                    if otr_cusip not in otr_cusips:
                        print 'trace file missing cusip %s, which is %d times the cusup1 for cusip %s' % (
                            otr_cusip,
                            n_occurrences,
                            cusip
                        )
                        n_missing += 1
            if n_missing > 0:
                print 'found %d missing cusip1 values' % n_missing
                pdb.set_trace()
            else:
                print 'no missing cusip1 values'
            return all_cusips, all_isins, cusip_counter

        expected_ticker(df)
        unique_index_values(df)
        return all_cusip1_values_present(df)

    def read_equity_ohlc(path):
        return read_csv(
            path,
            parse_dates=[0],
        )

    # BODY STARTS HERE

    # read trace file
    df_trace = read_csv.input(
        control.arg.ticker,
        'trace',
        nrows=1000 if control.arg.test else None,
    )
    all_cusips, all_isins, cusip_counter = validate_trace_records(df_trace)
    df_trace['effectivedatetime'] = feature_makers.make_effectivedatetime(df_trace)

    all_feature_makers = (
        FeatureMakerEtf(
            df=read_csv.input(control.arg.ticker, 'etf agg'),
            name='agg',
            all_isins=all_isins,
        ),
        FeatureMakerEtf(
            df=read_csv.input(control.arg.ticker, 'etf lqd'),
            name='lqd',
            all_isins=all_isins,
        ),
        FeatureMakerFund(
            df=read_csv.input(control.arg.ticker, 'fund'),
        ),
        FeatureMakerTradeId(),
        FeatureMakerOhlc(
            df_ticker=read_csv.input(control.arg.ticker, 'ohlc ticker'),
            df_spx=read_csv.input(control.arg.ticker, 'ohlc spx'),
        ),
        FeatureMakerSecurityMaster(
            df=read_csv.input(control.arg.ticker, 'security master'),
        ),
        FeatureMakerTrace(
            order_imbalance4_hps={
                'lookback': 10,
                'typical_bid_offer': 2,
                'proximity_cutoff': 20,
            },
        ),
    )

    # pass 1: create features for the trades, create d:Dict
    create_features = timeseries.CreateFeatures()
    pass1_dataframes = pass1(df_trace, all_feature_makers, control, cusip_counter, create_features)

    # pass 2: create new set of dataframews that incorporate cusip1 info, where it exists
    # keep only indices in both the cusip and its related cusip1 (possibly shortening the training data)
    pass2_dataframes = pass2(pass1_dataframes)
    print '# pass2 dataframes', len(pass2_dataframes)

    # report and write output file
    print 'read %d ticker records across %d cusips' % (create_features.n_input_records, len(pass2_dataframes))
    print 'created %d feature records across all cusips' % create_features.n_output_records
    print 'writing output files'
    for cusip, df in pass2_dataframes.iteritems():
        filename = cusip + '.csv'
        path = os.path.join(control.path['dir_out'], filename)
        df_sorted = df.sort_index(axis=1)
        df_sorted.to_csv(path)
        print 'wrote %d records to %s' % (len(df_sorted), filename)
    return None


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
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
