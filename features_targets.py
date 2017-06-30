'''create features and targets from trace prints and reference data

NOTE: This program looks at all trades for the CUSIP and related OTR CUSIPs from the
beginning of time. Thus trades 10 years ago are used to predict next oasspreads today.
That is way too much history. We should fix this when we have a streaming infrastructure.
Most likely, only the last 1000 or so trades are relevant.

INVOCATION
  python features_targets.py {cusip} {effective_date} {--test} {--trace} {--analyze_trace}
where
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 effective_date: YYYY-MM-DD is the date of the trade
 --analyze_trace means to print the frequency and date ranges of all cusips
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
  python features_targets.py 037833AG5 2017-06-26 --test --cache  # AAPL, no oasspreads for the relevant time period
  python features_targets.py 037833AG5 2013-09-09 --test --cache  # AAPL, no oasspreads for the relevant time period
  python features_targets.py 68389XAC9 2012-01-03 --test --cache  # ORCL
  python features_targets.py 68389XAC9 2017-06-26  # recent trades
  python features_targets.py 68389XAC9 2017-03-01  # 15 trades
  python features_targets.py 594918BW3 2003-01-31 --analyze_trace  # first trade in TRACE_production.csv
  python features_targets.py 68389XAC9 2012-01-03 --test # fails, date 2012-01-03 not in fundamentals
  python features_targets.py 68389XAC9 2012-01-29 --test # the first date in etf_weights_of_cusip_pcrt_agg.csv is this one

  python features_targets.py 68389XAC9 2016-12-15  # fails: key error 'out_features


See build.py for input and output files.

IDEA FOR FUTURE:
'''

from __future__ import division

import argparse
import collections
import datetime
import gc
import numpy as np
import os
import cPickle as pickle
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import applied_data_science.debug
import applied_data_science.dirutility
import applied_data_science.lower_priority
import applied_data_science.pickle_utilities

from applied_data_science.Bunch import Bunch
from applied_data_science.Date import Date
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.arg_type
import seven.GetSecurityMasterInfo
import seven.build
import seven.feature_makers
import seven.fit_predict_output
import seven.GetSecurityMasterInfo
import seven.read_csv
import seven.target_maker

pp = pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('cusip', type=seven.arg_type.cusip)
    parser.add_argument('effective_date', type=seven.arg_type.date)
    parser.add_argument('--analyze_trace', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')

    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = seven.build.features_targets(arg.cusip, arg.effective_date, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    # start building the history of trace prints from trace prints 100 calendar days prior to the effective date
    first_relevant_trace_print_date = Date(from_yyyy_mm_dd=arg.effective_date).value - datetime.timedelta(100)

    timer = Timer()

    return Bunch(
        arg=arg,
        first_relevant_trace_print_date=first_relevant_trace_print_date,
        get_security_master_info=seven.GetSecurityMasterInfo.GetSecurityMasterInfo(),
        path=paths,
        random_seed=random_seed,
        issuer=paths['issuer'],
        timer=timer,
    )


def make_elapsed(control):
    def elapsed():
        'return elapsed wallclock seconds'
        return control.timer.elapsed_wallclock_seconds()


def select_relevant_cusips(records, control):
    'return DataFrame containing just the records related to the query cusip'
    # these are the records for the CUSIP and for any on-the-run bond for the CUSIP
    # the on-the-run bonds can vary by trace print

    print 'STUB: select_relevant_cusips: just use the primary cusip'
    query_cusip = control.arg.cusip
    # the team is working on building a file that maps cusip -> effective_date -> otr-cusip
    # once they finish, this function will need to be modified
    if True:
        cusip_records = records.loc[records['cusip'] == query_cusip]
        return cusip_records
    if False:
        # explore otr_cusips for each cusip
        cusips = set(records['cusip'])
        for cusip in cusips:
            cusip_records = records.loc[cusip == records['cusip']]
            otr_cusips = set(cusip_records['cusip1'])
            print 'cusip', cusip, 'otr cusips', otr_cusips
    print 'columns in the trace print file', records.columns
    print 'all cusips in trace print file', set(records['cusip'])
    has_query_cusip = records['cusip'] == query_cusip
    cusip_records = records.loc[has_query_cusip]
    assert len(cusip_records > 0)
    otr_cusips = set(cusip_records['cusip1'])
    print 'all OTR cusips for query cusip %s: %s' % (query_cusip, otr_cusips)
    has_otr = pd.Series(
        data=False,
        index=records.index,
    )
    for otr_cusip in otr_cusips:
        has_otr |= records['cusip1'] == otr_cusip
    selected = has_otr | has_query_cusip
    result = records.loc[selected]
    assert len(result) >= len(cusip_records)
    assert len(result) <= len(records)
    return result


def sort_by_effectivedatetime(df):
    'return new DataFrame in sorted order'
    # Note: the following statement works but generates a SettingWithCopyWarning
    result = df.sort_values('effectivedatetime')
    return result


def analyze_trace_prints(trace_prints, control):
    'print to stdout'
    def print_trade(issuepriceid):
        print issuepriceid, trace_prints.loc[issuepriceid]

    print 'there are %d trace_prints' % len(trace_prints)
    cusips_set = set(trace_prints['cusip'])
    print 'there are %d cusips' % len(cusips_set)
    print 'cusip -> n_prints -> first_dateimte -> last_datetime'
    for cusip in sorted(cusips_set):
        trace_prints_cusip = trace_prints.loc[trace_prints['cusip'] == cusip]
        effectivedatetimes = trace_prints_cusip['effectivedatetime']
        print cusip, len(trace_prints_cusip), min(effectivedatetimes), max(effectivedatetimes)


def read_trace_prints_underlying_file(control):
    'return DataFrame'
    trace_prints = seven.read_csv.input(
        issuer=control.issuer,
        logical_name='trace',
        nrows=1000 if control.arg.test else None,
    )
    control.timer.lap('stop read_trace_prints_underlying_file')
    return trace_prints


def transform_trace_prints(trace_prints, control):
    'return (DataFrame of relevant and transformed trades, set of otr_cusips)'
    def select_relevant_dates(df):
        'return DataFrame with effective not at least control.first_relevant_trace_print_date'
        mask = df['effectivedatetime'] >= control.first_relevant_trace_print_date
        result = df.loc[mask]
        control.timer.lap('select_relevant_dates')
        return result

    def make_otr_cusips(df):
        'return Dict[trade_date, otr_cusip]'
        # return dict[effective_date, otr_cusip]
        # input file format to change to have 3 columns
        otr = seven.read_csv.input(
            issuer=control.issuer,
            logical_name='otr',
        )
        otr_cusip = otr.loc[otr['primary_cusip'] == control.arg.cusip]
        result = {}
        for timestamp, record in otr_cusip.iterrows():
            result[timestamp.date()] = record['otr_cusip']
        control.timer.lap('make_otr_cusips')
        return result

    def select_relevant_cusips(df, otr_cusips):
        'return DataFrame with just the primary and OTR cusips'
        # starting on the date column, the otr_cusip is as specified
        # before the first date, the otr_cusip is the primary_cusip
        mask = df['cusip'] == control.arg.cusip
        distinct_otr_cusips = set(otr_cusips.values())
        for otr_cusip in distinct_otr_cusips:
            mask_otr_cusip = df['cusip'] == otr_cusip
            if sum(mask_otr_cusip) == 0:
                print 'programming error: otr cusip %s is not in the trace print file' % otr_cusip
                pdb.set_trace()
            mask |= mask_otr_cusip
        result = df.loc[mask]
        control.timer.lap('select_relevant_cusips')
        return result

    def select_valid_oasspreads(df, verbose=True):
        'return df where the oasspread is not NaN'
        def print_cusip_count(df):
            for cusip in sorted(set(df['cusip'])):
                print cusip, sum(df['cusip'] == cusip)

        if verbose:
            print 'incoming cusip -> count'
            print_cusip_count(df)

        mask = np.isnan(df['oasspread'])
        print 'ignoring %d trace print that have NaN values' % sum(mask)
        result = df.loc[~mask]
        if verbose:
            print 'outgoing cusip -> count'
            print_cusip_count(result)
        control.timer.lap('select_valid_oasspread')
        return result

    def add_derived_fields(trace_prints):
        'mutate trace_prints'
        trace_prints['effectivedatetime'] = seven.feature_makers.make_effectivedatetime(trace_prints)
        trace_prints['effectiveyear'] = pd.DatetimeIndex(trace_prints['effectivedatetime']).year
        trace_prints['effectivemonth'] = pd.DatetimeIndex(trace_prints['effectivedatetime']).month
        trace_prints['effectiveday'] = pd.DatetimeIndex(trace_prints['effectivedatetime']).day
        trace_prints['issuepriceid'] = trace_prints.index
        control.timer.lap('add_derived_fields')

    def elapsed():
        return control.timer.elapsed_wallclock_seconds()

    start_wallclock = elapsed()
    add_derived_fields(trace_prints)
    if control.arg.analyze_trace:
        analyze_trace_prints(trace_prints, control)
    trace_prints_relevant_dates = select_relevant_dates(trace_prints)
    otr_cusips = make_otr_cusips(trace_prints_relevant_dates)
    distinct_otr_cusips = set(otr_cusips.values())
    print 'there are %d distinct otr cusips for cusip %s' % (len(distinct_otr_cusips), control.arg.cusip)
    trace_prints_for_cusips = select_relevant_cusips(trace_prints_relevant_dates, otr_cusips)
    ok_oasspreads = select_valid_oasspreads(trace_prints_for_cusips)
    sorted_trace_prints = ok_oasspreads.sort_values('effectivedatetime')
    print 'read %d trace_prints' % len(trace_prints)
    print 'of which, %d were on relevant dates' % len(trace_prints_relevant_dates)
    print 'of which, %d were for the specified CUSIP or its on-the-run CUSIPS' % len(trace_prints_for_cusips)
    print 'of which, %d had OK oasspread values' % len(ok_oasspreads)
    print '# cusips in reduced file:', len(otr_cusips)
    print 'transform_trace_prints took %f wallclock seconds' % (elapsed() - start_wallclock)
    return sorted_trace_prints, otr_cusips


def read_and_transform_trace_prints(control):
    'return (DataFrame, otr_cusips)'
    # use a cache to reduce wall clock time for the reading
    def same_invocation_parameters(arg1, arg2):
        return (
            arg1.cusip == arg2.cusip and
            arg1.effective_date == arg2.effective_date and
            arg1.test == arg2.test
        )

    # def modification_time():
    #     return os.path.getmtime(control.path['in_trace'])

    def write_cache(control_, trace_prints_, otr_cusips_):
        'write args to pickle file'
        obj = (control_, trace_prints_, otr_cusips_)
        with open(control.path['out_cache'], 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        control.timer.lap('write_cache')

    def read_cache():
        'return (control, trace_prints, otr_cusips) from pickle file'
        with open(control.path['out_cache'], 'rb') as f:
            control_, trace_prints_, otr_cusips_ = pickle.load(f)
        control.timer.lap('read_cache')
        return control_, trace_prints_, otr_cusips_

    def read_and_transform_underlying_file():
        'return (DataFrame, otr_cusip)'
        df1 = read_trace_prints_underlying_file(control)
        df2, otr_cusips = transform_trace_prints(df1, control)
        return df2, otr_cusips

    def elapsed():
        return control.timer.elapsed_wallclock_seconds()

    start_wallclock = elapsed()
    if control.arg.cache:
        if os.path.isfile(control.path['out_cache']):
            # cache exists
            # if it was created with the same invocation parameters, use it
            # otherwise, re-create it
            cached_control, cached_trace_prints, cached_otr_cusips = read_cache()
            if same_invocation_parameters(cached_control.arg, control.arg):
                # the cache file was built with same parameters as the current invocation
                # so we can use it
                trace_prints = cached_trace_prints
                otr_cusips = cached_otr_cusips
            else:
                trace_prints, otr_cusips = read_and_transform_underlying_file()
                write_cache(control, trace_prints, otr_cusips)
        else:
            # cache does not exist
            # create it and save the current invocation parameters
            trace_prints, otr_cusips = read_and_transform_underlying_file()
            write_cache(control, trace_prints, otr_cusips)
    else:
        trace_prints, otr_cusips = read_and_transform_underlying_file()
    print 'read_and_transform_trace_prints took %f wallclock seconds' % (elapsed() - start_wallclock)
    return trace_prints, otr_cusips


def do_work(control):
    'write predictions from fitted models to file system'
    def lap():
        'return ellapsed wall clock time:float since previous call to lap()'
        return control.timer.lap('lap', verbose=False)[1]

    def print_info(selected_date):
        mask_date_cusip = (trace_prints['effectivedate'] == selected_date) & (trace_prints['cusip'] == control.arg.cusip)
        n_predictions = sum(mask_date_cusip)
        print 'found %d trace prints on selected date %s with selected cusip %s' % (
            n_predictions,
            selected_date,
            control.arg.cusip,
        )
        print 'found %d distinct effective date times' % len(set(trace_prints['effectivedatetime']))
        print 'cusips in file:', set(trace_prints['cusip'])
        print 'prepared input in %s wall clock seconds' % lap()

    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    control.timer.lap('prelude to start do_work')
    trace_prints, otr_cusips = read_and_transform_trace_prints(control)
    all_distinct_cusips = set(trace_prints['cusip'])
    print 'found %d trace prints for the cusip %s and its related OTR cusips' % (len(trace_prints), control.arg.cusip)
    print 'will create features and targets for these cusips:', all_distinct_cusips

    # iterate over each relevant row
    # build and save the features for the cusip
    # the issuer may vary by cusip
    get_security_master_info = seven.GetSecurityMasterInfo.GetSecurityMasterInfo()
    features_accumulator = {}
    for cusip in all_distinct_cusips:
        features_accumulator[cusip] = seven.feature_makers.FeaturesAccumulator(
            issuer=get_security_master_info.issuer_for_cusip(cusip),
            cusip=cusip,
        )
    target_maker = seven.target_maker.TargetMaker()

    count = collections.Counter()
    exception_not_on_selected_date = collections.Counter()
    exception_on_selected_date = collections.Counter()

    def make_exception(msg):
        if on_selected_date:
            exception_on_selected_date[msg] += 1
        else:
            exception_not_on_selected_date[msg] += 1

    selected_date = Date(from_yyyy_mm_dd=control.arg.effective_date).value  # a datetime.date
    print_info(selected_date)

    last_trace_record_date = None
    count['feature and target records created'] = 0

    print 'cusip -> # trace prints'
    for cusip in sorted(set(trace_prints['cusip'])):
        print cusip, sum(trace_prints['cusip'] == cusip)
    print 'invocation cusip', control.arg.cusip

    debug = False

    for trace_index, trace_record in trace_prints.iterrows():
        # Note: the CUSIP for each record is control.arg.cusip or a related OTR cusip
        count['n trace records seen'] += 1
        if count['n trace records seen'] % 100 == 0:
            print 'processing trace record %d of %d' % (count['n trace records seen'], len(trace_prints))

        # assure records are in increasing order by effective datetime
        # NOTE: We have sorted the files, so this check is redundant
        # It's a guard against a maintenance error
        trace_record_date = trace_record['effectivedate'].to_pydatetime().date()  # a datetime.date
        if last_trace_record_date is not None:
            # if last_trace_record_date != trace_record_date:
            #     print 'found trace print record with date', trace_record_date
            assert last_trace_record_date <= trace_record_date
        last_trace_record_date = trace_record_date

        # stop once we go past the selected date
        if trace_record_date > selected_date:
            break  # stop, since the trace_prints are in non-decreasing order by effectivedatetime
        count['n_trace_records processed']

        on_selected_date = trace_record_date == selected_date

        errs = features_accumulator[trace_record['cusip']].append_features(trace_index, trace_record)
        if errs is not None:  # errs is possible a list of errors
            count['had feature errors'] += 1
            for err in errs:
                make_exception('feature_maker err: %s' % err)
            continue

        err = target_maker.append_targets(trace_index, trace_record)
        if err is not None:
            make_exception('target_maker err: %s' % err)
            count['had target errors'] += 1
            continue

        count['feature and target records created'] += 1
        count['feature and target records created on date %s' % trace_record_date] += 1
        count['feature and target records created for cusip %s' % trace_record['cusip']] += 1
        if on_selected_date and trace_record['cusip'] == control.arg.cusip:
            count['features and targets created for query'] += 1

        if debug:
            if count['feature and targets created for query'] > 10:
                print 'DEBUG CODE: discard output'
                break

        # try to keep memory usage roughly constant
        # that enables running parallel instances on a single system
        gc.collect()

    control.timer.lap('create features from all relevant trace print records')
    print 'summary across all trace print records examined'
    print 'counts'
    for k in sorted(count.keys()):
        print '%71s: %6d' % (k, count[k])

    def print_exceptions(count, title):
        print
        print title
        for k in sorted(count.keys()):
            print '%71s: %6d' % (k, count[k])

    print_exceptions(exception_not_on_selected_date, 'exceptions not on the selected date')
    print_exceptions(exception_on_selected_date, 'exceptions on the selected date')

    print 'merge all the otr_cusip features into the primary cusip features'
    print 'only for the query date!'
    print
    print 'cusip -> # features'
    for k, v in features_accumulator.iteritems():
        print k, len(v.features)

    if count['features and targets created for query'] == 0:
        print 'create no features for the primary custip %s' % control.arg.cusip
        print 'stopping without creating output files'
        sys.exit(1)  # 1 ==> some type of abnormal exit
    pdb.set_trace()
    mask = features_accumulator[control.arg.cusip].features['id_effectivedate'] == control.arg.effective_date
    primary_cusip_features = features_accumulator[control.arg.cusip].features.loc[mask]
    print 'primary_cusip_features on the selected date', len(primary_cusip_features)
    if len(primary_cusip_features) == 0:
        print 'no features for the primary cusip'
        sys.exit(1)
    # select just the rows on the query date
    pdb.set_trace()
    merged_dataframe = pd.DataFrame()
    for index, primary_features in primary_cusip_features.iterrows():
        otr_cusip = otr_cusips[primary_features['id_effectivedate'].date()]
        otr_cusip_features = features_accumulator[otr_cusip].features  # a DataFrame
        # find the earlist prior otr cusip trade
        time_mask = otr_cusip_features['id_effectivedatetime'] < primary_features['id_effectivedatetime']
        before = otr_cusip_features.loc[time_mask]
        if len(before) == 0:
            print 'no otr cusip trace prints before the query trace print'
        else:
            just_before = before.sort_values(by='id_effectivedatetime').iloc[-1]  # a series
            pdb.set_trace()
            features = {}
            for k, v in primary_features.iteritems():
                features[k] = v
            pdb.set_trace()
            for k, v in just_before.iteritems():
                if k.startswith('p_'):
                    new_feature_name = 'otr1_' + k[2:]
                    features[new_feature_name] = v
                elif k == 'id_cusip':
                    features['id_otr1_cusip'] = v
            pdb.set_trace()
            new_row = pd.DataFrame(data=features, index=[index])
            merged_dataframe = merged_dataframe.append(new_row)

    # write the merged data frame
    pdb.set_trace()
    merged_dataframe.to_csv(control.path['out_features'])

    # write the targets
    pdb.set_trace()
    all_targets = target_maker.get_dataframe()
    targets = all_targets.loc[merged_dataframe.index]
    targets.to_csv(control.path['out_targets'])

    print 'wrote %d features' % len(merged_dataframe)
    pritn 'wrote %d targets' % len(targets)

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
    print control.arg
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
        datetime

    main(sys.argv)
