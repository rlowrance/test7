'''create features and targets from trace prints and reference data

NOTE: This program looks at all trades for the CUSIP and related OTR CUSIPs from the
beginning of time. Thus trades 10 years ago are used to predict next oasspreads today.
That is way too much history. We should fix this when we have a streaming infrastructure.
Most likely, only the last 1000 or so trades are relevant.

INVOCATION
  python features_targets.py {cusip} {effective_date} {--test} {--trace}
where
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 effective_date: YYYY-MM-DD is the date of the trade
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
  python features_targets.py ORCL 68389XAC9 2017-01-03
  python features-targets.py ORCL 68389XAC9 2012-01-06 --test

See build.py for input and output files.

IDEA FOR FUTURE:
'''

from __future__ import division

import argparse
import collections
import datetime
import gc
import numpy as np
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
import seven.build
import seven.feature_makers
import seven.fit_predict_output
import seven.read_csv
import seven.target_maker

pp = pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('issuer', type=seven.arg_type.issuer)
    parser.add_argument('cusip', type=seven.arg_type.cusip)
    parser.add_argument('effective_date', type=seven.arg_type.date)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')

    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = seven.build.features_targets(arg.issuer, arg.cusip, arg.effective_date, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    # start building the history of trace prints from trace prints 100 calendar days prior to the effective date
    first_relevant_trace_print_date = Date(from_yyyy_mm_dd=arg.effective_date).value - datetime.timedelta(100)

    return Bunch(
        arg=arg,
        first_relevant_trace_print_date=first_relevant_trace_print_date,
        path=paths,
        random_seed=random_seed,
        timer=Timer(),
    )


def select_relevant_cusips(records, query_cusip):
    'return DataFrame containing just the records related to the query cusip'
    # these are the records for the CUSIP and for any on-the-run bond for the CUSIP
    # the on-the-run bonds can vary by trace print
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
    df['effectivedatetime'] = seven.feature_makers.make_effectivedatetime(df)
    result = df.sort_values('effectivedatetime')
    return result


def check_trace_prints_for_nans(df, title):
    print title
    print 'len(df)', len(df)
    print 'cusip', 'n_nans'
    for cusip in set(df['cusip']):
        oasspreads = df.loc[cusip == df['cusip']]['oasspread']
        n_nans = sum(np.isnan(oasspreads))
        print cusip, n_nans


def read_and_transform_trace_prints(issuer, cusip, test):
    'return sorted DataFrame containing trace prints for the cusip and all related OTR cusips'
    trace_prints = seven.read_csv.input(
        issuer,
        'trace',
        nrows=1000 if test else None,
    )
    # make sure the trace prints are in non-decreasing datetime order
    sorted_trace_prints = sort_by_effectivedatetime(select_relevant_cusips(trace_prints, cusip))
    sorted_trace_prints['issuepriceid'] = sorted_trace_prints.index  # make the index one of the values
    return sorted_trace_prints


def accumulate_features_and_targets(trace_index, trace_record, feature_maker, target_maker):
    'return err or None; mutute feature_maker and target_maker to possible include the trace_record'
    err = feature_maker.append_features(trace_index, trace_record)
    if err is not None:
        return err
    # accumulate targets
    # NOTE: the trace_records are just for the cusip and for all of its OTR cusips
    err = target_maker.append_targets(trace_index, trace_record)
    if err is not None:
        return err
    return None


def do_work(control):
    'write predictions from fitted models to file system'
    def lap():
        'return ellapsed wall clock time:float since previous call to lap()'
        return control.timer.lap('lap', verbose=False)[1]

    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    # input files are for a specific ticker
    trace_prints = read_and_transform_trace_prints(
        control.arg.issuer,
        control.arg.cusip,
        control.arg.test,
    )
    print 'found %d trace prints for the cusip %s and its related OTR cusips' % (len(trace_prints), control.arg.cusip)
    cusip1s = set(trace_prints['cusip1'])

    check_trace_prints_for_nans(trace_prints, 'NaNs in relevant trace prints')

    # iterate over each relevant row
    # build and save the features for the cusip
    feature_maker = seven.feature_makers.AllFeatures(control.arg.issuer, control.arg.cusip, cusip1s)
    target_maker = seven.target_maker.TargetMaker()

    counter = collections.Counter()
    errors_on_selected_date = collections.Counter()

    def count(s):
        counter[s] += 1

    def skip(reason):
        # print 'skipped', reason
        count('skipped: ' + reason)

    selected_date = Date(from_yyyy_mm_dd=control.arg.effective_date).value  # a datetime.date
    mask_date_cusip = (trace_prints['effectivedate'] == selected_date) & (trace_prints['cusip'] == control.arg.cusip)
    n_predictions = sum(mask_date_cusip)
    last_trace_record_date = None
    print 'found %d trace prints on selected date %s with selected cusip %s' % (
        n_predictions,
        selected_date,
        control.arg.cusip,
    )
    print 'found %d distinct effective date times' % len(set(trace_prints['effectivedatetime']))
    print 'cusips in file:', set(trace_prints['cusip'])
    print 'prepared input in %s wall clock seconds' % lap()

    counter['feature and target records created'] = 0

    for trace_index, trace_record in trace_prints.iterrows():
        # Note: the CUSIP for each record is control.arg.cusip or a related OTR cusip
        count('n trace records seen')

        # assure records are in increasing order by effective datetime
        trace_record_date = trace_record['effectivedate'].to_pydatetime().date()  # a datetime.date
        if last_trace_record_date is not None:
            # if last_trace_record_date != trace_record_date:
            #     print 'found trace print record with date', trace_record_date
            assert last_trace_record_date <= trace_record_date
        last_trace_record_date = trace_record_date

        # skip if we are before first relevant date
        if trace_record_date < control.first_relevant_trace_print_date:
            skip('before first relevant trace print date')
            continue

        # stop once we go past the selected date
        if trace_record_date > selected_date:
            break  # OK to stop, since the trace_prints are in non-decreasing order by effectivedatetime
        count('n_trace_records processed')

        # accumulate features
        # mutate feature_maker and target_maker with the info in the trace_record
        if trace_record_date == selected_date:
            print 'found selected date', selected_date
        err = accumulate_features_and_targets(trace_index, trace_record, feature_maker, target_maker)
        if err is not None:
            skip(err)
            if trace_record_date == selected_date:
                pdb.set_trace()
                errors_on_selected_date[err] += 1
        else:
            count('feature and target records created')
            count('features and target records created on date %s' % trace_record['effectivedate'])

        # try to keep memory usage roughly constant
        # that enables running parallel instances on a single system
        gc.collect()

    print 'end loop on input in %.3f wall clock seconds' % lap()
    print 'counts'
    for k in sorted(counter.keys()):
        print '%71s: %6d' % (k, counter[k])
    print
    print 'feature and target maker errors on selected date %s' % selected_date
    if len(errors_on_selected_date) == 0:
        print 'NONE'
    else:
        for k in sorted(errors_on_selected_date.keys()):
            print '%71s: %6d' % (k, errors_on_selected_date[k])

    def write_selected(df, path, id):
        'write the subset that is on the selected date to the path as a CSV file'
        mask_on_selected_date = df['id_effectivedate'] == selected_date
        on_selected_date = df.loc[mask_on_selected_date]
        on_selected_date.to_csv(path)
        print 'wrote %d %s' % (len(on_selected_date), id)

    # write output files
    write_selected(
        feature_maker.get_dataframe(),
        control.path['out_features'],
        'features'
        )

    write_selected(
        feature_maker.get_dataframe(),
        control.path['out_targets'],
        'targets'
    )

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
