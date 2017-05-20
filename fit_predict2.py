'''fit and predict all models on one CUSIP feature file for trade on one-date

NOTE: This program looks at all trades for the CUSIP and related OTR CUSIPs from the
beginning of time. Thus trades 10 years ago are used to predict next oasspreads today.
That is way too much history. We should fix this when we have a streaming infrastructure.
Most likely, only the last 1000 or so trades are relevant.

INVOCATION
  python fit_predict2.py {ticker} {cusip} {hpset} {effective_date} {--test} {--trace}
where
 ticker is the ticker symbol (ex: orcl)
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 hpset in {gridN} defines the hyperparameter set
 effective_date: YYYY-MM-DD is the date of the trade
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python fit_predict2.py orcl 68389XAS4 grid2 2016-11-01  # last day we have is 2016-11-08 for this cusip STOPPED WORKING
 python fit_predict2.py orcl 68389XAR6 grid2 2016-11-01  --test  # BUT no predictions, as XAR6 is usually missing oasspreads

See build.py for input and output files.

An earlier version of this program did checkpoint restart, but this version does not.
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

import applied_data_science.dirutility
import applied_data_science.lower_priority
import applied_data_science.pickle_utilities

from applied_data_science.Bunch import Bunch
from applied_data_science.Date import Date
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

from seven import arg_type
from seven.models import ExceptionFit, ModelNaive, ModelElasticNet, ModelRandomForests
from seven import HpGrids
import seven.feature_makers
import seven.read_csv
import seven.target_maker

import build
pp = pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', type=arg_type.ticker)
    parser.add_argument('cusip', type=arg_type.cusip)
    parser.add_argument('hpset', type=arg_type.hpset)
    parser.add_argument('effective_date', type=arg_type.date)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = build.fit_predict2(arg.ticker, arg.cusip, arg.hpset, arg.effective_date, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    model_spec_iterator = (
        HpGrids.HpGrid0 if arg.hpset == 'grid0' else
        HpGrids.HpGrid1 if arg.hpset == 'grid1' else
        HpGrids.HpGrid2 if arg.hpset == 'grid2' else
        None
        )().iter_model_specs
    model_specs = [
        model_spec
        for model_spec in model_spec_iterator()
    ]

    return Bunch(
        arg=arg,
        path=paths,
        model_specs=model_specs,
        random_seed=random_seed,
        timer=Timer(),
    )


def fit_predict(
    features=None,
    targets=None,
    desired_effective_date=None,
    model_specs=None,
    test=None,
    random_state=None,
    path=None,
):
    'write CSV files with predictions and importances for trades on specified date'
    # return true if files are written, false otherwise
    def make_model(model_spec, target_feature_name):
        'return a constructed Model instance'
        model_constructor = (
            ModelNaive if model_spec.name == 'n' else
            ModelElasticNet if model_spec.name == 'en' else
            ModelRandomForests if model_spec.name == 'rf' else
            None
        )
        if model_constructor is None:
            print 'error: bad model_spec.name %s' % model_spec.name
            pdb.set_trace()
        model = model_constructor(model_spec, target_feature_name, random_state)
        return model

    def already_seen_lambda(query_index, model_spec, predicted_feature_name):
        return False

    def select_rows_before(df, effectivedatetime):
        'return DataFrame with only those rows before the specified datatime'
        pdb.set_trace()
        mask = df['id_effectivedatetime'] < effectivedatetime
        selected = df.loc[mask]
        return selected

    print 'debug fit_predict'
    pdb.set_trace()
    skipped = collections.Counter()

    targets_on_requested_date = targets.loc[targets['info_this_effectivedate'] == desired_effective_date.value]
    print 'found %d trades on the requested date' % len(targets_on_requested_date)
    if len(targets_on_requested_date) == 0:
        msg = 'no targets for desired effective date %s' % str(desired_effective_date)
        print msg
        skipped[msg] += 1
        return False

    n_predictions_created = 0
    predictions = collections.defaultdict(list)
    importances = collections.defaultdict(list)
    target_feature_names = [
        'target_next_oasspread_%s' % tradetype
        for tradetype in ('B', 'D', 'S')
    ]
    n_predictions_created = 0
    test = False
    for query_index, query_target_row in targets_on_requested_date.iterrows():
        # print 'query_index', query_index
        # print query_target_row
        if query_index not in features.index:
            # the targets and features are independently constructed so that
            # there is not a one-to-one correspondence between their unique IDs (the query_index)
            skipped['query_index %s not in features' % query_index] += 1
            continue
        effectivedatetime = query_target_row['info_this_effectivedatetime']
        # train on all the features and targets not after the effectivedatetime
        # many of the training samples will be before the effectivedate
        training_features_all = features.loc[features['id_effectivedatetime'] <= effectivedatetime]
        training_targets_all = targets.loc[targets['info_this_effectivedatetime'] <= effectivedatetime]

        # scikit-learn doesn't handle missing value
        # we have some missing values (coded as NaNs) in the targets
        # find them and eliminate those targets
        indices_with_missing_targets = set()
        print 'training indices with at least one missing target feature'
        for training_index, training_row in training_targets_all.iterrows():
            for target_feature_name in target_feature_names:
                if np.isnan(training_row[target_feature_name]):
                    indices_with_missing_targets.add(training_index)
                    print training_index, target_feature_name

        common_indices = training_features_all.index.intersection(training_targets_all.index)
        usable_indices = common_indices.difference(indices_with_missing_targets)
        training_features = training_features_all.loc[usable_indices]
        training_targets = training_targets_all.loc[usable_indices]
        print 'query index %s effectivedatetime %s num training samples available %d' % (
            query_index,
            effectivedatetime,
            len(usable_indices),
        )
        # predict each of the 3 possible targets for each of the model_specs
        for target_feature_name in target_feature_names:
                actual = query_target_row[target_feature_name]
                for model_spec in model_specs:
                    print ' ', target_feature_name, str(model_spec)
                    m = make_model(model_spec, target_feature_name)
                    try:
                        m.fit(training_features, training_targets)
                    except ExceptionFit as e:
                        print 'fit failure for query_index %s model_spec %s: %s' % (query_index, model_spec, str(e))
                        skipped[e] += 1
                        continue  # give up on this model_spec
                    p = m.predict(features.loc[[query_index]])  # the arg is a DataFrame
                    assert len(p) == 1
                    # carry into output additional info needed for error analysis,
                    # so that the error analysis programs do not need the original trace prints
                    n_predictions_created += 1
                    predictions['id_query_index'].append(query_index)
                    predictions['id_modelspec_str'].append(str(model_spec))
                    predictions['id_target_feature_name'].append(target_feature_name)
                    # copy all the info fields from the target row
                    for k, v in query_target_row.iteritems():
                        if k.startswith('info_'):
                            predictions['target_%s' % k].append(v)

                    predictions['actual'].append(actual)
                    predictions['predicted'].append(p[0])

                    for feature_name, importance in m.importances.items():
                        importances['id_query_index'].append(query_index)
                        importances['id_modelspec_str'].append(str(model_spec))
                        importances['id_target_feature_name'].append(target_feature_name)
                        importances['id_feature_name'].append(feature_name)

                        importances['importance'].append(importance)
                    gc.collect()  # try to get memory usage roughly constant
        if test:
            break

    # create and write csv file for predictions and importances
    print 'created %d predictions' % n_predictions_created
    if len(skipped) > 0:
        print 'skipped some features; reasons and counts:'
        for k in sorted(skipped.keys()):
            print '%40s: %s' % (k, skipped[k])
    predictions_df = pd.DataFrame(
        data=predictions,
        index=[
            predictions['id_query_index'],
            predictions['id_modelspec_str'],
            predictions['id_target_feature_name'],
        ],
    )
    predictions_df.to_csv(path['out_predictions'])
    print 'wrote %d predictions to %s' % (len(predictions_df), path['out_predictions'])
    importances_df = pd.DataFrame(
        data=importances,
        index=[
            importances['id_query_index'],
            importances['id_modelspec_str'],
            importances['id_target_feature_name'],
            importances['id_feature_name'],
        ],
    )
    importances_df.to_csv(path['out_importances'])
    print 'wrote %d importances to %s' % (len(importances_df), path['out_importances'])
    return True


def align_features_and_targets(features, targets):
    'return (aligned_features, aligned_targets)'
    # print 'debug align_features_and_targets', len(features), len(targets)
    aligned_features = pd.DataFrame(columns=features.columns)
    aligned_targets = pd.DataFrame(columns=targets.columns)
    for feature_index, feature_row in features.iterrows():
        feature_row_effectivedatetime = feature_row['id_effectivedatetime']
        target_df_mask = targets['id_effectivedatetime'] == feature_row_effectivedatetime
        target_df = targets.loc[target_df_mask]
        if len(target_df) == 0:
            continue
        elif len(target_df) == 1:
            assert len(aligned_features) == len(aligned_targets)
            aligned_features.loc[len(aligned_features)] = feature_row
            aligned_targets.loc[len(aligned_targets)] = target_df.iloc[0]
        else:
            print 'error: more than one target with effectivedatetime from feature'
            print feature_row_effectivedatetime, len(target_df)
            pdb.set_trace()
    return aligned_features, aligned_targets


def make_relevant_cusips(records, query_cusip):
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
            # result for all records
            # cusip 68389XAS4 otr cusips set(['68389XAS4', '68389XBL8'])
            # cusip 68389XAU9 otr cusips set(['68389XAU9'])
            # cusip 76657RAB2 otr cusips set(['76657RAB2'])
            # cusip 68389XAW5 otr cusips set(['68389XAW5'])
            # cusip 68402LAC8 otr cusips set(['68402LAC8'])
            # cusip 68389XBE4 otr cusips set(['68389XBE4'])
            # cusip 68389XBG9 otr cusips set(['68389XBG9'])
            # cusip 68389XBA2 otr cusips set(['68389XBA2', '68389XBK0'])
            # cusip 68389XAK1 otr cusips set(['68389XAK1'])
            # cusip 68389XAM7 otr cusips set(['68389XAM7'])
            # cusip 68389XAX3 otr cusips set(['68389XAX3'])
            # cusip 68389XBF1 otr cusips set(['68389XBF1'])
            # cusip 68389XBC8 otr cusips set(['68389XBC8'])
            # cusip 68389XAG0 otr cusips set(['68389XAG0'])
            # cusip 68389XAE5 otr cusips set(['68389XAE5'])
            # cusip 68389XAC9 otr cusips set(['68389XAC9'])
            # cusip 68389XAR6 otr cusips set(['68389XAR6', '68389XAQ8'])
            # cusip 68389XAT2 otr cusips set(['68389XAT2'])
            # cusip U68308AA0 otr cusips set(['68389XAK1'])
            # cusip 64118QAB3 otr cusips set(['68389XAC9'])
            # cusip 68389XAV7 otr cusips set(['68389XAV7'])
            # cusip 68389XAL9 otr cusips set(['68389XAM7'])
            # cusip 68389XBD6 otr cusips set(['68389XBD6'])
            # cusip 68389XAJ4 otr cusips set(['68389XAK1'])
            # cusip 68389XAH8 otr cusips set(['68389XAH8'])
            # cusip 68389XBK0 otr cusips set(['68389XBA2', '68389XBK0'])
            # cusip 68389XBB0 otr cusips set(['68389XBB0'])
            # cusip 68389XAY1 otr cusips set(['68389XAY1', '68389XAX3'])
            # cusip 68389XBL8 otr cusips set(['68389XAS4', '68389XBL8'])
            # cusip 68389XAN5 otr cusips set(['68389XAN5'])
            # cusip 68389XAP0 otr cusips set(['68389XAP0'])
            # cusip 68389XAQ8 otr cusips set(['68389XAQ8'])
            # cusip U68308AB8 otr cusips set(['68389XAM7'])
            # cusip 68389XAD7 otr cusips set(['68389XAD7'])
            # cusip 68389XAF2 otr cusips set(['68389XAF2'])
            # cusip 68389XBJ3 otr cusips set(['68389XBJ3'])
            # cusip 68389XBM6 otr cusips set(['68389XBM6'])
            # cusip 68389XBH7 otr cusips set(['68389XBH7'])
    has_query_cusip = records['cusip'] == query_cusip
    cusip_records = records[has_query_cusip]
    assert len(cusip_records > 0), set(records['cusip'])
    otr_cusips = set(cusip_records['cusip1'])
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


def check_trace_prints(df, title):
    print title
    print 'len(df)', len(df)
    print 'cusip', 'n_nans'
    for cusip in set(df['cusip']):
        oasspreads = df.loc[cusip == df['cusip']]['oasspread']
        n_nans = sum(np.isnan(oasspreads))
        print cusip, n_nans


def make_training_features(trace_index, trace_record, fm):
    'return (True, DataFrame of training features) or (False, error message)'
    cusip = trace_record['cusip']
    cusip_d = fm[cusip].d.copy()  # a collections.defaultdict
    cusip_indices = fm[cusip].trace_indices
    for index, cusip1 in zip(cusip_indices, cusip_d['id_cusip1']):
        # print index, cusip1
        if index not in fm[cusip1].trace_indices:
            return (False, 'index %s not in cusip1 %s for cusip %s' % (index, cusip1, cusip))
        index_offset = 0
        while fm[cusip].trace_indices[index_offset] != index:
            index_offset += 1
        # append values from cusip1
        for feature_name in cusip_d.keys():
            if feature_name.startswith('p_'):
                otr_feature_name = 'otr1_' + feature_name[2:]
                otr_feature_value = fm[cusip1].d[feature_name][index_offset]
                cusip_d[otr_feature_name].append(otr_feature_value)
    try:
        result = pd.DataFrame(
            data=cusip_d,
            index=cusip_indices
        )
    except ValueError as e:
        print 'make_training_features exception:', str(e)
        pdb.set_trace()
        for k, v in cusip_d.iteritems():
            print k, len(v)
        pdb.set_trace()
    return (True, result)


def do_work(control):
    'write predictions from fitted models to file system'

    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    # input files are for a specific ticker
    trace_prints = seven.read_csv.input(
        control.arg.ticker,
        'trace',
        nrows=40000 if control.arg.test else None,
    )
    relevant_trace_prints = sort_by_effectivedatetime(make_relevant_cusips(trace_prints, control.arg.cusip))
    relevant_trace_prints['issuepriceid'] = relevant_trace_prints.index  # make the index one of the values

    print relevant_trace_prints.columns
    print relevant_trace_prints.index
    print relevant_trace_prints.effectivedatetime
    print len(relevant_trace_prints)
    check_trace_prints(relevant_trace_prints, 'NaNs in relevant trace prints')

    # iterate over each relevant row
    # build and save the features for the cusip
    # if the row is for the query cusip, create the training features and targets and predict the last trade
    fm = {}
    for cusip in set(relevant_trace_prints['cusip']):
        fm[cusip] = seven.feature_makers.AllFeatures(control.arg.ticker)
    n_predictions = 0
    n_features_created = collections.Counter()
    n_trace_records_seen = collections.Counter()
    skipped = collections.Counter()
    days_tolerance = 7
    pdb.set_trace()
    selected_date = pd.Timestamp(Date(from_yyyy_mm_dd=control.arg.effective_date).value)
    for trace_index, trace_record in relevant_trace_prints.iterrows():
        if trace_record['effectivedate'] > selected_date:
            skipped['after control.arg.effective_date'] += 1
            continue
        if (selected_date - trace_record['effectivedate']).days > days_tolerance:
            skipped['more than %d days before query trade' % days_tolerance] += 1
            continue
        pdb.set_trace()
        cusip = trace_record['cusip']
        n_trace_records_seen[cusip] += 1
        created_features = fm[cusip].append_features(trace_index, trace_record, verbose=False)
        if created_features:
            # here if we created more features
            n_features_created[cusip] += 1
            if cusip == control.arg.cusip:
                ok, training_features = make_training_features(trace_index, trace_record, fm)
                if not ok:
                    print 'skipped:', training_features
                    skipped[training_features] += 1
                    continue
                training_targets = seven.target_maker.make_training_targets(training_features, verbose=False, trace=False)
                aligned_features, aligned_targets = align_features_and_targets(training_features, training_targets)
                assert len(aligned_features) == len(aligned_targets)
                if len(aligned_features) > 0:
                    skip_fitting = False
                    if skip_fitting:
                        print 'SKIPPING fit_predict', trace_index, len(aligned_features)
                    else:
                        fit_predict(
                            features=aligned_features,
                            targets=aligned_targets,
                            desired_effective_date=Date(from_yyyy_mm_dd=control.arg.effective_date),
                            model_specs=control.model_specs,
                            test=control.arg.test,
                            random_state=control.random_seed,
                            path=control.path,
                        )
                    n_predictions += 1
            else:
                print 'not creating training data for cusip', cusip
    print ' '
    print '****************************************'
    print 'control.arg.cusip', control.arg.cusip
    print 'created %d predictions from %d relevant trace print records' % (n_predictions, len(relevant_trace_prints))
    print ' '
    print 'cusip/features_created/trace_records seen by cusip'
    for cusip, count in n_trace_records_seen.iteritems():
        print '%10s %6d %6d' % (cusip, n_features_created[cusip], count)
    print ' '
    print 'reasons trace records were skipped by feature makers, by cusip'
    for cusip, fm in fm.iteritems():
        for reason, count in fm.skipped.iteritems():
            print '%10s %40s %6d %6d' % (cusip, reason, count, sum(relevant_trace_prints['cusip'] == cusip))
        print
    pdb.set_trace()
    pass

    # # OLD BELOW ME: create the features and targets on the fly

    # # TODO: sort the trace file by effectivedatetime
    # # the index was built from the column issuepriceid, which isn't in the dataframe columns
    # features = pd.read_csv(
    #     control.path['in_features'],
    #     nrows=None,
    #     parse_dates=['id_effectivedatetime'],
    #     index_col=0,
    # )
    # print 'read %d rows from feature file %s' % (len(features), control.path['in_features'])
    # assert len(features) > 0
    # # deconstruct effective datetime into its date components
    # features['id_effectiveyear'] = features['id_effectivedatetime'].dt.year
    # features['id_effectivemonth'] = features['id_effectivedatetime'].dt.month
    # features['id_effectiveday'] = features['id_effectivedatetime'].dt.day
    # print 'features.columns'
    # pp(features.columns)
    # targets = None
    # # targets = read_csv(
    # #     control.path['in_targets'],
    # #     nrows=None,
    # #     parse_dates=['info_subsequent_effectivedatetime', 'info_this_effectivedatetime'],
    # # )
    # # print 'targets.columns'
    # # pp(targets.columns)
    # # assert len(targets) > 0
    # # targets['info_this_effectivedate'] = targets['info_this_effectivedatetime'].dt.date
    # # print targets.columns

    # # NOTE: The features and targets files are build using independent criteria,
    # # so that the indices should not in general be the same
    # print 'len(features): %d  len(targets): %d' % (len(features), len(targets))
    # # keep only the features and targets that have common indices
    # # common_indices = features.index.copy().intersection(targets.index)
    # # common_features = features.loc[common_indices]
    # # common_targets = targets.loc[common_indices]
    # result = fit_predict(  # write records to output files
    #     features=features,
    #     targets=targets,
    #     desired_effective_date=Date(from_yyyy_mm_dd=control.arg.effective_date),
    #     model_specs=control.model_specs,
    #     test=control.arg.test,
    #     # already_seen=already_seen,
    #     random_state=control.random_seed,
    #     path=control.path,
    # )
    # if not result:
    #     print 'actual data records not written, perhaps input is empty'
    #     # write empty output files so that the build process will know that this program has run

    #     def touch(path):
    #         with open(path, 'a'):
    #             os.utime(path, None)

    #     touch(control.path['out_importances'])
    #     touch(control.path['out_predictions'])


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
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
        datetime

    main(sys.argv)
