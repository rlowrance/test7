'''fit and predict all models on one CUSIP feature file for trade on one-date

NOTE: This program looks at all trades for the CUSIP and related OTR CUSIPs from the
beginning of time. Thus trades 10 years ago are used to predict next oasspreads today.
That is way too much history. We should fix this when we have a streaming infrastructure.
Most likely, only the last 1000 or so trades are relevant.

INVOCATION
  python fit_predict3.py {ticker} {cusip} {hpset} {effective_date} {--test} {--trace}
where
 ticker is the ticker symbol (ex: orcl)
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 hpset in {gridN} defines the hyperparameter set
 effective_date: YYYY-MM-DD is the date of the trade
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python fit_predict3.py ORCL 68389XAS4 grid2 2016-11-01  # last day we have is 2016-11-08 for this cusip STOPPED WORKING
 python fit_predict3.py ORCL 68389XAR6 grid2 2016-11-01  --test  # BUT no predictions, as XAR6 is usually missing oasspreads

See build.py for input and output files.

An earlier version of this program did checkpoint restart, but this version does not.
'''

from __future__ import division

import argparse
import collections
import copy
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

from seven import arg_type
from seven.models import ExceptionFit, ExceptionPredict, ModelNaive, ModelElasticNet, ModelRandomForests
from seven import HpGrids
import seven.feature_makers
import seven.read_csv
import seven.target_maker3

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

    paths = build.fit_predict3(arg.ticker, arg.cusip, arg.hpset, arg.effective_date, test=arg.test)
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
    skipped = collections.Counter()

    targets_on_requested_date = targets.loc[targets['id_effectivedate'] == desired_effective_date.value]
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
    pdb.set_trace()
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


def common_features_targets(features, targets):
    'return (DataFrame, DataFrame, err) where the data frames have the same indices'
    common_indices = features.index.intersection(targets.index)
    common_features = features.loc[common_indices]
    common_targets = targets.loc[common_indices]
    return common_features, common_targets


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
    assert len(cusip_records > 0)
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


def check_trace_prints_for_nans(df, title):
    print title
    print 'len(df)', len(df)
    print 'cusip', 'n_nans'
    for cusip in set(df['cusip']):
        oasspreads = df.loc[cusip == df['cusip']]['oasspread']
        n_nans = sum(np.isnan(oasspreads))
        print cusip, n_nans


def do_work(control):
    'write predictions from fitted models to file system'
    def trace_record_info(trace_record):
        return (
            trace_record['cusip'],
            trace_record['effectivedatetime'],
            trace_record['trade_type'],
            trace_record['quantity'],
            trace_record['oasspread'],
        )

    def fit_and_predict_tradetype_modelspec(trade_type, model_spec, query_features, training_features, training_targets):
        'return (prediction, importances, err)'
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
            model = model_constructor(model_spec, target_feature_name, control.random_seed)
            return model

        target_feature_name = 'target_next_oasspread_%s' % trade_type
        m = make_model(model_spec, target_feature_name)
        try:
            m.fit(training_features, training_targets)
        except ExceptionFit as e:
            msg = 'fit failed %s %s: %s' % (target_feature_name, model_spec, str(e))
            print msg
            return (None, None, msg)
        try:
            p = m.predict(query_features)  # the arg is a DataFrame
        except ExceptionPredict as e:
            msg = 'predict failed %s %s: %s' % (target_feature_name, model_spec, str(e))
            print msg
            return (None, None, msg)
        assert len(p) == 1
        prediction = p[0]
        # OUTPUT produced by previous version (expected downstream in our pipeline)
        # # carry into output additional info needed for error analysis,
        # # so that the error analysis programs do not need the original trace prints
        # n_predictions_created += 1
        # predictions['id_query_index'].append(query_index)
        # predictions['id_modelspec_str'].append(str(model_spec))
        # predictions['id_target_feature_name'].append(target_feature_name)
        # # copy all the info fields from the target row
        # for k, v in query_target_row.iteritems():
        #     if k.startswith('info_'):
        #         predictions['target_%s' % k].append(v)

        # predictions['actual'].append(actual)
        # predictions['predicted'].append(p[0])

        return (prediction, copy.deepcopy(m.importances), None)

    def fit_and_predict_query(trace_index, trace_record, fm):
        ' return [(trade_type, model_spec, prediction, importances, err)]'
        result = []
        print 'fit_and_predict_1', trace_record_info(trace_record)
        trace_record_cusip = trace_record['cusip']
        make_training_features = None  # need to redesign this
        training_features_all, err = make_training_features(trace_record_cusip, fm, trace=False)
        if err is not None:
            return None, None, 'make_training_features: %s' % err
        training_targets_all = seven.target_maker.make_training_targets(training_features_all, trace=False)
        align_features_and_targets = None
        training_features, training_targets, warnings = align_features_and_targets(training_features_all, training_targets_all)
        print 'warnings from align_features_and_targets'
        for warning in warnings:
            print trace_index, trace_record_cusip, warning
        assert len(training_features) == len(training_targets)
        # in order to buld the query features, we need to mutate the fms
        # rather than mutate the fm dict, we copy it and mutate the copy
        # mutate fm would be a mistake
        query_fm = copy.deepcopy(fm)
        err = query_fm[trace_record_cusip].append_features(trace_index, trace_record)
        if err is not None:
            msg = 'creating query features: %s' % err
            print msg
            result.append((None, None, None, None, msg))
        else:
            all_features, err = make_training_features(trace_record_cusip, query_fm)
            if err is not None:
                print 'not able to create features for the query', err
                pdb.set_trace()
            query_features = all_features.iloc[-1]
            # check that we got the right record
            assert query_features['id_effectivedatetime'] == trace_record['effectivedatetime']
            assert query_features['id_ticker_index'] == trace_index
            print query_features
            query_features_tradetype = (
                'B' if query_features['p_trade_type_is_B'] == 1 else
                'D' if query_features['p_trade_type_is_D'] == 1 else
                'S' if query_features['p_trade_type_is_S'] == 1 else
                None
            )
            assert query_features_tradetype is not None
            actual = query_features['p_oasspread']
            for model_spec in control.model_specs:
                # NOTE: Need to return the trade_type and model_spec as well, or write everything to disk
                prediction, importances, err = fit_and_predict_tradetype_modelspec(
                    query_features_tradetype,
                    model_spec,
                    query_features,
                    training_features,
                    training_targets,
                )
                if err is not None:
                    print 'fit_and_predict_query err:', err
                    result.append((query_features_tradetype, model_spec, actual, None, None, err))
                else:
                    result.append((query_features_tradetype, model_spec, actual, prediction, importances, err))
        return result

    def fit_and_predict(queries, fm, control):
        'return predictions: [float], importances: [Dict[feature_name, feature_value], err'
        predictions, importances, errs = [], [], []
        for trace_index, trace_record in queries.iterrows():
            prediction, importance, err = fit_and_predict_query(trace_index, trace_record, fm)
            predictions.append(prediction)
            importances.append(importance)
            errs.append(err)
        return predictions, importances, errs

    def save(predictions, importances, control):
        'write predictions and importancers to file system'
        pdb.set_trace()
        print 'save stub'
        return None

    def append_features(queries, fm):
        'mutate fm by appending each query; return errs'
        # MAYBE: append predictions only for trades near the query trade (a heuristic to run faster)
        errs = []
        for trace_index, trace_record in queries.iterrows():
            trace_record_cusip = trace_record['cusip']
            err = fm[trace_record_cusip].append_features(trace_index, trace_record)
            if err is not None:
                errs.append('trace index %s: %s' % (trace_index, err))
            else:
                errs.append(None)
        return errs

    def on_same_date(a, b):
        'return True or False'
        return (
            a.year == b.year and
            a.month == b.month and
            a.day == b.day
        )

    def after_date(a, b):
        'return True iff a is after b'
        return (
            a.year > b.year or
            a.month > b.month or
            a.day > b.day
        )

    def predict_if_selected(effectivedatetime, selected_date, selected_cusip, queries, fm):
        'return parallel lists ([prediction], [importances], [err]) potentially with no items'
        if on_same_date(effectivedatetime, selected_date):
            with_query_cusip = queries.loc[queries['cusip'] == selected_cusip]
            if len(with_query_cusip) > 0:
                predictions, importances, errs = fit_and_predict(with_query_cusip, fm, control)
                return predictions, importances, errs
            else:
                return [], [], []
        else:
            return [], [], []

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
    cusip1s = set(relevant_trace_prints['cusip1'])

    check_trace_prints_for_nans(relevant_trace_prints, 'NaNs in relevant trace prints')

    # iterate over each relevant row
    # build and save the features for the cusip
    # if the row is for the query cusip, create the training features and targets and predict the last trade
    fm = seven.feature_makers.AllFeatures(control.arg.ticker, control.arg.cusip, cusip1s)
    tm = seven.target_maker3.TargetMaker()

    n_predictions = 0
    n_features_created = collections.Counter()
    n_trace_records_seen = collections.Counter()
    skipped = collections.Counter()
    counts = collections.Counter()
    # days_tolerance = 7
    selected_date = pd.Timestamp(Date(from_yyyy_mm_dd=control.arg.effective_date).value)
    n_found = sum(relevant_trace_prints['effectivedate'] == selected_date)
    print 'found %d trace prints on date %s' % (n_found, selected_date)
    effectivedatetimes = set(relevant_trace_prints['effectivedatetime'])
    print 'found %d effective date times' % len(effectivedatetimes)
    for trace_index, trace_record in relevant_trace_prints.iterrows():
        print trace_index, trace_record_info(trace_record)
        # accumulate features
        err = fm.append_features(trace_index, trace_record)
        if err is not None:
            skipped[err] += 1
            print 'skipped', trace_index, err
            continue
        # accumulate targets
        cusip = trace_record['cusip']
        if cusip == control.arg.cusip:
            err = tm.append_targets(trace_index, trace_record)
            if err is not None:
                skipped[err] += 1
                print 'skipped', trace_index, err
                continue
            #
            features = fm.get_primary_features_dataframe()
            targets = tm.get_targets_dataframe()
            pdb.set_trace()
            common_features, common_targets = common_features_targets(features, targets)
            assert len(common_features) == len(common_targets)
            if len(common_features) == 0:
                err = 'no common indices in features and targets'
                print err
                skipped[err] += 1
                continue
            # fit_predict_features_targets(aligned_features, aligned_targets, control)
            print features
            print targets
            print 'found usable record'
            pdb.set_trace()
            continue
    print 'end loop on input'
    pdb.set_trace()
    return

    # OLD BELOW ME
    for i, effectivedatetime in enumerate(sorted(effectivedatetimes)):
        if after_date(effectivedatetime, selected_date):
            skipped['%s after selected date %s' % (effectivedatetime, selected_date)] += 1
            continue
        queries = relevant_trace_prints.loc[relevant_trace_prints['effectivedatetime'] == effectivedatetime]
        if i % 1000 == 1:
            print 'effectivedatetime %s # queries %3d (%d of %d)' % (
                effectivedatetime,
                len(queries),
                i + 1,
                len(effectivedatetimes),
            )
        # predict each query that was selected by the invocation
        predictions, importances, errs = predict_if_selected(effectivedatetime, selected_date, control.arg.cusip, queries, fm)
        for (prediction, importance, err) in zip(predictions, importances, errs):
            if err is not None:
                skipped['predict: %s' % err]
            else:
                save(prediction, importance, control)
                counts['saved'] += 1
        # for all queries, accumulate the features to be used in subsequent predictions
        errs = append_features(queries, fm)
        for err in errs:
            if err is not None:
                skipped['append_features: %s' % err] += 1
    # report on results
    print 'reasons trace prints were skipped'
    total_skipped = 0
    for k in sorted(skipped.keys()):
        print '%90s: %6d' % (k, skipped[k])
        total_skipped += skipped[k]
    print '%90s: %6d' % ('TOTAL SKIPPED', total_skipped)
    print
    print 'counts'
    for k in sorted(counts.keys()):
        print '%50s: %6d' % (k, counts[k])
    pdb.set_trace()
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
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
        datetime

    main(sys.argv)
