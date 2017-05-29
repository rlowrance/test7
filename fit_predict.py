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
 python fit_predict.py ORCL 68389XAS4 grid3 2016-11-01  # production
 python fit_predict.py ORCL 68389XAS4 grid1 2016-11-01  # grid1 is a small mesh
 python fit_predict.py ORCL 68389XAS4 grid1 2013-08-01 --test  # 1 CUSIP, 15 trades
 python fit_predict.py ORCL 68389XAR6 grid1 2016-11-01  --test  # BUT no predictions, as XAR6 is usually missing oasspreads

See build.py for input and output files.

An earlier version of this program did checkpoint restart, but this version does not.

Operation deployment.

See the file fit_predict.org for an explanation of the design of this program.
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import datetime
import gc
import numpy as np
import os
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
from seven.models import ExceptionFit, ExceptionPredict, ModelNaive, ModelElasticNet, ModelRandomForests
from seven import HpGrids
import seven.feature_makers
import seven.fit_predict_output
import seven.read_csv
import seven.target_maker

pp = pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', type=seven.arg_type.ticker)
    parser.add_argument('cusip', type=seven.arg_type.cusip)
    parser.add_argument('hpset', type=seven.arg_type.hpset)
    parser.add_argument('effective_date', type=seven.arg_type.date)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = seven.build.fit_predict(arg.ticker, arg.cusip, arg.hpset, arg.effective_date, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])
    # delete output files, so that they are rebuilt
    # the code in function append_to_csv() will extend existing files
    for logical_name, path_to_file in paths.iteritems():
        if logical_name.startswith('out_'):
            if os.path.isfile(path_to_file):
                os.remove(path_to_file)

    model_spec_iterator = (
        HpGrids.HpGrid0 if arg.hpset == 'grid0' else
        HpGrids.HpGrid1 if arg.hpset == 'grid1' else
        HpGrids.HpGrid2 if arg.hpset == 'grid2' else
        HpGrids.HpGrid3 if arg.hpset == 'grid3' else
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
    cusip_records = records.loc[has_query_cusip]
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


def fit_and_predict(trade_type, model_spec, training_features, training_targets, queries, random_seed):
    'return (prediction, importance, err)'
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
        model = model_constructor(model_spec, target_feature_name, random_seed)
        return model

    # train only on features that were built for the trade_type
    if len(training_features) == 0:
        return None, None, 'no training samples'
    with_trade_type = training_targets['id_trade_type'] == trade_type
    training_features_with_trade_type = training_features.loc[with_trade_type]
    training_targets_with_trade_type = training_targets.loc[with_trade_type]
    assert len(training_features_with_trade_type) == len(training_targets_with_trade_type)
    if len(training_features_with_trade_type) == 0:
        return (None, None, 'no training features with trade type')

    target_feature_name = 'target_oasspread_%s' % trade_type
    m = make_model(model_spec, target_feature_name)

    try:
        m.fit(training_features_with_trade_type, training_targets_with_trade_type)
    except ExceptionFit as e:
        msg = 'fit failed %s %s: %s' % (target_feature_name, model_spec, str(e))
        print msg
        return (None, None, msg)

    try:
        p = m.predict(queries)  # the arg is a DataFrame
    except ExceptionPredict as e:
        msg = 'predict failed %s %s: %s' % (target_feature_name, model_spec, str(e))
        print msg
        return (None, None, msg)
    assert len(p) == 1
    prediction = p[0]
    return prediction, m.importances, None


def fit_predict_all_modelspecs(control, trace_index, trace_record, common_features, common_targets, importances, predictions):
    'return [err]; mutute importances and predictions to the predictions from fitting all model specs'
    features_queries = common_features.iloc[[-1]]   # a DataFrame with one row
    target_query = common_targets.iloc[-1]          # a Series
    trade_type = target_query['id_trade_type']
    errs = []
    for model_spec in control.model_specs:
        prediction, importance, err = fit_and_predict(
            trade_type=trade_type,
            model_spec=model_spec,
            training_features=common_features.iloc[:-1],   # training features exclude the query
            training_targets=common_targets.iloc[:-1],     # training targets exclude the query
            queries=features_queries,                      # the query is the most recent trace-print
            random_seed=control.random_seed,
        )
        if err is not None:
            errs.append(err)
        effectivedatetime = trace_record['effectivedatetime']
        output_key = seven.fit_predict_output.OutputKey(
            trace_index=trace_index,
            model_spec=model_spec,
        )
        assert output_key not in predictions
        assert output_key not in importances
        predictions[output_key] = seven.fit_predict_output.Prediction(
            effectivedatetime=effectivedatetime,
            trade_type=trade_type,
            quantity=trace_record['quantity'],
            interarrival_seconds=features_queries.iloc[0]['p_interarrival_seconds'],
            actual=trace_record['oasspread'],
            prediction=prediction,
        )
        importances[output_key] = seven.fit_predict_output.Importance(
            effectivedatetime=effectivedatetime,
            trade_type=trade_type,
            importance=importance,
        )
    return errs


def read_and_transform_trace_prints(ticker, cusip, test):
    'return sorted DataFrame containing trace prints for the cusip and all related OTR cusips'
    trace_prints = seven.read_csv.input(
        ticker,
        'trace',
        nrows=40000 if test else None,
    )
    # make sure the trace prints are in non-decreasing datetime order
    sorted_trace_prints = sort_by_effectivedatetime(make_relevant_cusips(trace_prints, cusip))
    sorted_trace_prints['issuepriceid'] = sorted_trace_prints.index  # make the index one of the values
    return sorted_trace_prints


def append_to_csv(df, path):
    if os.path.isfile(path):
        # append to existing files
        with open(path, 'a') as f:
            df.to_csv(f, header=False)
    else:
        # create a new file
        with open(path, 'w') as f:
            df.to_csv(f, header=True)


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


def trace_print_is_selected(trace_record, selected_date, selected_cusip):
    'return err or None (in which case the record is selected)'
    if not (trace_record['cusip'] == selected_cusip):
        return 'not on invocation-specified cusip %s' % selected_cusip
    if not (trace_record['effectivedate'].date() == selected_date):
        return 'not on invocation-specified date %s' % selected_date
    return None


def do_work(control):
    'write predictions from fitted models to file system'
    def lap():
        'return ellapsed wall clock time:float since previous call to lap()'
        return control.timer.lap('lap', verbose=False)[1]

    def trace_record_info(trace_record):
        return (
            trace_record['cusip'],
            trace_record['effectivedatetime'],
            trace_record['trade_type'],
            trace_record['quantity'],
            trace_record['oasspread'],
        )

    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    # input files are for a specific ticker
    trace_prints = read_and_transform_trace_prints(control.arg.ticker, control.arg.cusip, control.arg.test)
    print 'found %d trace prints for the cusip %s and its related OTR cusips' % (len(trace_prints), control.arg.cusip)
    # trace_prints = seven.read_csv.input(
    #     control.arg.ticker,
    #     'trace',
    #     nrows=40000 if control.arg.test else None,
    # )
    # relevant_trace_prints = sort_by_effectivedatetime(make_relevant_cusips(trace_prints, control.arg.cusip))
    # relevant_trace_prints['issuepriceid'] = relevant_trace_prints.index  # make the index one of the values
    cusip1s = set(trace_prints['cusip1'])

    check_trace_prints_for_nans(trace_prints, 'NaNs in relevant trace prints')

    # iterate over each relevant row
    # build and save the features for the cusip
    # if the row is for the query cusip, create the training features and targets and predict the last trade
    feature_maker = seven.feature_makers.AllFeatures(control.arg.ticker, control.arg.cusip, cusip1s)
    target_maker = seven.target_maker.TargetMaker()

    counter = collections.Counter()

    def count(s):
        counter[s] += 1

    def skip(reason):
        # print 'skipped', reason
        count('skipped: ' + reason)

    # days_tolerance = 7
    selected_date = Date(from_yyyy_mm_dd=control.arg.effective_date).value  # a datetime.date
    mask_date_cusip = (trace_prints['effectivedate'] == selected_date) & (trace_prints['cusip'] == control.arg.cusip)
    n_predictions = sum(mask_date_cusip)
    last_trace_record_date = None
    importances = {}
    predictions = {}
    print 'found %d trace prints on selected date %s with selected cusip %s' % (
        n_predictions,
        selected_date,
        control.arg.cusip,
    )
    print 'found %d distinct effective date times' % len(set(trace_prints['effectivedatetime']))
    print 'cusips in file:', set(trace_prints['cusip'])
    print 'prepared input in %s wall clock seconds' % lap()

    for trace_index, trace_record in trace_prints.iterrows():
        # Note: the CUSIP for each record is control.arg.cusip or a related OTR cusip
        count('n trace recordds seen')

        # assure records are in increasing order by effective datetime
        trace_record_date = trace_record['effectivedate']
        if last_trace_record_date is not None:
            assert last_trace_record_date <= trace_record_date
        last_trace_record_date = trace_record_date

        # stop once we go past the selected date
        if trace_record_date.date() > selected_date:
            break  # OK, since the trace_prints are in non-decreasing order by effectivedatetime
        count('n_trace_records processed')

        # accumulate features
        err = accumulate_features_and_targets(trace_index, trace_record, feature_maker, target_maker)
        if err is not None:
            skip(err)
            continue
        count('feature and target records created')

        # make predictions only for the selected cusip trace records on the selected date
        err = trace_print_is_selected(trace_record, selected_date, control.arg.cusip)
        if err is not None:
            skip(err)
            continue
        count('trace_prints selected for prediction')

        # align the features and targets
        # there may be some feature records for which there are not target
        # there may be some target records for which there are no features
        common_features, common_targets = common_features_targets(
            feature_maker.get_primary_features_dataframe(),  # all the features so far
            target_maker.get_targets_dataframe(),            # all the targets so far
        )
        assert len(common_features) == len(common_targets)
        if len(common_features) == 0:
            skip('no common indices in features and targets')
            continue
        if not len(common_features > 2):
            # the final sample is used as ground truth, the previous ones for training
            skip('not at least 2 samples')
            continue

        lap()  # time just the fit and predict code, not the assembling-the-data code
        # append to predictions and importances
        errs = fit_predict_all_modelspecs(
            control=control,
            trace_index=trace_index,
            trace_record=trace_record,
            common_features=common_features,
            common_targets=common_targets,
            importances=importances,
            predictions=predictions,
        )
        for err in errs:
            skip(err)
        count('calls to fit_predict_all')
        print 'fitted and predicted %02d of %02d %d model specs for %s %s %s in %.3f wall clock seconds' % (
            counter['calls to fit_predict_all'],
            n_predictions,
            len(control.model_specs),
            control.arg.ticker,
            control.arg.cusip,
            control.arg.effective_date,
            lap(),
        )

        gc.collect()   # try to keep memory usage roughly constant

    print 'end loop on input in %.3f wall clock seconds' % lap()
    print 'counts'
    for k in sorted(counter.keys()):
        print '%70s: %6d' % (k, counter[k])

    # write the output files
    # make sure that they are readable
    with open(control.path['out_importances'], 'wb') as f:
        pickle.dump(importances, f, pickle.HIGHEST_PROTOCOL)
        print 'wrote %d importances' % len(importances)
    with open(control.path['out_importances'], 'rb') as f:
        obj = pickle.load(f)
        assert isinstance(obj, object)
    with open(control.path['out_predictions'], 'wb') as f:
        pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)
        print 'wrote %d predictions' % len(predictions)
    with open(control.path['out_predictions'], 'rb') as f:
        obj = pickle.load(f)
        assert isinstance(obj, object)
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
