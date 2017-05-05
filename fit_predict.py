'''fit and predict all models on one CUSIP feature file

INVOCATION
  python fit_predict.py {ticker} {cusip} {hpset} {effective_date} {--test} {--trace}
where
 ticker is the ticker symbol (ex: orcl)
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 hpset in {gridN} defines the hyperparameter set
 effective_date: YYYY-MM-DD is the date of the trade
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python fit_predict.py orcl 68389XAS4 grid2 2016-11-01  # last day we have is 2016-11-08 for this cusip

See build.py for input and output files.

An earlier version of this program did checkpoint restart, but this version does not.
'''

from __future__ import division

import argparse
import collections
import datetime
import gc
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import applied_data_science.dirutility
import applied_data_science.lower_priority
import applied_data_science.pickle_utilities
import applied_data_science.timeseries as timeseries

from applied_data_science.Bunch import Bunch
from applied_data_science.Date import Date
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

from seven import arg_type
from seven.models import ModelNaive, ModelElasticNet, ModelRandomForests
from seven import HpGrids

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

    paths = build.fit_predict(arg.ticker, arg.cusip, arg.hpset, arg.effective_date, test=arg.test)
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
    'append to pickler file, a prediction (when possible) for each target sample on the effectve_date'
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

    skipped = collections.Counter()
    relevant_targets = targets.loc[targets.id_effectivedate == desired_effective_date.value]
    print 'found %d trades on the requested date' % len(relevant_targets)
    if len(relevant_targets) == 0:
        msg = 'no targets for desired effective date %s' % str(desired_effective_date)
        print msg
        skipped[msg] += 1
        return False

    n_predictions_created = 0
    predictions = collections.defaultdict(list)
    importances = collections.defaultdict(list)
    tsfp = timeseries.FitPredict()
    for fit_predict_ok, fit_predict_result in tsfp.fit_predict(
        df_features=features,
        df_targets=relevant_targets,
        make_model=make_model,
        model_specs=model_specs,
        timestamp_feature_name='id_effectivedatetime',
        already_seen_lambda=already_seen_lambda,
    ):
        if fit_predict_ok:
            # save the results
            n_predictions_created += 1
            if n_predictions_created % 1000 == 1:
                print 'new row # %d %s %s %s' % (
                    n_predictions_created,
                    fit_predict_result.query_index,
                    fit_predict_result.model_spec,
                    fit_predict_result.predicted_feature_name,
                )
            if n_predictions_created % 100 == 0:
                gc.collect()  # keep memory usage low, so that multiple fit_predict's can be run concurrently
            # build unique ID
            predictions['query_index'].append(fit_predict_result.query_index)
            predictions['model_spec_str'].append(str(fit_predict_result.model_spec))
            predictions['predicted_feature_name'].append(fit_predict_result.predicted_feature_name)

            # build payload
            if np.isnan(fit_predict_result.predicted_feature_value):
                print 'found NaN prediction'
                print fit_predict_result
                pdb.set_trace()
            predictions['predicted'].append(fit_predict_result.prediction)
            predictions['actual'].append(fit_predict_result.predicted_feature_value)
            predictions['n_training_samples'].append(fit_predict_result.n_training_samples)

            for feature_name, importance in fit_predict_result.fitted_model.importances.items():
                # build unique ID
                importances['query_index'].append(fit_predict_result.query_index)
                importances['model_spec_str'].append(str(fit_predict_result.model_spec))
                importances['predicted_feature_name'].append(fit_predict_result.predicted_feature_name)
                importances['feature_name'].append(feature_name)

                # build payload
                importances['importances'].append(importance)
        else:
            print 'skipped', fit_predict_result
            skipped[fit_predict_result] += 1

    print 'created %d predictions' % n_predictions_created
    print 'skipped some features; reasons and counts:'
    for k in sorted(skipped.keys()):
        print '%40s: %s' % (k, skipped[k])
    predictions_df = pd.DataFrame(
        data=predictions,
        index=[
            predictions['query_index'],
            predictions['model_spec_str'],
            predictions['predicted_feature_name'],
        ],
    )
    predictions_df.to_csv(path['out_predictions'])
    print 'wrote %d predictions to %s' % (len(predictions_df), path['out_predictions'])
    importances_df = pd.DataFrame(
        data=importances,
        index=[
            importances['query_index'],
            importances['model_spec_str'],
            importances['predicted_feature_name'],
            importances['feature_name'],
        ],
    )
    importances_df.to_csv(path['out_importances'])
    print 'wrote %d importances to %s' % (len(importances_df), path['out_importances'])
    return True


def do_work(control):
    'write fitted models to file system'
    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    def read_csv(path, nrows, parse_dates):
        df = pd.read_csv(
            path,
            nrows=nrows,
            parse_dates=parse_dates,
            index_col=0,
        )
        print 'read %d rows from %s' % (len(df), path)
        return df

    applied_data_science.lower_priority.lower_priority()

    # input files are for a specific cusip
    features = read_csv(
        control.path['in_features'],
        nrows=None,
        parse_dates=['id_effectivedatetime'],
    )
    assert len(features) > 0
    # deconstruct effective datetime into its date components
    features['id_effectiveyear'] = features['id_effectivedatetime'].dt.year
    features['id_effectivemonth'] = features['id_effectivedatetime'].dt.month
    features['id_effectiveday'] = features['id_effectivedatetime'].dt.day
    print 'features.columns'
    pp(features.columns)
    targets = read_csv(
        control.path['in_targets'],
        nrows=None,
        parse_dates=['id_effectivedatetime'],
    )
    print 'targets.columns'
    pp(targets.columns)
    assert len(targets) > 0
    targets['id_effectivedate'] = targets['id_effectivedatetime'].dt.date
    print targets.columns
    # NOTE: The features and targets files are build using independent criteria,
    # so that the indices should not in general be the same
    print 'len(features): %d  len(targets): %d' % (len(features), len(targets))

    result = fit_predict(  # write records to output files
        features=features,
        targets=targets,
        desired_effective_date=Date(from_yyyy_mm_dd=control.arg.effective_date),
        model_specs=control.model_specs,
        test=control.arg.test,
        # already_seen=already_seen,
        random_state=control.random_seed,
        path=control.path,
    )
    if not result:
        print 'actual data records not written, perhaps input is empty'
        # write empty output files so that the build process will know that this program has run

        def touch(path):
            with open(path, 'a'):
                os.utime(path, None)

        touch(control.path['out_importances'])
        touch(control.path['out_predictions'])


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
