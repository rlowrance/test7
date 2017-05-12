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
import os
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
    'write CSV files with predictions and importances for trades on specified date'
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

    def select_rows_before(df, effectivedatetime):
        'return DataFrame with only those rows before the specified datatime'
        pdb.set_trace()
        mask = df['id_effectivedatetime'] < effectivedatetime
        selected = df.loc[mask]
        return selected

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


def do_work(control):
    'write fitted models to file system'
    def read_csv(path, nrows, parse_dates):
        df = pd.read_csv(
            path,
            nrows=nrows,
            parse_dates=parse_dates,
            index_col=0,
        )
        print 'read %d rows from %s' % (len(df), path)
        return df

    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
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
        parse_dates=['info_subsequent_effectivedatetime', 'info_this_effectivedatetime'],
    )
    print 'targets.columns'
    pp(targets.columns)
    assert len(targets) > 0
    targets['info_this_effectivedate'] = targets['info_this_effectivedatetime'].dt.date
    print targets.columns

    # NOTE: The features and targets files are build using independent criteria,
    # so that the indices should not in general be the same
    print 'len(features): %d  len(targets): %d' % (len(features), len(targets))
    # keep only the features and targets that have common indices
    # common_indices = features.index.copy().intersection(targets.index)
    # common_features = features.loc[common_indices]
    # common_targets = targets.loc[common_indices]
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
