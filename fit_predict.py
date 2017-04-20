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

See build.py for input and output files.BaseException

The main output is in out_file. It is a pickled file with each record a FitPredictOutput instance

The program reads the output file as input, in order to implement a checkpoint-restart.
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import datetime
import gc
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
from seven.FitPredictOutput import Id, Payload, Record
from seven.models import ModelNaive, ModelElasticNet, ModelRandomForests
from seven import HpGrids

import build


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


def make_prediction(
    training_features=None,
    training_targets=None,
    predicted_feature=None,
    model_spec=None,
    random_state=None,
    query_sample=None,
):
    'return (error, prediction, importances)'
    model_constructor = (
        ModelNaive if model_spec.name == 'n' else
        ModelElasticNet if model_spec.name == 'en' else
        ModelRandomForests if model_spec.name == 'rf' else
        None
    )
    if model_constructor is None:
        print 'error: bad model_spec.name %s' % model_spec.name
        pdb.set_trace()
    try:
        model = model_constructor(model_spec, predicted_feature, random_state)
        model.fit(training_features, training_targets)
        predicted = model.predict(query_sample)
        assert len(predicted) == 1
        return (None, predicted[0], model.importances)
    except Exception as e:
        print 'make_prediction exception:', e
        pdb.set_trace()
        return (e, None, None)


def fit_predict(
    pickler=None,
    features=None,
    targets=None,
    desired_effective_date=None,
    model_specs=None,
    test=None,
    already_seen=None,
    random_state=None,
):
    'append to pickler file, a prediction (when possible) for each target sample on the effectve_date'
    # NOTE: already_seen is ignored
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
        id = Id(
            query_index=query_index,
            model_spec=model_spec,
            predicted_feature_name=predicted_feature_name,
        )
        result = id in already_seen
        return result

    skipped = collections.Counter()
    relevant_targets = targets.loc[targets.id_effectivedate == desired_effective_date.value]
    print 'found %d trades on the requested date' % len(relevant_targets)
    if len(relevant_targets) == 0:
        msg = 'no targets for desired effective date %s' % desired_effective_date
        print msg
        skipped[msg] += 1
        return 0

    n_written = 0
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
            fit_predict_output = Record(
                id=Id(
                    query_index=fit_predict_result.query_index,
                    model_spec=fit_predict_result.model_spec,
                    predicted_feature_name=fit_predict_result.predicted_feature_name,
                ),
                payload=Payload(
                    predicted_feature_value=fit_predict_result.prediction,
                    actual_value=fit_predict_result.predicted_feature_value,
                    importances=fit_predict_result.fitted_model.importances,
                    n_training_samples=fit_predict_result.n_training_samples,
                ),
            )
            pickler.dump(fit_predict_output)
            n_written += 1
            print 'appended new record # %d %s %s %s' % (
                n_written,
                fit_predict_output.id.query_index,
                fit_predict_output.id.model_spec,
                fit_predict_output.id.predicted_feature_name,
            )
            gc.collect()
        else:
            print 'skipped', fit_predict_result
            skipped[fit_predict_result] += 1

    print 'aopended %d predictions' % n_written
    print 'skipped some features; reasons and counts:'
    n_already_seen = 0
    for k in sorted(skipped.keys()):
        if k.startswith('already seen:'):
            n_already_seen += 1
        else:
            print '%40s: %s' % (k, skipped[k])
    print 'n_already_seen:', n_already_seen
    return n_written


def do_work(control):
    'write fitted models to file system'
    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    def read_csv(path, nrows, parse_dates):
        df = pd.read_csv(
            path,
            nrows=nrows,
            parse_dates=parse_dates,
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
    print features.columns
    targets = read_csv(
        control.path['in_targets'],
        nrows=None,
        parse_dates=['id_effectivedatetime'],
    )
    assert len(targets) > 0
    targets['id_effectivedate'] = targets['id_effectivedatetime'].dt.date
    print targets.columns
    # NOTE: The features and targets files are build using independent criteria,
    # so that the indices should not in general be the same
    print 'len(features): %d  len(targets): %d' % (len(features), len(targets))

    # read output file and determine records in it
    class ProcessObject(object):
        def __init__(self):
            self.seen = set()

        def process(self, obj):
            id = obj.id
            assert id not in self.seen
            self.seen.add(id)

    def on_EOFError(e):
        print e
        return

    def on_ValueError(e):
        print e
        if e.args[0] == 'insecure string pickle':
            # possibly caused by the file being open for writing in another process
            # possibly caused by killing of the writing process
            return  # treat like EOFError
        else:
            raise e

    process_object = ProcessObject()
    applied_data_science.pickle_utilities.unpickle_file(
        path=control.path['out_file'],
        process_unpickled_object=process_object.process,
        on_EOFError=on_EOFError,
        on_ValueError=on_ValueError,
        )
    already_seen = process_object.seen
    print 'have already seen %d results' % len(already_seen)

    # append new records to the pickle file
    with open(control.path['out_file'], 'a') as f:
        pickler = pickle.Pickler(f)
        count_appended = fit_predict(  # write records to out_file
            pickler=pickler,
            features=features,
            targets=targets,
            desired_effective_date=Date(from_yyyy_mm_dd=control.arg.effective_date),
            model_specs=control.model_specs,
            test=control.arg.test,
            already_seen=already_seen,
            random_state=control.random_seed,
        )
        print 'appended %d records' % count_appended
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
