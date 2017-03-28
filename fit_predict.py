'''fit and predict all models on one CUSIP feature file

TODO: use a class to capture the logic

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

INPUTS
 WORKING/features-{ticker}/{cusip}.csv
 WORKING/targets-{ticker}/{cusip}.csv

INPUTS and OUTPUTS
 WORKING/features/fit_predict-{ticker}-{cusip}-{hpset}-{effective_date}/fit-predict-output.pickle
   each record is a FitPredictOutput isinstance
   read to implement checkpoint restart

OUTPUTS
 WORKING/fit-predict-{ticker}-{cusip}-{hpset}-{effective_date}/0log.txt
where
  model_spec is a string specifying both the model family (naive, en, rf) and its hyperparameters
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import datetime
import gc
import os
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

import seven
import seven.path
from seven import arg_type
from seven.FitPredictOutput import Id, Payload, Record
from seven import models
from seven.models import ModelNaive, ModelElasticNet, ModelRandomForests
from seven import HpGrids


class Doit(object):
    def __init__(self, ticker, cusip, hpset, effective_date, test=False, me='fit_predict'):
        self.ticker = ticker
        self.cusip = cusip
        self.hpset = hpset
        self.effective_date = effective_date
        self.test = test
        self.me = me
        # define directories
        working = seven.path.working()
        out_dir = os.path.join(
            working,
            '%s-%s-%s-%s-%s' % (me, ticker, cusip, hpset, effective_date) + ('-test' if test else '')
        )
        # path to files abd durecties
        in_filename = '%s.csv' % cusip
        self.in_features = os.path.join(working, 'features-%s' % ticker, in_filename)
        self.in_targets = os.path.join(working, 'targets-%s' % ticker, in_filename)

        self.out_file = os.path.join(out_dir, 'output.pickle')
        self.out_log = os.path.join(out_dir, '0log.txt')

        self.out_dir = out_dir
        # used by Doit tasks
        self.actions = [
            'python %s.py %s %s %s %s' % (me, ticker, cusip, hpset, effective_date)
        ]
        self.targets = [
            self.out_file,
            self.out_log,
        ]
        self.file_dep = [
            self.me + '.py',
            self.in_features,
            self.in_targets,
        ]

    def __str__(self):
        for k, v in self.__dict__.iteritems():
            print 'doit.%s = %s' % (k, v)
        return self.__repr__()


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

    doit = Doit(arg.ticker, arg.cusip, arg.hpset, arg.effective_date, test=arg.test)
    applied_data_science.dirutility.assure_exists(doit.out_dir)
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
        doit=doit,
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
    # return count of records appended to pickler file

    n_appended = 0
    skipped = collections.Counter()
    count_by_date = collections.Counter()
    zero_error = collections.Counter()
    # TODO: coordinate already_seen with code in do_work

    relevant_targets = targets.loc[targets.id_effectivedate == desired_effective_date.value]
    print 'found %d trades on the requested date' % len(relevant_targets)
    if len(relevant_targets) == 0:
        msg = 'no targets for desired effective date %s' % desired_effective_date
        print msg
        skipped[msg] += 1
        return 0
    for target_counter, target_index in enumerate(relevant_targets.index):
        target_sample = relevant_targets.loc[target_index]  # a Series
        if target_index not in features.index:
            msg = 'target index %s not in features' % target_index
            print msg
            skipped[msg] += 1
            continue
        mask_training = features.id_effectivedatetime < target_sample.id_effectivedatetime
        training_features = features.loc[mask_training]
        training_targets = targets.loc[training_features.index]  # pick same trades as in training_features
        query_sample = features.loc[[target_index]]  # we need a dataframe, not a series
        assert len(training_features) == len(training_targets)
        for predicted_feature in targets.columns:
            if predicted_feature.startswith('id_'):  # TODO: do only future prices, not direction
                continue
            if '_spread_' in predicted_feature:
                # feature names skipped: {tradetype}_spread_{decreased|increased}
                msg = 'for now, skipping feature %s' % predicted_feature
                print msg
                skipped[msg] += 1
                continue
            for model_spec_index, model_spec in enumerate(model_specs):
                prediction_id = Id(
                    target_index=target_index,
                    model_spec=model_spec,
                    predicted_feature=predicted_feature,
                )
                if prediction_id in already_seen:
                    msg = 'skipping %s as already seen' % prediction_id
                    print msg
                    skipped[msg] += 1
                    continue
                error, prediction, importances = make_prediction(
                    training_features=training_features,
                    training_targets=training_targets,
                    predicted_feature=predicted_feature,
                    model_spec=model_spec,
                    random_state=random_state,
                    query_sample=query_sample,
                )
                if error is not None:
                    print 'prediction error:', error
                    skipped[error] += 1
                    continue
                fit_predict_output = Record(
                    id=prediction_id,
                    payload=Payload(
                        predicted_value=prediction,
                        actual_value=target_sample[predicted_feature],
                        importances=importances,
                        n_training_samples=len(training_features),
                    ),
                )
                pickler.dump(fit_predict_output)
                n_appended += 1
                print 'trade %d of %d' % (target_counter + 1, len(relevant_targets)),
                print predicted_feature,
                print n_appended,
                print model_spec
                gc.collect()  # keep memory usage about constant
    print 'wrote %d predictions' % n_appended
    print 'skipped some features; reasons and counts:'
    for k in sorted(skipped.keys()):
        print '%40s: %s' % (k, skipped[k])
    print 'count of trades by date'
    for date in sorted(count_by_date.keys()):
        print date, count_by_date[date]
    print 'zero errors'
    for description, count in zero_error.iteritems():
        print 'zero error', description, count
    return n_appended


def do_work(control):
    'write fitted models to file system'
    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    # input files are for a specific cusip
    features = models.read_csv(
        control.doit.in_features,
        nrows=None,
        parse_dates=['id_effectivedatetime'],
    )
    assert len(features) > 0
    # deconstruct effective datetime into its date components
    features['id_effectiveyear'] = features['id_effectivedatetime'].dt.year
    features['id_effectivemonth'] = features['id_effectivedatetime'].dt.month
    features['id_effectiveday'] = features['id_effectivedatetime'].dt.day
    print features.columns
    targets = models.read_csv(
        control.doit.in_targets,
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
        path=control.doit.out_file,
        process_unpickled_object=process_object.process,
        on_EOFError=on_EOFError,
        on_ValueError=on_ValueError,
        )
    already_seen = process_object.seen
    print 'have already seen %d results' % len(already_seen)

    with open(control.doit.out_file, 'w') as f:
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
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
        datetime

    main(sys.argv)
