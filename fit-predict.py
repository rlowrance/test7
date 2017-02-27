'''fit and predict all models on one CUSIP feature file

INVOCATION
  python fit-predict.py {ticker} {cusip} {--test} {--trace}

where
 WORKING/features/{cusip}.csv  is a CSV file, one sample per row, ordered by column datetime
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python fit-predict.py orcl 68389XAS4

INPUTS
 WORKING/features/{cusip}.csv

OUTPUTS
 WORKING/fit-predict/{ticker}-{cusip}-predictions.pickle  file containing predictions for each fitted model
  The file is a sequence of records, each record a tuple:
  (model_spec, original_print_file_index,
   actual_B, predicted_B, actual_D, predicted_D, actual_S, predicted_S,
  )
 WORKING/fit-predict/{ticker}-{cusip}-importances.pickle
where
  model_spec is a string specifying both the model family (naive, en, rf) and its hyperparameters
'''

from __future__ import division

import argparse
import cPickle as pickle
import gc
import os
import pdb
from pprint import pprint
import random
import sys

import arg_type
from Bunch import Bunch
import dirutility
from Logger import Logger
from lower_priority import lower_priority
import seven
import seven.path
from seven import models
from Timer import Timer


class Doit(object):
    def __init__(self, ticker, cusip, test=False, me='fit-predict'):
        self.ticker = ticker
        self.cusip = cusip
        self.me = me
        self.test = test
        # define directories
        working = seven.path.working()
        out_dir = os.path.join(working, me + ('-test' if test else ''))
        # read in CUSIPs for the ticker
        with open(os.path.join(working, 'cusips', ticker + '.pickle'), 'r') as f:
            self.cusips = pickle.load(f).keys()
        # path to files abd durecties
        in_filename = '%s-%s.csv' % (ticker, cusip)
        self.in_features = os.path.join(working, 'features', in_filename)
        self.in_targets = os.path.join(working, 'targets', in_filename)

        self.out_importances = os.path.join(out_dir, '%s-%s-importances.csv' % (ticker, cusip))
        self.out_predictions = os.path.join(out_dir, '%s-%s-predictions.csv' % (ticker, cusip))
        self.out_log = os.path.join(out_dir, '0log.txt')

        self.out_dir = out_dir
        # used by Doit tasks
        self.actions = [
            'python %s.py %s %s' % (me, ticker, cusip)
        ]
        self.targets = [
            self.out_importances,
            self.out_predictions,
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
    parser.add_argument('ticker')
    parser.add_argument('cusip', type=arg_type.cusip)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])
    arg.me = parser.prog.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    doit = Doit(arg.ticker, arg.cusip)
    dirutility.assure_exists(doit.out_dir)

    return Bunch(
        arg=arg,
        doit=doit,
        random_seed=random_seed,
        timer=Timer(),
    )


class FitPredictOutput(object):
    def __init__(
        self,
        query_index=None,
        model_spec=None,
        trade_type=None,
        predicted_value=None,
        actual_value=None,
        importances=None
    ):
        def test(feature_name):
            pdb.set_trace()
            value = self.__dict__[feature_name]
            if value is None:
                raise ValueError('%s cannot be None' % feature_name)
            else:
                return value

        self.query_index = test('query_index')
        self.model_spec = test('model_spec')
        self.trade_type = test('trade_type')
        self.predicted_value = test('predicted_value')
        self.actual_value = test('actual_value')
        self.importances = importances  # will be None, when the method doesn't provide importances


def fit_predict(pickler, features, targets, test):
    'append to pickler file, a prediction (when possible) for each sample'
    def target_value(query_index, trade_type):
        pdb.set_trace()
        row = targets.loc[query_index]
        result = (
            row.price_B if trade_type == 'B' else
            row.price_D if trade_type == 'D' else
            row.price_S if trade_type == 'S' else
            None
        )
        if result is None:
            raise ValueError('unknown trade_type: %s' % trade_type)
        return result

    pdb.set_trace()
    counter = 1
    max_counter = len(features) * len(models.all_model_specs) * len(models.trade_types)
    for query_index in features.index:
        pdb.set_trace()
        # the features DataFrame is not guaranteed to be sorted by effectivedatetime
        # select for training all the trades that occurred before the query trade
        mask_training = features.effectivedatetime < features.loc[query_index].effectivedatetime
        pdb.set_trace()
        training_features = features.loc[mask_training]
        training_targets = targets.loc[training_features.index]
        assert len(training_features) == len(training_targets)
        print 'query_index %d; %d of %d on %d training samples' % (
            query_index,
            counter,
            len(features),
            len(training_features),
            )
        if len(training_features) == 0:
            print ' skipping, as no training samples'
            continue

        for model_spec in models.all_model_specs:
            # fit the specified model to the training features and targets
            # targets are the future price of each trade_type
            for trade_type in models.trade_types:
                pdb.set_trace()
                fitted, importances = models.fit(
                    model_spec,
                    training_features,
                    training_targets,
                    trade_type,
                    )
                predicted = models.predict(
                    fitted,
                    model_spec,
                    features.loc[query_index],
                    trade_type,
                )
                if predicted is None:
                    # For now, this cannot happen
                    # Later we may allow methods to refuse to predict
                    # Perhaps they will raise an exception when that happens
                    print 'bad prediction transformation', model_spec.transform_y
                    pdb.set_trace()
                assert len(predicted) == 1  # because there is one query sample
                predicted_value = predicted[0]
                if counter % 100 == 1:
                    print counter, max_counter, trade_type, predicted, importances
                # write to file referenced by pickler
                # NOTE: if disk space becomes an issue, the model_spec values could
                # be written to a common file and referenced by ID here
                obj = FitPredictOutput(
                    query_index=query_index,
                    model_spec=model_spec,
                    trade_type=trade_type,
                    predicted_value=predicted_value,
                    actual_value=target_value(query_index, trade_type),
                    importances=importances
                )
                pickler.dump(obj)
                # Keep memory usage roughly constant
                # This helps when we run multiple fit-predict instances on one system
                gc.collect()
                if test and counter > 10:
                    return
                counter += 1


def do_work(control):
    'write fitted models to file system'
    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    lower_priority()

    pdb.set_trace()
    # input files are for a specific cusip
    features = models.read_csv(
        control.doit.in_features,
        nrows=10 if control.arg.test else None,
    )
    assert len(features) > 0
    targets = models.read_csv(
        control.doit.in_targets,
        nrows=10 if control.arg.test else None,
    )
    # validate that the indexes are the same
    assert len(features) == len(targets)
    for index in features.index:
        if index not in targets.index:
            print 'targets is missing index %s' % index
            pdb.set_trace()
    with open(control.doit.out_predictions, 'w') as f:
        pickler = pickle.Pickler(f)
        fit_predict(pickler, features, targets, control.arg.test)  # mutate file accessed via pickler
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
    print control
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()

    main(sys.argv)
