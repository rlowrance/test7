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
import pandas as pd
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


def fit_predict(pickler, features, targets, test):
    'append to pickler file, a prediction (when possible) for each sample'
    pdb.set_trace()
    counter = 1
    max_counter = len(features) * len(models.all_model_specs) * len(models.trade_types)
    for query_index in features.index:
        # train on samples before the query transaction
        pdb.set_trace()
        # the features DataFrame is not guaranteed to be sorted by effectivedatetime
        mask_training = features.effectivedatetime < features.loc[query_index].effectivedatetime
        training_samples = features.loc[mask_training]
        print 'query_index %d of %d on %d training samples' % (
            query_index,
            len(features),
            len(training_samples),
            )
        if len(training_samples) == 0:
            print ' skipping, as no training samples'
            continue

        for model_spec in models.all_model_specs:
            # fit one model for each targets
            # targets are the future price of each trade_type
            for trade_type in models.trade_types:
                fitted, importances = models.fit(model_spec, features, trade_type)
                predicted_raw = models.predict(
                    fitted,
                    model_spec,
                    features.loc[query_index],
                    trade_type,
                )
                predicted_vector = (
                    predicted_raw if model_spec.transform_y is None else
                    predicted_raw.exp() if model_spec.transform_y == 'log' else
                    None  # an internal error
                )
                if predicted_vector is None:
                    print 'bad prediction transformation', model_spec.transform_y
                    pdb.set_trace()
                assert len(predicted_vector) == 1  # because there is one query sample
                predicted = predicted_vector[0]
                if counter % 100 == 1:
                    print counter, max_counter, trade_type, predicted, importances
                counter += 1
                # write to file referenced by pickler
                # NOTE: if disk space becomes an issue, the model_spec values could
                # be written to a common file and referenced by ID here
                obj = (query_index, model_spec, trade_type, predicted, importances)
                pickler.dump(obj)
                gc.collect()  # try to keep memory usage roughly constant
                if test and counter > 100:
                    return


def do_work(control):
    'write fitted models to file system'

    def read_csv(path):
        df = pd.read_csv(
            path,
            index_col=0,
            nrows=10 if control.arg.test else None,
            low_memory=False
        )
        print 'read %d samples from file %s' % (len(df), path)
        return df

    # reduce process priority, to try to keep the system responsive if multiple jobs are run
    lower_priority()

    features = read_csv(control.doit.in_features)
    targets = read_csv(control.doit.in_targets)
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
