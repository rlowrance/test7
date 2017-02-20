'''fit and predict all models on one CUSIP feature file

INVOCATION
  python fit-predict.py {cusip}.csv {--test} {--trace}

where
 WORKING/features/{cusip}.csv  is a CSV file, one sample per row, ordered by column datetime
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python fit-predict.py 68389XAN5.csv

INPUTS
 WORKING/features/{cusip}.csv

OUTPUTS
 WORKING/fit-predict/{cusip}-predictions.pickle  file containing predictions for each fitted model
  The file is a sequence of records, each record a tuple:
  (model_spec, original_print_file_index,
   actual_B, predicted_B, actual_D, predicted_D, actual_S, predicted_S,
  )
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


def make_control(argv):
    'return a Bunch'

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('cusipfile', type=arg_type.cusipfile)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])
    arg.me = parser.prog.split('.')[0]
    arg.cusip = arg.cusipfile.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    dir_working = seven.path.working()
    if arg.test:
        dir_out = os.path.join(dir_working, arg.me + '-test')
    else:
        dir_out = os.path.join(dir_working, arg.me)
    dirutility.assure_exists(dir_out)

    return Bunch(
        arg=arg,
        path_in_training_samples=os.path.join(dir_working, 'features', arg.cusip + '.csv'),
        path_out_predictions=os.path.join(dir_out, arg.cusip + '-predictions.pickle'),
        path_out_log=os.path.join(dir_out, '0log.txt'),
        random_seed=random_seed,
        timer=Timer(),
    )


def fit_predict(pickler, training_samples_all, test):
    'append to pickler file, a prediction (when possible) for each sample'
    counter = 1
    max_counter = len(training_samples_all) * len(models.all_model_specs) * len(models.trade_types)
    for query_index in training_samples_all.index:
        # train on samples before the query transaction
        mask_training = training_samples_all.datetime < training_samples_all.loc[query_index].datetime
        training_samples = training_samples_all.loc[mask_training]
        mask_query = training_samples_all.index == query_index
        query_samples = training_samples_all.loc[mask_query]
        assert len(query_samples) == 1
        print 'query_index %d of %d on %d training samples' % (
            query_index,
            len(training_samples_all),
            len(training_samples),
            )
        if len(training_samples) == 0:
            print ' skipping, as no training samples'
            continue

        for model_spec in models.all_model_specs:
            for trade_type in models.trade_types:
                fitted, importances = models.fit(model_spec, training_samples, trade_type)
                predicted_raw = models.predict(
                    fitted,
                    model_spec,
                    query_samples,
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
            nrows=10 if control.arg.test else None,
            low_memory=False
        )
        print 'read %d samples from file %s' % (len(df), path)
        return df

    # reduce process priority, to try to keep the system responsive if multiple jobs are run
    lower_priority()

    training_samples_all = read_csv(control.path_in_training_samples)
    with open(control.path_out_predictions, 'w') as f:
        pickler = pickle.Pickler(f)
        fit_predict(pickler, training_samples_all, control.arg.test)  # mutate file accessed via pickler
    return None


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path_out_log)  # now print statements also write to the log file
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
