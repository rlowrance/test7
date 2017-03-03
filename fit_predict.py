'''fit and predict all models on one CUSIP feature file

INVOCATION
  python fit_predict.py {ticker} {cusip} {hpset} {effective_date} {--test} {--trace}
where
 ticker is the ticker symbol (ex: orcl)
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 hpset in {gridN} defines the hyperparameter set
 effective_date: YYYY-MM-DD is the date of the trade

EXAMPLES OF INVOCATION
 python fit-predict.py orcl 68389XAS4 grid2 2016-03-15

INPUTS
 WORKING/features/{cusip}.csv

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
import cPickle as pickle
import gc
import os
import pdb
from pprint import pprint
import random
import sys

from Bunch import Bunch
import dirutility
from Logger import Logger
from lower_priority import lower_priority
import pickle_utilities
import seven
import seven.path
from seven import arg_type
from seven.FitPredictOutput import FitPredictOutput
from seven import models
from Timer import Timer


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
            '%s-%s-%s-%s-%s' % (me, ticker, cusip, hpset, effective_date)
        )
        # path to files abd durecties
        in_filename = '%s-%s.csv' % (ticker, cusip)
        self.in_features = os.path.join(working, 'features', in_filename)
        self.in_targets = os.path.join(working, 'targets', in_filename)

        # FIXME: write to one pickle file that contains both the importances and predictins
        # record format is FitPredictOutput
        self.out_file = os.path.join(out_dir, 'fit-predict-output.pickle')
        self.out_log = os.path.join(out_dir, '0log.txt')

        self.out_dir = out_dir
        # used by Doit tasks
        self.actions = [
            'python %s.py %s %s' % (me, ticker, cusip)
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
    dirutility.assure_exists(doit.out_dir)

    return Bunch(
        arg=arg,
        doit=doit,
        random_seed=random_seed,
        timer=Timer(),
    )


def fit_predict(pickler, features, targets, test, already_seen):
    'append to pickler file, a prediction (when possible) for each sample'
    def target_value(query_index, trade_type):
        row = targets.loc[query_index]
        result = (
            row.next_price_B if trade_type == 'B' else
            row.next_price_D if trade_type == 'D' else
            row.next_price_S if trade_type == 'S' else
            None
        )
        if result is None:
            raise ValueError('unknown trade_type: %s' % trade_type)
        return result

    pdb.set_trace()
    counter = 1
    skipped = 0
    max_counter = len(features) * len(models.all_model_specs) * len(models.trade_types)
    for query_index in features.index:
        if query_index not in targets.index:
            print 'query index %s is in feature, but not targets: skipped' % query_index
            skipped += 1
            continue
        # the features DataFrame is not guaranteed to be sorted by effectivedatetime
        # select for training all the trades that occurred before the query trade
        # Train on all transactions at or before the current trade's date and time
        mask_training = features.effectivedatetime <= features.loc[query_index].effectivedatetime
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
                print '', model_spec, trade_type
                if (query_index, model_spec, trade_type) in already_seen:
                    print 'skipping as already seens:', query_index, model_spec, trade_type
                    continue
                fitted, importances = models.fit(
                    model_spec,
                    training_features,
                    training_targets,
                    trade_type,
                    )
                predicted = models.predict(
                    fitted,
                    model_spec,
                    features.loc[[query_index]],  # return DataFrame, not Series
                    trade_type,
                )
                if predicted is None:
                    # For now, this cannot happen
                    # Later we may allow methods to refuse to predict
                    # Perhaps they will raise an exception when that happens
                    print 'bad predicted value', model_spec.transform_y
                    pdb.set_trace()
                assert len(predicted) == 1  # because there is one query sample
                if counter % 100 == 1:
                    print counter, max_counter, trade_type, predicted, importances
                # write to file referenced by pickler
                # NOTE: if disk space becomes an issue, the model_spec values could
                # be written to a common file and referenced by ID here
                obj = FitPredictOutput(
                    query_index=query_index,
                    model_spec=model_spec,
                    trade_type=trade_type,
                    predicted_value=predicted[0],
                    actual_value=target_value(query_index, trade_type),
                    importances=importances,
                    n_training_samples=len(training_features),
                )
                pickler.dump(obj)
                # Keep memory usage roughly constant
                # This helps when we run multiple fit-predict instances on one system
                gc.collect()
                if test and counter > 10:
                    return
                counter += 1
    print 'wrote %d predictions' % counter
    print 'skiped %d features records because target values were not available' % skipped


def do_work(control):
    'write fitted models to file system'
    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    lower_priority()

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
    assert len(targets) > 0
    # NOTE: The features and targets files are build using independent criteria,
    # so that the indices should not in general be the same
    print 'len(features): %d  len(targets): %d' % (len(features), len(targets))

    # read output file and determine records in it
    class ProcessObject(object):
        def __init__(self):
            self.seen = set()

        def process(self, obj):
            self.seen.add((obj.query_index, obj.model_spec, obj.trade_type))

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
    pickle_utilities.unpickle_file(
        path=control.doit.out_file,
        process_unpickled_object=process_object.process,
        on_EOFError=on_EOFError,
        on_ValueError=on_ValueError,
        )
    already_seen = process_object.seen
    print 'have already seen %d results' % len(already_seen)

    with open(control.doit.out_file, 'w') as f:
        pickler = pickle.Pickler(f)
        fit_predict(pickler, features, targets, control.arg.test, already_seen)  # mutate file accessed via pickler
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
