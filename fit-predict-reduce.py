'''reduce all the fit-predict output into a single large CSV file with all predictions

INVOCATION
  python fit-predict-reduce.py {ticker} {cusip} [--test] [--trace]

EXAMPLES OF INVOCATIONS
 python fit-predict.py orcl 68389XAS4

INPUTS
 WORKING/fit-predict/{ticker}-{cusip}.csv  # TODO: change to .pickle, once fit-predict is fixed

OUTPUTS
 WORKING/fit-predict-reduce/{ticker}-{cusip}-actuals-predictions.pickle
 TODO: also produce an analysis of the importances
'''

from __future__ import division

import argparse
import cPickle as pickle
import os
import pdb
from pprint import pprint
import random
import sys

import arg_type
from Bunch import Bunch
import dirutility
from FitPredictOutput import FitPredictOutput
from Logger import Logger
from lower_priority import lower_priority
import pickle_utilities
import seven
import seven.path
from Timer import Timer


class Doit(object):
    def __init__(self, ticker, cusip, test=False, me='fit-predict-reduce', old=False):
        self.ticker = ticker
        self.cusip = cusip
        self.me = me
        self.test = test
        # define directories
        working = seven.path.working()
        out_dir = os.path.join(working, me + ('-test' if test else ''))
        # path to files abd durectirs
        if old:
            self.in_file = os.path.join(working, 'fit-predict', '%s-%s-predictions-old.csv' % (ticker, cusip))
        else:
            self.in_file = os.path.join(working, 'fit-predict', '%s-%s.pickle' % (ticker, cusip))
        self.out_dir = out_dir
        self.out_loss = os.path.join(working, 'fit-predict-reduce', '%s-%s-loss.pickle' % (ticker, cusip))
        self.out_log = os.path.join(out_dir, '0log.txt')
        # used by Doit tasks
        self.actions = [
            'python %s.py %s' % (me, ticker)
        ]
        self.targets = [
            self.out_loss,
            self.out_log,
        ]
        self.file_dep = [
            self.me + '.py',
            self.in_file,
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
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--old', action='store_true')  # TODO: remove this arg, once debugged
    arg = parser.parse_args(argv[1:])
    arg.me = parser.prog.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    doit = Doit(arg.ticker, arg.cusip, old=arg.old)
    dirutility.assure_exists(doit.out_dir)

    return Bunch(
        arg=arg,
        doit=doit,
        random_seed=random_seed,
        timer=Timer(),
    )


class ProcessObject(object):
    def __init__(self):
        self.count = 0
        self.result = {}

    def process(self, obj):
        'tabulate accurate by model_spec'
        verbose = False
        self.count += 1
        if verbose or self.count % 1000 == 1:
            print self.count, obj.query_index, obj.model_spec, obj.trade_type, obj.predicted_value, obj.actual_value
        diff = obj.predicted_value - obj.actual_value
        loss = diff * diff
        self.result[obj.model_spec] = loss


def on_EOFError(e):
    print 'EOFError', e
    return


def on_ValueError(e):
    pdb.set_trace()
    print 'ValueError', e
    if e.args[0] == 'insecure string pickle':
        return
    else:
        raise e


def do_work(control):
    'create csv file that summarizes all actual and predicted prices'
    # BODY STARTS HERE
    # determine training and testing transactions
    pdb.set_trace()
    lower_priority()  # try to give priority to interactive tasks

    # read input file record by record
    process_object = ProcessObject()
    pickle_utilities.unpickle_file(
        path=control.doit.in_file,
        process_unpickled_object=process_object.process,
        on_EOFError=on_EOFError,
        on_ValueError=on_ValueError,
    )
    print 'processed %d results' % process_object.count
    with open(control.doit.out_loss, 'w') as f:
        pickle.dump(process_object.result, f)
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
        FitPredictOutput()

    main(sys.argv)
