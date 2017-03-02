'''print report comparing model accuracy

INVOCATION
  python targets.py {ticker} {cusip} [--test] [--trace]
  # TODO: read all the cusips for a ticker in fit-predict-reduction

where
 {ticker}.csv is a CSV file in MidPredictors/data
 --cusip CUSIP means to create the targets only for the specified CUSIP
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python report-comparison-of-model-accuracy.py orcl 68389XAS4

INPUTS
 WORKING/fit-predict-reduce/{ticker}-{cusip}-loss.pickle

OUTPUTS
 WORKING/ME/report-{ticker}-{cusip}.txt
 WORKING/tagets/0log-{ticker}.txt

'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import numpy as np
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
import pickle_utilities
from ReportColumns import ReportColumns
import seven
import seven.models
import seven.path
import seven.reports
from Timer import Timer


class Doit(object):
    def __init__(self, ticker, cusip, test=False, me='report-comparsion-of-model-accuracy.py'):
        self.ticker = ticker
        self.cusip = cusip
        self.me = me
        self.test = test
        # define directories
        working = seven.path.working()
        out_dir = os.path.join(working, self.me + ('-test' if test else ''))
        # read in CUSIPs for the ticker
        # TODO: read the fit-predict-reduct directory to find the CUSIPS that have been fit
        with open(os.path.join(working, 'cusips', ticker + '.pickle'), 'r') as f:
            self.cusips = pickle.load(f).keys()
        # path to files abd durecties
        ticker_cusip = '%s-%s' % (ticker, cusip)

        self.in_file = os.path.join(working, 'fit-predict', ticker_cusip + '.pickle')

        self.out_report = os.path.join(out_dir, 'report-' + ticker_cusip + '.txt')
        self.out_log = os.path.join(out_dir, '0log-' + ticker_cusip + '.txt')

        self.out_dir = out_dir
        # used by Doit tasks
        self.actions = [
            'python %s.py %s %s' % (self.me, ticker, cusip)
        ]
        self.targets = [
            self.out_report,
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
    arg = parser.parse_args(argv[1:])

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


class ProcessObject(object):
    def __init__(self):
        self.count = 0
        self.losses = collections.defaultdict(list)
        self.model_specs = collections.defaultdict(set)

    def process(self, obj):
        'accumulate losses by model_spec.model'
        diff = obj.predicted_value - obj.actual_value
        loss = diff * diff

        name = obj.model_spec.name
        self.losses[name].append(loss)
        self.model_specs[name].add(obj.model_spec)


def on_EOFError(e):
    print 'EOFError', e
    return


def on_ValueError(e):
    print 'ValueError', e
    if e.args[0] == 'insecure string pickle':
        return
    else:
        raise e


def do_work(control):
    'write order imbalance for each trade in the input file'
    # extract info from the fit-predict objects
    process_object = ProcessObject()
    pickle_utilities.unpickle_file(
        path=control.doit.in_file,
        process_unpickled_object=process_object.process,
        on_EOFError=on_EOFError,
        on_ValueError=on_ValueError,
    )

    # process_object.result: Dict[model_spec.name, List[loss]]
    report = ReportColumns(seven.reports.columns(
        'model_name',
        'n_hp_sets',
        'n_samples',
        'min_loss',
        'mean_loss',
        'median_loss',
        'max_loss',
        'std_loss',
        ))
    report.append_header('Comparison of Model Accuracy')
    report.append_header('For ticker %s cusips %s' % (control.arg.ticker, control.arg.cusip))
    report.append_header('Summary Statistics Across All Trades')
    report.append_header(' ')

    pdb.set_trace()
    for model_spec_name, losses_list in process_object.losses.iteritems():
        print model_spec_name, len(losses_list)
        losses = np.array(losses_list)
        report.append_detail(
            model_name=model_spec_name,
            n_hp_sets=len(process_object.model_specs[model_spec_name]),
            n_samples=len(losses_list),
            min_loss=np.min(losses),
            mean_loss=np.mean(losses),
            median_loss=np.median(losses),
            max_loss=np.max(losses),
            std_loss=np.std(losses),
        )
    pdb.set_trace()
    report.write(control.doit.out_report)
    return


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
        # avoid pyflake warnings for debugging tools
        pdb.set_trace()
        pprint()
        FitPredictOutput

    main(sys.argv)
