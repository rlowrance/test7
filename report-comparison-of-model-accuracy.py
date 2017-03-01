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
import cPickle as pickle
import os
import pdb
from pprint import pprint
import random
import sys

import arg_type
from Bunch import Bunch
import dirutility
from Logger import Logger
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

        self.in_reduction = os.path.join(working, 'fit-predict-reduce', ticker_cusip + '-loss.pickle')

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
            self.in_reduction,
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


def do_work(control):
    'write order imbalance for each trade in the input file'
    # BODY STARTS HERE
    with open(control.doit.in_reduction, 'r') as f:
        losses = pickle.load(f)  # a Dict[ModelSpec], Float]
    report = ReportColumns(seven.reports.columns('ticker', 'cusip', 'model_spec', 'loss'))
    report.append_header('Comparison of Model Accuracy')
    report.append_header('TODO: which trades?')
    report.append_header('TODO: fix reduction then report min, mean, max loss')
    report.append_header(' ')
    for model_spec, loss in losses.iteritems():
        report.append_detail(
            ticker=control.arg.ticker,
            cusip=control.arg.cusip,
            model_spec=model_spec,
            loss=loss,
        )
    report.write(control.doit.out_report)
    pdb.set_trace()
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

    main(sys.argv)
