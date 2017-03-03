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
 python report-compare-mmodels.py orcl 68389XAS4

INPUTS
 WORKING/fit-predict-reduce/{ticker}-{cusip}-loss.pickle

OUTPUTS
 WORKING/ME/report-{ticker}-{cusip}-accuracy.txt
 WORKING/ME/report-{ticker}-{cusip}-importances.txt
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
    def __init__(self, ticker, cusip, test=False, me='report-compare-models'):
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

        self.out_report_accuracy = os.path.join(out_dir, 'report-accuracy-%s.txt' % ticker_cusip)
        self.out_report_importances = os.path.join(out_dir, 'report-importances-%s.txt' % ticker_cusip)
        self.out_log = os.path.join(out_dir, '0log-' + ticker_cusip + '.txt')

        self.out_dir = out_dir
        # used by Doit tasks
        self.actions = [
            'python %s.py %s %s' % (self.me, ticker, cusip)
        ]
        self.targets = [
            self.out_report_accuracy,
            self.out_report_importances,
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

    doit = Doit(arg.ticker, arg.cusip, test=arg.test)
    dirutility.assure_exists(doit.out_dir)

    return Bunch(
        arg=arg,
        doit=doit,
        random_seed=random_seed,
        timer=Timer(),
    )


class ProcessObject(object):
    def __init__(self, test):
        self.test = test
        self.count = 0
        self.counts = collections.Counter()
        self.importances_model_spec = collections.defaultdict(list)
        self.losses_name = collections.defaultdict(list)
        self.losses_model_spec = collections.defaultdict(list)
        self.model_specs = set()
        self.model_specs_for_model_name = collections.defaultdict(list)

    def process(self, obj):
        'accumulate losses by model_spec.model'
        self.count += 1
        if self.count % 10000 == 1:
            print 'processing input record number', self.count
        diff = obj.predicted_value - obj.actual_value
        loss = diff * diff

        model_spec = obj.model_spec
        name = model_spec.name
        self.counts[name] += 1
        if self.test:
            print 'process', name, self.counts

        self.importances_model_spec[model_spec].append(obj.importances)
        self.losses_name[name].append(loss)
        self.losses_model_spec[model_spec].append(loss)
        self.model_specs.add(model_spec)
        self.model_specs_for_model_name[name].append(model_spec)
        if len(self.importances_model_spec[model_spec]) == 0:
            print 'zero length model importances'
            print obj.model_spec
            pdb.set_trace()
        if self.test and self.counts['rf'] == 3:
            print 'debugging artificial EOF'
            self.p()
            raise EOFError

    def p(self):
        print 'instance of ProcessObject'
        for k, v in self.counts.iteritems():
            print 'counts', k, v
        for k, vs in self.importances_model_spec.iteritems():
            print 'importances', k
            for v in vs:
                print 'v', v
        for k, v in self.losses_name.iteritems():
            print 'losses', k, v
        for k, v in self.losses_model_spec.iteritems():
            print 'losses_model_spec', k, v
        for model_spec in self.model_specs:
            print 'model_specs', model_spec
        for model_name, model_specs in self.model_specs_for_model_name.iteritems():
            print 'model_name', model_name, 'model_specs', model_specs


def on_EOFError(e):
    print 'EOFError', e
    return


def on_ValueError(e):
    print 'ValueError', e
    if e.args[0] == 'insecure string pickle':
        return
    else:
        raise e


def make_report_importances(process_object, ticker, cusip):
    'return a Report'
    verbose = False

    def make_mean_feature_importance(importances, i):
        'return mean importance of i-th feature'
        importances_i = []
        for importance in importances:
            importances_i.append(importance[i])
        return np.mean(np.array(importances_i))

    report = ReportColumns(seven.reports.columns(
        'model_spec',
        'mean_loss',
        'feature_name',
        'mean_feature_importance',
        ))
    report.append_header('Mean Feature Importances for Random Forests Model')
    report.append_header('For Ticker %s Cusip %s' % (ticker, cusip))
    report.append_header(' ')

    for model_spec in seven.models.all_model_specs:
        if verbose:
            print model_spec
        if model_spec.name != 'rf':
            continue
        if model_spec not in process_object.model_specs:
            print 'did not find model spec in fit-predict output:', model_spec
            continue
        for i, feature_name in enumerate(seven.models.features):
            losses = process_object.losses_model_spec[model_spec]
            if verbose:
                print 'losses', losses
            importances = process_object.importances_model_spec[model_spec]
            if verbose:
                print 'importances', importances
            if len(losses) == 0:
                print 'zero length losses'
                pdb.set_trace()
            if len(importances) == 0:
                print 'zero length importances'
                pdb.set_trace()
            report.append_detail(
                model_spec=model_spec,
                mean_loss=(
                    np.nan if len(losses) == 1 else
                    np.mean(np.array(losses))
                ),
                feature_name=feature_name,
                mean_feature_importance=(
                    np.nan if len(importances) == 0 else
                    make_mean_feature_importance(importances, i)
                ),
            )
    return report


def make_out_report_accuracy(process_object, ticker, cusip):
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
    report.append_header('For ticker %s cusips %s' % (ticker, cusip))
    report.append_header('Summary Statistics Across All Trades')
    report.append_header(' ')

    for model_spec_name, losses_list in process_object.losses_name.iteritems():
        print model_spec_name, len(losses_list)
        losses = np.array(losses_list)
        report.append_detail(
            model_name=model_spec_name,
            n_hp_sets=len(process_object.model_specs_for_model_name[model_spec_name]),
            n_samples=len(losses_list),
            min_loss=np.min(losses),
            mean_loss=np.mean(losses),
            median_loss=np.median(losses),
            max_loss=np.max(losses),
            std_loss=np.std(losses),
        )
    return report


def do_work(control):
    'write order imbalance for each trade in the input file'
    # extract info from the fit-predict objects
    process_object = ProcessObject(control.arg.test)
    pickle_utilities.unpickle_file(
        path=control.doit.in_file,
        process_unpickled_object=process_object.process,
        on_EOFError=on_EOFError,
        on_ValueError=on_ValueError,
    )

    report_importances = make_report_importances(process_object, control.arg.ticker, control.arg.cusip)
    report_importances.write(control.doit.out_report_importances)

    out_report_accuracy = make_out_report_accuracy(process_object, control.arg.ticker, control.arg.cusip)
    out_report_accuracy.write(control.doit.out_report_accuracy)

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
