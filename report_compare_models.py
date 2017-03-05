'''print report comparing model accuracy

INVOCATION
  python targets.py {ticker} {cusip} {hpset} {effective_date} [--test] [--trace]
where
 ticker is the ticker symbol (ex: orcl)
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 hpset is the name of the hyperparameter set (ex: grid2)
 effective_date: YYYY-MM-DD is the date of the trade
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python report_compare_models.py orcl 68389XAS4 grid2 2016-11-01

INPUTS
 WORKING/fit_predict_{ticker}_{cusip}_*_{effective_date}/fit-predict-output.pickle

OUTPUTS
 WORKING/ME-{ticker}-{cusip}-{effective_date}/report-stat-by-modelname.txt
 WORKING/ME-{ticker}-{cusip}-{effective_date}/report-importances.txt
 WORKING/tagets/0log-{ticker}.txt

'''

from __future__ import division

import argparse
import collections
import numpy as np
import os
import pdb
from pprint import pprint
import random
import sys

from Bunch import Bunch
import dirutility
from Logger import Logger
import pickle_utilities
from ReportColumns import ReportColumns
import seven.arg_type
from seven.FitPredictOutput import FitPredictOutput
import seven.models
from seven.ModelSpec import ModelSpec
import seven.path
import seven.reports
from Timer import Timer


class Doit(object):
    def __init__(self, ticker, cusip, hpset, effective_date, test=False, me='report_compare_models'):
        'create paths for I/O and receipts for doit'
        self.ticker = ticker
        self.cusip = cusip
        self.effective_date = effective_date
        self.me = me
        self.test = test

        # define directories
        dir_working = seven.path.working()
        self.dir_out = os.path.join(
            dir_working,
            '%s-%s-%s-%s-%s' % (self.me, ticker, cusip, hpset, effective_date) + ('-test' if test else '')
        )

        # input/output files
        self.in_file = os.path.join(
            dir_working,
            'fit_predict-%s-%s-%s-%s' % (ticker, cusip, hpset, effective_date),
            'fit-predict-output.pickle'
        )
        self.out_log = os.path.join(self.dir_out, '0log.txt')
        self.out_report_importances = os.path.join(self.dir_out, 'report-importances.txt')
        self.out_report_stats_by_modelname = os.path.join(self.dir_out, 'report-stats-by-modelname.txt')
        self.out_report_stats_by_modelspec = os.path.join(self.dir_out, 'report-stats-by-modelspec.txt')

        # used by Doit tasks
        self.actions = [
            'python %s.py %s %s %s' % (self.me, ticker, cusip, effective_date)
        ]
        self.targets = [
            self.out_log,
            self.out_report_importances,
            self.out_report_stats_by_modelname,
        ]
        self.file_deps = [
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
    arg_type = seven.arg_type
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
    dirutility.assure_exists(doit.dir_out)

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
        self.query_indices = set()
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
        self.query_indices.add(obj.query_index)
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


def make_report_importances(process_object, ticker, cusip, hpset, effective_date):
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
    report.append_header('For Ticker %s Cusip %s HpSet %s Effective Date %s' % (
        ticker,
        cusip,
        hpset,
        effective_date,
    ))
    report. append_header('Covering %d distinct query trades' % len(process_object.query_indices))
    report.append_header('Summary Statistics Across All %d Trades' % process_object.count)
    report.append_header(' ')

    for model_spec in process_object.model_specs:
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


def make_report_stats_by_modelname(process_object, ticker, cusip, hpset, effective_date):
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
    report.append_header('Loss Statitics By Model Name')
    report.append_header('For ticker %s cusips %s HpSet %s Effective Date %s' % (
        ticker,
        cusip,
        hpset,
        effective_date,
    ))
    report. append_header('Covering %d distinct query trades' % len(process_object.query_indices))
    report.append_header('Summary Statistics Across All %d Trades' % process_object.count)
    report.append_header(' ')

    for model_spec_name, losses_list in process_object.losses_name.iteritems():
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


def make_report_stats_by_modelspec(process_object, ticker, cusip, hpset, effective_date):
    report = ReportColumns(seven.reports.columns(
        'model_spec',
        'n_hp_sets',
        'n_samples',
        'min_loss',
        'mean_loss',
        'median_loss',
        'max_loss',
        'std_loss',
        ))
    report.append_header('Loss Statitics By Model Spec')
    report.append_header('For ticker %s cusips %s HpSet %s Effective Date %s' % (
        ticker,
        cusip,
        hpset,
        effective_date,
    ))
    report. append_header('Covering %d distinct query trades' % len(process_object.query_indices))
    report.append_header('Summary Statistics Across All %d Trades' % process_object.count)
    report.append_header(' ')

    # determine sort order, which is by model spec string
    sorted_model_specs = sorted(process_object.losses_model_spec.keys())
    model_spec_strs = [
        str(model_spec)
        for model_spec in process_object.losses_model_spec.keys()
    ]
    sorted_model_specs = sorted(model_spec_strs)
    for model_spec_str in sorted_model_specs:
        model_spec = ModelSpec.make_from_str(model_spec_str)
        losses_list = process_object.losses_model_spec[model_spec]
    # for model_spec, losses_list in process_object.losses_model_spec.iteritems():
        print model_spec, len(losses_list)
        losses = np.array(losses_list)
        report.append_detail(
            model_spec=model_spec,
            n_hp_sets=len(process_object.model_specs_for_model_name[model_spec.name]),
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
    # extract info from the fit-predict objects across the files in the hpset
    process_object = ProcessObject(control.arg.test)
    pickle_utilities.unpickle_file(
        path=control.doit.in_file,
        process_unpickled_object=process_object.process,
        on_EOFError=on_EOFError,
        on_ValueError=on_ValueError,
    )

    def create_and_write_report(make_report_fn, write_path):
        report_importances = make_report_fn(
            process_object,
            control.arg.ticker,
            control.arg.cusip,
            control.arg.hpset,
            control.arg.effective_date,
        )
        report_importances.write(write_path)

    create_and_write_report(make_report_importances, control.doit.out_report_importances)
    create_and_write_report(make_report_stats_by_modelname, control.doit.out_report_stats_by_modelname)
    create_and_write_report(make_report_stats_by_modelspec, control.doit.out_report_stats_by_modelspec)

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
