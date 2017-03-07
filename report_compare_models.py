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
import pandas as pd
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
        self.out_report_stats_by_modelname_ntradesback = os.path.join(self.dir_out, 'report-stats-by-modelspec-tradesback.txt')
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
        self.df = pd.DataFrame()  # default index (not query_index values)
        self.column_values = collections.defaultdict(list)

    def process(self, obj):
        'accumulate losses by model_spec.model'
        self.count += 1
        if self.count % 10000 == 1:
            print 'processing input record number', self.count

        # accumulate values that will go into self.df
        for column_name, value in obj.as_dict().iteritems():
            self.column_values[column_name].append(value)

    def close(self):
        'create self.df, which depends on self.column_values'
        self.df = pd.DataFrame(
            data=self.column_values,
            index=None,  # use default index which is np.arange(n)
        )
        # add columns not in the process object
        error = self.df.predicted_value - self.df.actual_value
        self.df['error'] = error
        self.df['squared_error'] = error * error

        names = []
        n_trades_back = []
        for model_spec in self.df.model_spec:
            names.append(model_spec.name)
            n_trades_back.append(model_spec.n_trades_back)
        self.df['name'] = pd.Series(data=names, index=self.df.index)
        self.df['n_trades_back'] = pd.Series(data=n_trades_back, index=self.df.index)

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
    'return a Report for feature importances for the rf models'
    verbose = False

    def make_mean_feature_importance(importances, i):
        'return mean importance of i-th feature'
        importances_i = []
        for importance in importances:
            importances_i.append(importance[i])
        return np.mean(np.array(importances_i))

    report = ReportColumns(seven.reports.columns(
        'model_spec',
        'rmse',
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
    report. append_header('Covering %d distinct query trades' % len(set(process_object.df.query_index)))
    report.append_header('Summary Statistics Across All %d Predictions' % process_object.count)
    report.append_header(' ')

    for model_spec in sorted(set(process_object.df.model_spec)):
        if verbose:
            print model_spec
        if model_spec.name != 'rf':
            continue
        subset = process_object.df.loc[process_object.df.model_spec == model_spec]
        importances = subset.importances
        for i, feature_name in enumerate(seven.models.features):
            report.append_detail(
                model_spec=model_spec,
                rmse=np.mean(subset.squared_error),
                feature_name=feature_name,
                mean_feature_importance=make_mean_feature_importance(importances, i),
            )
    return report


def make_report_stats_by_modelname(process_object, ticker, cusip, hpset, effective_date):
    report = ReportColumns(seven.reports.columns(
        'trade_type',
        'model_name',
        'n_hp_sets',
        'n_samples',
        'n_predictions',
        'min_abs_error',
        'rmse',
        'median_abs_error',
        'max_abs_error',
        'std_abs_error',
        ))
    report.append_header('Error Statistics By Model Name')
    report.append_header('For Ticker %s Cusip %s HpSet %s Effective Date %s' % (
        ticker,
        cusip,
        hpset,
        effective_date,
    ))
    report. append_header('Covering %d distinct query trades' % len(set(process_object.df.query_index)))
    report.append_header('Summary Statistics Across All %d Predictions' % process_object.count)
    report.append_header(' ')

    for trade_type in set(process_object.df.trade_type):
        for name in set(process_object.df.name):
            subset = process_object.df.loc[
                (process_object.df.trade_type == trade_type) &
                (process_object.df.name == name)]
            report.append_detail(
                trade_type=trade_type,
                model_name=name,
                n_hp_sets=len(set(subset.model_spec)),
                n_samples=len(set(subset.query_index)),
                n_predictions=len(subset),
                min_abs_error=np.min(np.abs(subset.error)),
                rmse=np.mean(subset.squared_error),
                median_abs_error=np.median(subset.squared_error),
                max_abs_error=np.max(np.abs(subset.error)),
                std_abs_error=np.std(subset.error),
            )
    return report


def make_report_stats_by_modelname_ntradesback(process_object, ticker, cusip, hpset, effective_date):
    report = ReportColumns(seven.reports.columns(
        'trade_type',
        'model_name',
        'n_trades_back',
        'n_hp_sets',
        'n_samples',
        'n_predictions',
        'min_abs_error',
        'rmse',
        'median_abs_error',
        'max_abs_error',
        'std_abs_error',
        ))
    report.append_header('Error Statistics By Model Name')
    report.append_header('For Ticker %s Cusip %s HpSet %s Effective Date %s' % (
        ticker,
        cusip,
        hpset,
        effective_date,
    ))
    report. append_header('Covering %d distinct query trades' % len(set(process_object.df.query_index)))
    report.append_header('Summary Statistics Across All %d Predictions' % process_object.count)
    report.append_header(' ')

    for trade_type in set(process_object.df.trade_type):
        for name in set(process_object.df.name):
            if name == 'n':
                continue
            subset_tradetype_name = process_object.df.loc[
                (process_object.df.trade_type == trade_type) &
                (process_object.df.name == name)]
            for n_trades_back in sorted(set(subset_tradetype_name.n_trades_back)):
                subset = subset_tradetype_name[
                    subset_tradetype_name.n_trades_back == n_trades_back
                ]
                report.append_detail(
                    trade_type=trade_type,
                    model_name=name,
                    n_trades_back=None if np.isnan(n_trades_back) else n_trades_back,
                    n_hp_sets=len(set(subset.model_spec)),
                    n_samples=len(set(subset.query_index)),
                    n_predictions=len(subset),
                    min_abs_error=np.min(np.abs(subset.error)),
                    rmse=np.mean(subset.squared_error),
                    median_abs_error=np.median(subset.squared_error),
                    max_abs_error=np.max(np.abs(subset.error)),
                    std_abs_error=np.std(subset.error),
                )
    return report


def make_report_stats_by_modelspec(process_object, ticker, cusip, hpset, effective_date):
    report = ReportColumns(seven.reports.columns(
        'model_spec',
        'n_hp_sets',
        'n_samples',
        'n_predictions',
        'min_abs_error',
        'rmse',
        'median_abs_error',
        'max_abs_error',
        'std_abs_error',
        ))
    report.append_header('Error Statitics By Model Spec')
    report.append_header('For ticker %s cusips %s HpSet %s Effective Date %s' % (
        ticker,
        cusip,
        hpset,
        effective_date,
    ))
    report. append_header('Covering %d distinct query trades' % len(set(process_object.df.query_index)))
    report.append_header('Summary Statistics Across All %d Predictions' % process_object.count)
    report.append_header(' ')

    for model_spec in sorted(set(process_object.df.model_spec)):
        subset = process_object.df.loc[process_object.df.model_spec == model_spec]
        report.append_detail(
            model_spec=model_spec,
            n_hp_sets=len(set(subset.model_spec)),
            n_samples=len(set(subset.query_index)),
            n_predictions=len(subset),
            min_abs_error=np.min(np.abs(subset.error)),
            rmse=np.mean(subset.squared_error),
            median_abs_error=np.median(subset.squared_error),
            max_abs_error=np.max(np.abs(subset.error)),
            std_abs_error=np.std(subset.error),
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
    process_object.close()

    def create_and_write_report(make_report_fn, write_path):
        report_importances = make_report_fn(
            process_object,
            control.arg.ticker,
            control.arg.cusip,
            control.arg.hpset,
            control.arg.effective_date,
        )
        report_importances.write(write_path)

    create_and_write_report(
        make_report_importances,
        control.doit.out_report_importances,
    )
    create_and_write_report(
        make_report_stats_by_modelname,
        control.doit.out_report_stats_by_modelname,
    )
    create_and_write_report(
        make_report_stats_by_modelname_ntradesback,
        control.doit.out_report_stats_by_modelname_ntradesback,
    )
    create_and_write_report(
        make_report_stats_by_modelspec,
        control.doit.out_report_stats_by_modelspec,
    )

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
