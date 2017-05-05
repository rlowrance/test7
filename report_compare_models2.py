'''compare one set of output from fit_predict

INVOCATION
  python report_compare_models2.py {ticker} {cusip} {hpset} {--test} {--trace}  # all dates found in WORKING

EXAMPLES OF INVOCATION
 python report_compare_models2.py orcl 68389XAS4 grid2
'''

from __future__ import division

import argparse
import collections
import numpy as np
import pdb
import pandas as pd
import pprint
import random
import sys

import applied_data_science.dirutility

from applied_data_science.Bunch import Bunch
from applied_data_science.ColumnsTable import ColumnsTable
from applied_data_science.Date import Date
from applied_data_science.Logger import Logger
from applied_data_science.Report import Report
from applied_data_science.Timer import Timer

from seven import arg_type
import seven.reports

import build
pp = pprint.pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', type=arg_type.ticker)
    parser.add_argument('cusip', type=arg_type.cusip)
    parser.add_argument('hpset', type=arg_type.hpset)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = build.report_compare_models2(arg.ticker, arg.cusip, arg.hpset, test=arg.test)
    pp(paths)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=Timer(),
    )


def reports_mean_errors(predictions, path_both, path_model_spec_str, path_predicted_feature_name, arg):
    'write several reports on the mean errors'
    d = collections.defaultdict(list)
    d_model_spec_str = collections.defaultdict(list)
    d_predicted_feature_name = collections.defaultdict(list)
    found_nan = False
    for row_index, row_series in predictions.iterrows():
        model_spec_str = row_series['model_spec_str']
        predicted_feature_name = row_series['predicted_feature_name']
        absolute_error = row_series['absolute_error']
        if np.isnan(absolute_error):
            print 'found nan', model_spec_str, predicted_feature_name
            found_nan = True
        d[(model_spec_str, predicted_feature_name)].append(absolute_error)
        d_model_spec_str[model_spec_str].append(absolute_error)
        d_predicted_feature_name[predicted_feature_name].append(absolute_error)
    if found_nan:
        print 'found at least one nan'
        pdb.set_trace()

    def make_report(ct):
        report = Report()
        report.append('Mean Absolute Error Sorted By Increasing Mean Absolute Error')
        report.append('For ticker %s cusip %s hpset %s' % (arg.ticker, arg.cusip, arg.hpset))
        report.append('For all predicatable prints in 2016-11')
        report.append(' ')
        for line in ct.iterlines():
            report.append(line)
        return report

    def make_ct_d(d):
        'return ColumnTable with 3 columns'
        column_defs = seven.reports.columns('model_spec', 'predicted_feature', 'mean_absolute_error')
        ct = ColumnsTable(column_defs)

        d1 = {k: np.mean(v) for k, v in d.iteritems()}
        d2 = collections.OrderedDict(sorted(d1.items(), key=lambda x: x[1]))

        for (model_spec, predicted_feature), mean_absolute_error in d2.iteritems():
            ct.append_detail(
                model_spec=model_spec,
                predicted_feature=predicted_feature,
                mean_absolute_error=mean_absolute_error,
            )
        return ct

    def make_ct_d_model_spec_str(d):
        'return ColumnTable with 2 columns'
        column_defs = seven.reports.columns('model_spec', 'mean_absolute_error')
        ct = ColumnsTable(column_defs)

        d1 = {k: np.mean(v) for k, v in d.iteritems()}
        d2 = collections.OrderedDict(sorted(d1.items(), key=lambda x: x[1]))

        for model_spec, mean_absolute_error in d2.iteritems():
            ct.append_detail(
                model_spec=model_spec,
                mean_absolute_error=mean_absolute_error,
            )
        return ct

    def make_ct_d_predicted_feature_name(d):
        'return ColumnTable with 2 columns'
        column_defs = seven.reports.columns('predicted_feature', 'mean_absolute_error')
        ct = ColumnsTable(column_defs)

        d1 = {k: np.mean(v) for k, v in d.iteritems()}
        d2 = collections.OrderedDict(sorted(d1.items(), key=lambda x: x[1]))

        for predicted_feature, mean_absolute_error in d2.iteritems():
            ct.append_detail(
                predicted_feature=predicted_feature,
                mean_absolute_error=mean_absolute_error,
            )
        return ct

    make_report(make_ct_d(d)).write(path_both)
    make_report(make_ct_d_model_spec_str(d_model_spec_str)).write(path_model_spec_str)
    make_report(make_ct_d_predicted_feature_name(d_predicted_feature_name)).write(path_predicted_feature_name)


def report_each_prediction(predictions, path, arg):
    'write report summarizing all predictions all the predictions'
    d = collections.defaultdict(collections.Counter)
    for row_index, row_series in predictions.iterrows():
        key = (row_series['model_spec_str'], row_series['predicted_feature_name'])
        d[key][row_series['predicted']] += 1

    sorted_d = collections.OrderedDict(sorted(d.items(), key=lambda x: x[0]))

    column_defs = seven.reports.columns('model_spec', 'predicted_feature', 'prediction', 'count')
    ct = ColumnsTable(column_defs)
    for k, v in sorted_d.iteritems():
        for prediction, count in v.iteritems():
            ct.append_detail(
                model_spec=k[0],
                predicted_feature=k[1],
                prediction=prediction,
                count=count,
            )

    report = Report()
    report.append('Count of Number of Times Each Prediction Was Made')
    report.append('For ticker %s cusip %s hpset %s' % (arg.ticker, arg.cusip, arg.hpset))
    report.append('For all predicatable prints in 2016-11')
    report.append(' ')
    for line in ct.iterlines():
        report.append(line)
    report.write(path)


def report_accuracy_for(filter, predictions):
    'return a ColumnsTable containing the mean absolute error'
    column_defs = seven.reports.columns('model_spec', 'predicted_feature', 'mean_absolute_error')
    ct = ColumnsTable(column_defs)

    mean_absolute_errors = {}  # for now, report only on this metric
    # median_absolute_errors = {}
    # min_absolute_errors = {}
    # max_absolute_errors = {}
    for model_spec_str in set(predictions['model_spec_str']):
        predictions_model_spec_str = predictions[predictions['model_spec_str'] == model_spec_str]
        for predicted_feature_name in set(predictions_model_spec_str['predicted_feature_name']):
            if not filter(model_spec_str, predicted_feature_name):
                continue
            predictions_predicted_feature_name = predictions[predictions['predicted_feature_name'] == predicted_feature_name]
            absolute_errors = predictions_predicted_feature_name['absolute_error']
            key = (model_spec_str, predicted_feature_name)
            mean_absolute_errors[key] = np.mean(absolute_errors)
            # median_absolute_errors[key] = np.median(absolute_errors)
            # min_absolute_errors[key] = np.min(absolute_errors)
            # max_absolute_errors[key] = np.max(absolute_errors)

    sorted_mean_absolute_errors = collections.OrderedDict(sorted(mean_absolute_errors.items(), key=lambda x: x[1]))

    for k, v in sorted_mean_absolute_errors.iteritems():
        ct.append_detail(
            model_spec=k[0],
            predicted_feature=k[1],
            mean_absolute_error=v,
        )
    return ct


def do_work(control):
    'produce report'
    def make_trade_date(path):
        date_str = '-'.join(path.split('.')[0].split('\\')[-2].split('-')[-3:])
        d = Date(from_yyyy_mm_dd=date_str)
        return d.as_datetime_date()

    all_predictions = pd.DataFrame()
    for in_file_path in control.path['in_files']:
        predictions = pd.read_csv(
            in_file_path,
            index_col=[0, 1, 2],
        )
        print 'read %d predictions from file %s' % (len(predictions), in_file_path.split('\\')[-2])
        if len(predictions) > 0:
            predictions['trade_date'] = make_trade_date(in_file_path)
            predictions['absolute_error'] = np.abs(predictions['predicted'] - predictions['actual'])
            all_predictions = all_predictions.append(predictions)
            is_null_predicted = pd.isnull(predictions['predicted'])
            is_null_actual = pd.isnull(predictions['actual'])
            is_null_absolute_error = pd.isnull(predictions['absolute_error'])
            if is_null_predicted.any() or is_null_actual.any() or is_null_absolute_error.any():
                print 'found null'
                print in_file_path
                pdb.set_trace()
    print
    print 'read %d records from %d input files' % (len(all_predictions), len(control.path['in_files']))

    reports_mean_errors(
        all_predictions.copy(deep=True),
        control.path['out_accuracy_both'],
        control.path['out_accuracy_model_spec'],
        control.path['out_accuracy_predicted_feature_name'],
        control.arg,
    )
    report_each_prediction(
        all_predictions.copy(deep=True),
        control.path['out_prediction_counts'],
        control.arg,
    )


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    # print control
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb

    main(sys.argv)
