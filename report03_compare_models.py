'''compare one set of output from fit_predict

INVOCATION
  python report03_compare_models.py {ticker} {cusip} {hpset} {--test} {--testinput} {--trace}  # all dates found in WORKING

EXAMPLES OF INVOCATION
 python report03_compare_models.py ORCL 68389XAS4 grid3 -# 30 directories
 python report03_compare_models.py ORCL 68389XAS4 grid3 --testinput # 1 directory
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import os
import pdb
import pandas as pd
import pprint
import random
import sys

import applied_data_science.dirutility

from applied_data_science.Bunch import Bunch
from applied_data_science.columns_table import columns_table
import applied_data_science.debug
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.arg_type
import seven.build
import seven.fit_predict_output
import seven.reports
import seven.ModelSpec

pp = pprint.pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', type=seven.arg_type.ticker)
    parser.add_argument('cusip', type=seven.arg_type.cusip)
    parser.add_argument('hpset', type=seven.arg_type.hpset)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--testinput', action='store_true')  # read from input directory ending in '-test'
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--just_importances', action='store_true')
    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = seven.build.report03_compare_models(arg.ticker, arg.cusip, arg.hpset, test=arg.test, testinput=arg.testinput)
    pp(paths)
    if len(paths['in_predictions']) == 0:
        print arg
        print 'no predictions found'
        sys.exit(1)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        path=paths,
        max_absolute_error_processed=200.0,
        random_seed=random_seed,
        timer=Timer(),
    )


Error = collections.namedtuple(
    'Error',
    'query_index model_spec_str target_feature_name actual predicted absolute_error',
)
Importance = collections.namedtuple(
    'Importance',
    'query_index model_spec_str target_feature_name prediction_feature_name importance'
)


def column_def(column_name):
    return seven.reports.all_columns_2[column_name]


def column_defs(column_names):
    return map(column_def, column_names)


def write_the_lines(heading_lines, table_lines, path, verbose=True):
    verbose = True
    if verbose:
        print 'lines written to %s' % path
        for line in heading_lines + table_lines:
            print line
    with open(path, 'w') as f:
        f.writelines('%s\n' % line for line in heading_lines + table_lines)


def write_report(lines, path, verbose=True):
    if verbose:
        print 'lines written to %s' % path
        for line in lines:
            print line
    with open(path, 'w') as f:
        f.writelines('%s\n' % line for line in lines)


def reports_best_features(best_modelspec, importances, control):
    'return None; also create and write reports on the immportance of features'
    def make_report(extra_headings, columns_table_lines):
        'return list of lines'
        common_headings = [
            'Feature Importances for the Best Model (%s)' % best_modelspec,
            'Including all predictions for ticker %s cusip %s hpset %s' % (
                control.arg.ticker,
                control.arg.cusip,
                control.arg.hpset,
            ),
            'Mean importance across predictions and trade types',
        ]
        return common_headings + extra_headings + [' '] + columns_table_lines

    def report_on_best_model():
        # best = make_best()
        all_importances = collections.defaultdict(list)  # Dict[feature_name, feature_importance list]
        relevant = importances[importances['model_spec'] == best_modelspec]
        for index, row in relevant.iterrows():
            all_importances[row['feature_name']].append(row['feature_importance'])

        mean_importances = []
        for feature_name, importances_list in all_importances.iteritems():
            mean_importances.append((
                feature_name,
                sum(importances_list) / (1.0 * len(importances_list))
            ))

        ct = columns_table(
            column_defs(('feature_name', 'feature_importance')),
            sorted(mean_importances, key=lambda x: x[1], reverse=True),  # sort ascending by mean importance
        )
        report = make_report([], ct)
        write_report(report, control.path['out_importances'])

    report_on_best_model()
    return None


def reports_mean_absolute_errors(df, control):
    'return best model_specs across trade types; also create and write reports'
    def make_confidence_interval(v):
        'return (lower_bound, upper_bound) for 95% confidence interval for values in vector v'
        v_sample = v.sample(
            n=10000,
            replace=True,
            random_state=control.random_seed,
        )
        # Note: if most of the predictions have zero errors, then the confidence interval wiill be [0,0]
        # even when the mean absolute error is positive
        ci_05 = v_sample.quantile(0.05)
        ci_95 = v_sample.quantile(0.95)
        return ci_05, ci_95

    def make_table_details(df, model_spec):
        'return column table containing just the model spec'
        relevant_mask = df['model_spec'] == model_spec
        relevant = df[relevant_mask]
        sorted_df = relevant.sort_values(by=['effectivedatetime'], axis='index')
        details = []
        column_names = ('trace_index', 'effectivedatetime', 'quantity', 'actual', 'prediction', 'absolute_error')
        details = []
        for index, row in sorted_df.iterrows():
            line = []
            for column_name in column_names:
                if column_name == 'effectivedatetime':
                    line.append(str(row[column_name]))   # pandas Timestamps don't convert to %s formats correctly
                else:
                    line.append(row[column_name])
            details.append(line)
        return columns_table(
            column_defs(column_names),
            details,
        )

    def make_table_modelspec(df):
        'return list of lines in the column table'
        def make_detail_lines(df):
            'return DataFrame with data for the detail lines'
            result = pd.DataFrame(columns=[
                'model_spec',
                'n_prints',
                'mean_absolute_error',
                'mae_ci05',
                'mae_ci95',
            ])
            for model_spec in set(df['model_spec']):
                relevant = df['model_spec'] == model_spec
                absolute_errors_relevant = df['absolute_error'].loc[relevant]
                n_prints = len(absolute_errors_relevant)
                mean_absolute_error = absolute_errors_relevant.mean()
                ci_05, ci_95 = make_confidence_interval(absolute_errors_relevant)
                result.loc[len(result)] = [model_spec, n_prints, mean_absolute_error, ci_05, ci_95]
            return result

        sorted_detail_lines = make_detail_lines(df).sort_values(
            by=['mean_absolute_error'],
            axis='index',
        )
        details = []
        column_names = ('model_spec', 'n_prints', 'mean_absolute_error', 'mae_ci05', 'mae_ci95')
        for index, row in sorted_detail_lines.iterrows():
            line = [row[column_name] for column_name in column_names]
            details.append(line)
        return columns_table(
            column_defs(column_names),
            details,
        )

    def make_table_tradetype_modelspec(df):
        'return list of lines in the column table'
        def make_detail_lines(df):
            'return DataFrame containing detail lines'
            result = pd.DataFrame(columns=[
                'model_spec',
                'trade_type',
                'n_prints',
                'mean_absolute_error',
                'mae_ci05',
                'mae_ci95',
            ])
            for model_spec in set(df['model_spec']):
                for trade_type in set(df['trade_type']):
                    relevant = (
                        (df['model_spec'] == model_spec) &
                        (df['trade_type'] == trade_type)
                    )
                    absolute_errors_relevant = df['absolute_error'].loc[relevant]
                    n_prints = len(absolute_errors_relevant)
                    mean_absolute_error = absolute_errors_relevant.mean()
                    ci_05, ci_95 = make_confidence_interval(absolute_errors_relevant)
                    result.loc[len(result)] = [model_spec, trade_type, n_prints, mean_absolute_error, ci_05, ci_95]
            return result

        sorted_detail_lines = make_detail_lines(df).sort_values(
            by=['trade_type', 'mean_absolute_error'],
            axis='index',
        )
        details = []
        column_names = ('trade_type', 'model_spec', 'n_prints', 'mean_absolute_error', 'mae_ci05', 'mae_ci95')
        for index, row in sorted_detail_lines.iterrows():
            line = [row[column_name] for column_name in column_names]
            details.append(line)
        return columns_table(
            column_defs(column_names),
            details,
        )

    def make_report(prefix_headings, suffix_headings, body_lines):
        'return list of lines'
        common_headings = [
            'Mean Absolute Errors and 95% Confidence Intervals',
            'Including all predictions for ticker %s cusip %s hpset %s' % (
                control.arg.ticker,
                control.arg.cusip,
                control.arg.hpset,
            ),
            'Excluding predictions with an absolute error exceededing %f' % control.max_absolute_error_processed,
        ]
        return prefix_headings + common_headings + suffix_headings + [' '] + body_lines

    def find_best_modelspec(df):
        'return model spec with lowest mean absolute error'
        mean_absolute_errors = []
        for model_spec in set(df['model_spec']):
            relevant_mask = df['model_spec'] == model_spec
            relevant = df[relevant_mask]
            absolute_errors = relevant['absolute_error']
            mean_absolute_error = absolute_errors.mean()
            mean_absolute_errors.append((mean_absolute_error, model_spec))
        result = sorted(mean_absolute_errors)[0][1]  # lowest mean absolute error.model_spec
        return result

    best_modelspec = find_best_modelspec(df)

    write_report(
        lines=make_report(
            prefix_headings=['Trade Details for Model Spec %s' % best_modelspec],
            suffix_headings=[],
            body_lines=make_table_details(df, best_modelspec),
        ),
        path=control.path['out_details'],
    )

    write_report(
        lines=make_report(
            prefix_headings=['Mean Absolute Errors and 95% Confidence Intervals'],
            suffix_headings=['By Model Spec', 'Sorted by Increasing Mean Absolute Error'],
            body_lines=make_table_modelspec(df),
        ),
        path=control.path['out_accuracy_modelspec'],
    )

    write_report(
        lines=make_report(
            prefix_headings=['Mean Absolute Errors and 95% Confidence Intervals'],
            suffix_headings=['By Target Feature and Model Spec', 'Sorted by Increasing Mean Absolute Error'],
            body_lines=make_table_tradetype_modelspec(df),
        ),
        path=control.path['out_accuracy_targetfeature_modelspec'],
    )

    return best_modelspec


def read_and_transform_importances(control):
    'return DataFrame containing non-zero feature importances'
    skipped = collections.Counter()
    counters = collections.Counter()

    def skip(s):
        skipped[s] += 1

    def count(s):
        counters[s] += 1

    data = collections.defaultdict(list)
    for in_file_path in control.path['in_importances']:
        print in_file_path
        count('file paths examined')
        if os.path.getsize(in_file_path) == 0:
            print 'skipping empty file: %s' % in_file_path
            continue
        with open(in_file_path, 'rb') as f:
            count('file paths read')
            obj = pickle.load(f)
            if obj is None:
                skip('obj is none in %s' % in_file_path)
                continue
            if len(obj) == 0:
                skip(('len(obj) == 0 in %s' % in_file_path))
            for output_key, importance in obj.iteritems():
                if importance is None:
                    skip('importance is None')
                    continue
                if any_nans(output_key):
                    skip('nan in outputkey %s' % output_key)
                    continue
                if any_nans(importance):
                    skip('NaN in importance %s %s' % (output_key, importance))
                    continue
                if importance.importance is None:
                    skip('importance.importance is None')
                    continue

                # copy data from fit_predict
                for feature_name, feature_importance in importance.importance.iteritems():
                    data['feature_name'].append(feature_name)
                    data['feature_importance'].append(feature_importance)

                    data['trace_index'].append(output_key.trace_index)
                    data['model_spec'].append(output_key.model_spec)
                    data['effectivedatetime'].append(importance.effectivedatetime)
                    data['trade_type'].append(importance.trade_type)
                    count('importances')
        if control.arg.test and len(data['feature_name']) > 1000:
            print 'stopping input: TRACE'
            break

    importances = pd.DataFrame(data=data)

    print 'skipped %d input records' % len(skipped)
    for reason, count in sorted(skipped.iteritems()):
        print '%70s: %d' % (reason, count)

    print 'counters'
    for counter, count in counters.iteritems():
        print '%40s: %d' % (counter, count)

    print 'retained %d importance records' % len(importances)
    return importances


def any_nans(obj):
    def isnan(x):
        return x != x

    if isinstance(obj, seven.fit_predict_output.OutputKey):
        return (
            isnan(obj.trace_index) or
            isnan(obj.model_spec)
        )
    elif isinstance(obj, seven.fit_predict_output.Prediction):
        return (
            isnan(obj.effectivedatetime) or
            isnan(obj.trade_type) or
            isnan(obj.actual) or
            isnan(obj.prediction)
        )
    elif isinstance(obj, seven.fit_predict_output.Importance):
        return (
            isnan(obj.effectivedatetime) or
            isnan(obj.trade_type) or
            isnan(obj.importance)
        )
    else:
        print 'bad type', type(obj)
        pdb.set_trace()


def read_and_transform_predictions(control):
    'return DataFrame with absolute_error column added'
    skipped = collections.Counter()
    counters = collections.Counter()

    def skip(s):
        skipped[s] += 1

    def count(s):
        counters[s] += 1

    data = collections.defaultdict(list)
    large_absolute_errors = {}
    for in_file_path in control.path['in_predictions']:
        count('file paths examined')
        if os.path.getsize(in_file_path) == 0:
            skip('empty file: %s' % in_file_path)
            continue
        with open(in_file_path, 'rb') as f:
            count('files opened')
            obj = pickle.load(f)
            if len(obj) == 0:
                skip('obj in file %s has length 0' % in_file_path)
                continue
            for output_key, prediction in obj.iteritems():
                if any_nans(output_key):
                    skip('nan in outputkey %s' % output_key)
                    continue
                if any_nans(prediction):
                    skip('NaN in prediction %s %s' % (output_key, prediction))
                    continue
                if prediction.prediction is None:
                    skip('prediction is None')
                    continue
                # copy data from fit_predict
                absolute_error = abs(prediction.prediction - prediction.actual)
                if absolute_error > control.max_absolute_error_processed:
                    skip("large_absolute_error trace index %s model_spec %s" % (
                        output_key.trace_index,
                        output_key.model_spec,
                    ))
                    large_absolute_errors[output_key] = prediction
                    continue
                data['trace_index'].append(output_key.trace_index)
                data['model_spec'].append(output_key.model_spec)
                data['effectivedatetime'].append(prediction.effectivedatetime)
                data['trade_type'].append(prediction.trade_type)
                data['quantity'].append(prediction.quantity)
                data['actual'].append(prediction.actual)
                data['prediction'].append(prediction.prediction)
                # create columns
                data['absolute_error'].append(absolute_error)
                count('prediction records appended to data frame')
        if control.arg.test and len(data['trace_index']) > 1000:
            print 'stopping input: TRACE'
            break
    predictions = pd.DataFrame(data=data)

    print 'skipped %d input records' % len(skipped)
    for reason, count in skipped.iteritems():
        print '%40s: %d' % (reason, count)
    print 'counts'
    for what, count in skipped.iteritems():
        print '%40s: %d' % (what, count)
    with open(control.path['out_large_absolute_errors'], 'w') as f:
        # write a CSV file
        headers = [
            'trace_index',
            'model_spec',
            'effectivedatetime',
            'trade_type',
            'quantity',
            'actual',
            'prediction',
        ]
        f.write(','.join(headers))
        f.write('\n')
        for output_key, prediction in large_absolute_errors.iteritems():
            values = []
            values.append('%s' % output_key.trace_index)
            values.append('%s' % output_key.model_spec)
            values.append('%s' % prediction.effectivedatetime)
            values.append('%s' % prediction.trade_type)
            values.append('%s' % prediction.quantity)
            values.append('%s' % prediction.actual)
            values.append('%s' % prediction.prediction)
            f.write(','.join(values))
            f.write('\n')
    print 'wrote %d records with large absolute errors; all were skipped' % len(large_absolute_errors)
    print 'retained %d predictions' % len(predictions)
    return predictions


def check_errors(errors):
    'stop if a problem with the input data'
    index_counts = collections.Counter()
    for x in errors:
        index_counts[x.query_index] += 1

    print 'counts by query_index'
    pp(dict(index_counts))
    # check that each index count is the same
    # If so, the same number of models ran for each index (each print)
    # this code assumes that the same models ran for each index (but that could be checked)
    expected_index_count = None
    for k, v in index_counts.iteritems():
        if expected_index_count is None:
            expected_index_count = v
        else:
            assert v == expected_index_count, (k, expected_index_count)


def do_work(control):
    'produce reports'
    # produce reports on mean absolute errors
    if not control.arg.just_importances:
        predictions = read_and_transform_predictions(control)
        best_model = reports_mean_absolute_errors(predictions, control)

    # produce reports on importances of features for the most accurate models
    importances = read_and_transform_importances(control)
    reports_best_features(best_model, importances, control)


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
