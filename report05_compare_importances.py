'''compare one set of output from fit_predict

INVOCATION
  python report05_compare_importances.py {ticker} {cusip} {hpset} {n} {--test} {--testinput} {--trace}  # all dates found in WORKING

EXAMPLES OF INVOCATION
 python report05_compare_importances.py ORCL 68389XAS4 grid3 1 # importances of the best model
 python report05_compare_importances.py ORCL 68389XAS4 grid3 2 --test --testinput # importances of the 2 best models
 py report03_compare_importances.py ORCL 68389XBM6 grid1
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
    parser.add_argument('n', type=seven.arg_type.positive_int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--testinput', action='store_true')  # read from input directory ending in '-test'
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = seven.build.report05_compare_importances(arg.ticker, arg.cusip, arg.hpset, arg.n, test=arg.test, testinput=arg.testinput)
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


def reports_best_features(best_modelspecs, importances, control):
    'return None; also create and write reports on the immportance of features'
    def make_report(n, best_modelspec, extra_headings, columns_table_lines):
        'return list of lines'
        common_headings = [
            'Feature Importances for the %d Nth Most Accurate Model' % n,
            'Which Had Model Spec %s' % best_modelspec,
            'Including all predictions for ticker %s cusip %s hpset %s' % (
                control.arg.ticker,
                control.arg.cusip,
                control.arg.hpset,
            ),
            'Mean importance across predictions and trade types',
        ]
        return common_headings + extra_headings + [' '] + columns_table_lines

    def report_on_best_model(n, best_modelspec):
        # return list of lines
        all_importances = collections.defaultdict(list)  # Dict[feature_name, feature_importance list]
        relevant = importances[importances['model_spec'] == best_modelspec]
        for index, row in relevant.iterrows():
            all_importances[row['feature_name']].append(row['feature_importance'])

        details = []
        data = collections.defaultdict(list)
        for feature_name, importances_list in all_importances.iteritems():
            mean_importance = sum(importances_list) / (1.0 * len(importances_list))
            details.append((
                feature_name,
                abs(mean_importance),
                mean_importance,
            ))
            data['model_spec'].append(best_modelspec)
            data['feature_name'].append(feature_name)
            data['abs_mean_importance'].append(abs(mean_importance))
            data['mean_importance'].append(mean_importance)

        df = pd.DataFrame(data=data).sort_values(by='abs_mean_importance', ascending=False)
        reordered = df[['model_spec', 'feature_name', 'abs_mean_importance', 'mean_importance']]
        with open(control.path['out_importance_%d_csv' % n], 'w') as f:
            reordered.to_csv(f)

        ct = columns_table(
            column_defs(('feature_name', 'absolute_feature_importance', 'feature_importance')),
            sorted(details, key=lambda x: x[1], reverse=True),  # sort ascending by mean importance
        )
        report = make_report(n, best_modelspec, [], ct)
        return report

    for i, best_modelspec in enumerate(best_modelspecs):
        n = i + 1
        report = report_on_best_model(n, best_modelspec)
        path = control.path['out_importance_%d' % n]
        write_report(report, path)
    return None


def read_and_transform_importances(control):
    'return (set of best modelspecs, DataFrame containing feature importances for them)'
    def pass1(control):
        'return ordered list of best modelspecs'
        mae_modelspecs = pd.read_csv(
            control.path['in_mae_modelspec'],
            index_col=0
        )
        best_modelspecs = list(mae_modelspecs.iloc[:control.arg.n]['modelspec'].values)  # List[modelspec:str]
        print 'best modelspecs:', best_modelspecs
        return best_modelspecs

    def pass2(best_modelspecs_strs, control):
        'return DataFrame containing non-zero importances for modelspecs in best_modelspecs'
        skipped = collections.Counter()
        counters = collections.Counter()

        def skip(s):
            skipped[s] += 1

        def count(s):
            counters[s] += 1

        best_modelspecs = set()
        # convert from str rep to native obj rep
        for best_modelspec_str in best_modelspecs_strs:
            best_modelspecs.add(seven.ModelSpec.ModelSpec.make_from_str(best_modelspec_str))

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
                    continue
                for output_key, importance in obj.iteritems():

                    # skip all but the best modelspecs
                    if output_key.model_spec not in best_modelspecs:
                        skip('model spec %s not in best n' % output_key.model_spec)
                        continue

                    # skip if strange output_key values
                    if any_nans(output_key):
                        skip('nan in outputkey %s' % output_key)
                        continue

                    # skip if strange imortance values
                    if importance is None:
                        skip('importance is None')
                        continue
                    if any_nans(importance):
                        skip('NaN in importance %s %s' % (output_key, importance))
                        continue
                    if importance.importance is None:
                        skip('importance.importance is None')
                        continue

                    # accumulate rows to be used to create the DataFrame
                    for feature_name, feature_importance in importance.importance.iteritems():
                        data['feature_name'].append(feature_name)
                        data['feature_importance'].append(feature_importance)

                        data['trace_index'].append(output_key.trace_index)
                        data['model_spec'].append(str(output_key.model_spec))  # convert to str rep
                        data['effectivedatetime'].append(importance.effectivedatetime)
                        data['trade_type'].append(importance.trade_type)
                        count('importances')
            if control.arg.test and len(data['feature_name']) > 1000:
                print 'stopping input: TRACE'
                break

        print 'skipped %d input records' % len(skipped)
        for reason in sorted(skipped.keys()):
            count = skipped[reason]
            print '%70s: %d' % (reason, count)

        print 'counters'
        for counter in sorted(counters.keys()):
            count = counters[counter]
            print '%40s: %d' % (counter, count)

        importances = pd.DataFrame(data=data)
        print 'retained %d importance records' % len(importances)
        return importances

    # BODY STARTS HERE
    best_modelspecs = pass1(control)
    importances = pass2(best_modelspecs, control)
    return best_modelspecs, importances


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
    # produce reports on importances of features for the most accurate models
    best_modelspecs, importances = read_and_transform_importances(control)
    reports_best_features(best_modelspecs, importances, control)


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    print control.arg
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb

    main(sys.argv)
