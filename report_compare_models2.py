'''compare one set of output from fit_predict

INVOCATION
  python report_compare_models2.py {ticker} {cusip} {hpset} {--test} {--trace}  # all dates found in WORKING

EXAMPLES OF INVOCATION
 python report_compare_models2.py orcl 68389XAS4 grid2
'''

from __future__ import division

import argparse
import collections
import os
import pdb
import pandas as pd
import pprint
import random
import sys

import applied_data_science.dirutility

from applied_data_science.Bunch import Bunch
from applied_data_science.columns_table import columns_table
from applied_data_science.Logger import Logger
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


Data = collections.namedtuple(
    'Data',
    'query_index model_spec_str predicted_feature_name actual predicted absolute_error',
)


def column_def(column_name):
    return seven.reports.all_columns_2[column_name]


def write_the_lines(heading_lines, table_lines, path):
    with open(path, 'w') as f:
        f.writelines('%s\n' % line for line in heading_lines + table_lines)


def reports_mean_errors(data, path, arg):
    'write several reports on the mean errors'
    def make_mean_errors(data):
        'return (mean_errors, mean_errors_by_model_spec_str, mean_errors_by_predicted_feature_name'
        data_by_modelspecstr_predictedfeaturename = collections.defaultdict(list)
        data_by_modelspecstr = collections.defaultdict(list)
        data_by_predictedfeaturename = collections.defaultdict(list)
        data_by_queryindex_predictedfeaturename = collections.defaultdict(list)

        for x in data:
            data_by_modelspecstr_predictedfeaturename[(x.model_spec_str, x.predicted_feature_name)].append(x.absolute_error)
            data_by_modelspecstr[x.model_spec_str].append(x.absolute_error)
            data_by_predictedfeaturename[x.predicted_feature_name].append(x.absolute_error)
            data_by_queryindex_predictedfeaturename[(x.query_index, x.predicted_feature_name)].append(x.absolute_error)

        def mean(a_list):
            return float(sum(a_list)) / (1.0 * len(a_list))

        def sort_by_mean(d):
            return collections.OrderedDict(sorted(d.items(), key=lambda x: x[1]))

        def sort_by_key(d):
            return collections.OrderedDict(sorted(d.items(), key=lambda x: x[0]))

        return (
            sort_by_mean({
                k: mean(v)
                for k, v in data_by_modelspecstr_predictedfeaturename.iteritems()
            }),
            sort_by_mean({
                k: mean(v)
                for k, v in data_by_modelspecstr.iteritems()
            }),
            sort_by_mean({
                k: mean(v)
                for k, v in data_by_predictedfeaturename.iteritems()
            }),
            sort_by_key({
                k: mean(v)
                for k, v in data_by_queryindex_predictedfeaturename.iteritems()
            })
        )

    by_modelspec_predictedfeaturename, by_modelspecstr, by_predictedfeaturename, by_queryindex_predictedfeaturename = make_mean_errors(data)

    headings = [
        'Means Absolute Error Sorted by Increasing Mean Absolute Error',
        'Including all prints for ticker %s cusip %s hpset %s' % (arg.ticker, arg.cusip, arg.hpset),
        ''
    ]

    write_the_lines(
        headings,
        columns_table(
            (column_def('model_spec'), column_def('predicted_feature'), column_def('mean_absolute_error')),
            [(k[0], k[1], v) for k, v in by_modelspec_predictedfeaturename.iteritems()],
        ),
        path['out_accuracy_modelspec_predictedfeaturename'],
    )
    write_the_lines(
        headings,
        columns_table(
            (column_def('model_spec'), column_def('mean_absolute_error')),
            [(k, v) for k, v in by_modelspecstr.iteritems()],
        ),
        path['out_accuracy_modelspec'],
    )
    write_the_lines(
        headings,
        columns_table(
            (column_def('predicted_feature'), column_def('mean_absolute_error')),
            [(k, v) for k, v in by_predictedfeaturename.iteritems()],
        ),
        path['out_accuracy_predictedfeaturename'],
    )
    write_the_lines(
        headings,
        columns_table(
            (column_def('query_index'), column_def('predicted_feature'), column_def('mean_absolute_error')),
            [(k[0], k[1], v) for k, v in by_queryindex_predictedfeaturename.iteritems()],
        ),
        path['out_accuracy_queryindex_predictedfeaturename'],
    )


def read_and_transform_input(control):
    'return [Data]'
    data = []
    for in_file_path in control.path['in_files']:
        if os.path.getsize(in_file_path) == 0:
            print 'skipping empty file: %s' % in_file_path
            continue
        predictions = pd.read_csv(
            in_file_path,
            index_col=[0, 1, 2],
        )
        print 'read %d predictions from file %s' % (len(predictions), in_file_path.split('\\')[-2])
        n_has_nan = 0
        for row_index, row_data in predictions.iterrows():
            has_nan = False
            # exclude records with any NaN values
            for series_index, series_value in row_data.iteritems():
                if series_value != series_value:
                    has_nan = True
                    break
            if has_nan:
                n_has_nan += 1
            else:
                data.append(Data(
                    query_index=row_data['query_index'],
                    model_spec_str=row_data['model_spec_str'],
                    predicted_feature_name=row_data['predicted_feature_name'],
                    actual=row_data['actual'],
                    predicted=row_data['predicted'],
                    absolute_error=abs(row_data['actual'] - row_data['predicted']),
                ))
        if control.arg.test:
            print 'stopping reading, because testing'
            break
    print
    print 'read %d records from %d input files' % (len(data), len(control.path['in_files']))
    print 'skipped %d records because at least one field was missing (NaN)' % n_has_nan
    return data


def check_data(data):
    'stop if a problem with the input data'
    index_counts = collections.Counter()

    def count(x):
        index_counts[x.query_index] += 1

    map(count, data)
    print 'counts by index'
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
    data = read_and_transform_input(control)
    check_data(data)

    # produce each report or set of reports
    reports_mean_errors(
        data,
        control.path,
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
