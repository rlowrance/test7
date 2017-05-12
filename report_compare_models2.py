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
    'query_index model_spec_str target_feature_name actual predicted absolute_error',
)


def column_def(column_name):
    return seven.reports.all_columns_2[column_name]


def write_the_lines(heading_lines, table_lines, path, verbose=True):
    verbose = True
    if verbose:
        print 'lines written to %s' % path
        for line in heading_lines + table_lines:
            print line
    with open(path, 'w') as f:
        f.writelines('%s\n' % line for line in heading_lines + table_lines)


def reports_mean_errors(data, path, arg):
    'write several reports on the mean errors'
    def sort_by_value(d):
        return collections.OrderedDict(sorted(d.items(), key=lambda x: x[1]))

    def sort_by_key(d):
        return collections.OrderedDict(sorted(d.items(), key=lambda x: x[0]))

    def make_summary(data):
        'return dict of summaries of the data'
        counter_modelspecstr = collections.Counter()
        counter_queryindex = collections.Counter()
        counter_targetfeaturename = collections.Counter()
        # one way
        data_by_modelspecstr = collections.defaultdict(list)
        data_by_targetfeaturename = collections.defaultdict(list)
        data_by_queryindex = collections.defaultdict(list)
        # two ways
        data_by_modelspecstr_targetfeaturename = collections.defaultdict(lambda: collections.defaultdict(list))
        data_by_targetfeaturename_modelspecstr = collections.defaultdict(lambda: collections.defaultdict(list))
        data_by_queryindex_targetfeaturename = collections.defaultdict(lambda: collections.defaultdict(list))

        for x in data:
            # counters
            counter_modelspecstr[x.model_spec_str] += 1
            counter_queryindex[x.query_index] += 1
            counter_targetfeaturename[x.target_feature_name] += 1
            # one way
            data_by_modelspecstr[x.model_spec_str].append(x.absolute_error)
            data_by_targetfeaturename[x.target_feature_name].append(x.absolute_error)
            data_by_queryindex[x.query_index].append(x.absolute_error)
            # two ways
            data_by_modelspecstr_targetfeaturename[x.model_spec_str][x.target_feature_name].append(x.absolute_error)
            data_by_targetfeaturename_modelspecstr[x.target_feature_name][x.model_spec_str].append(x.absolute_error)
            data_by_queryindex_targetfeaturename[x.query_index][x.target_feature_name].append(x.absolute_error)

        def mean(a_list):
            return float(sum(a_list)) / (1.0 * len(a_list))

        def make_means1(d):
            return {
                k: mean(v)
                for k, v in d.iteritems()
            }

        def make_means2(d):
            result = {}
            for topk, topv in d.iteritems():
                result[topk] = make_means1(topv)
            return result

        return {
            'counter_modelspecstr': counter_modelspecstr,
            'counter_queryindex': counter_queryindex,
            'counter_targetfeaturename': counter_targetfeaturename,

            'means_by_modelspecstr': make_means1(data_by_modelspecstr),
            'means_by_targetfeaturename': make_means1(data_by_targetfeaturename),
            'means_by_queryindex': make_means1(data_by_queryindex),

            'means_by_modelspecstr_targetfeaturename': make_means2(data_by_modelspecstr_targetfeaturename),
            'means_by_targetfeaturename_modelspecstr': make_means2(data_by_targetfeaturename_modelspecstr),
            'means_by_queryindex_targetfeaturename': make_means2(data_by_queryindex_targetfeaturename),
        }

    summary = make_summary(data)

    headings = [
        'Mean Absolute Errors',
        'Including all predictions for ticker %s cusip %s hpset %s' % (arg.ticker, arg.cusip, arg.hpset),
        'Predicting %d targets using %d hyperparameter settings for %d prints' % (
            len(summary['counter_targetfeaturename']),
            len(summary['counter_modelspecstr']),
            len(summary['counter_queryindex']),
        ),
        ' ',
    ]

    # one  way
    def write_1_way(summary_name, column_name, path_name):
        write_the_lines(
            headings,
            columns_table(
                (column_def(column_name), column_def('mean_absolute_error')),
                [
                    (k, v)
                    for k, v in sort_by_value(summary[summary_name]).iteritems()
                ]
            ),
            path[path_name],
        )
    write_1_way(
        'means_by_modelspecstr',
        'model_spec',
        'out_accuracy_modelspec',
    )
    write_1_way(
        'means_by_targetfeaturename',
        'target_feature',
        'out_accuracy_targetfeaturename',
    )
    write_1_way(
        'means_by_queryindex',
        'query_index',
        'out_accuracy_queryindex',
    )

    # two ways
    def write_2_ways(summary_name, column_names, path_name):
        write_the_lines(
            headings,
            columns_table(
                (column_def(column_names[0]), column_def(column_names[1]), column_def('mean_absolute_error')),
                [
                    (k1, k2, v2)
                    for k1, v1 in sort_by_key(summary[summary_name]).iteritems()
                    for k2, v2 in sort_by_value(v1).iteritems()
                ],
            ),
            path[path_name],
        )
    write_2_ways(
        'means_by_modelspecstr_targetfeaturename',
        ('model_spec', 'target_feature'),
        'out_accuracy_modelspec_targetfeaturename',
    )
    write_2_ways(
        'means_by_targetfeaturename_modelspecstr',
        ('target_feature', 'model_spec'),
        'out_accuracy_targetfeaturename_modelspecstr',
    )
    write_2_ways(
        'means_by_queryindex_targetfeaturename',
        ('query_index', 'target_feature'),
        'out_accuracy_queryindex_targetfeaturename',
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
                    query_index=row_data['id_query_index'],
                    model_spec_str=row_data['id_modelspec_str'],
                    target_feature_name=row_data['id_target_feature_name'],
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
    for x in data:
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
