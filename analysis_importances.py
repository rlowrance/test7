'''analyze the importances using the normalized weights from forming an ensemble model

INVOCATION
  python analyze_importances.py {test_train_output_location} [--debug] [--test] [--trace]
where
 test_train_location is {midpredictor, dev} tells where to find the output of the test_train program
  dev  --> its in .../Dropbox/data/7chord/7chord-01/working/test_train/
  prod --> its in .../Dropbox/MidPredictor/output/
 start_predictions: YYYY-MM-DD is the first date on which we attempt to test and train
 stop_predictions: YYYY-MM-DD is the last date on which we attempt to test and train
 --debug means to call pdb.set_trace() instead of raisinng an exception, on calls to logging.critical()
   and logging.error()
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
  python anayze_importances.py 2016-01-01 2017-12-31 --debug

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''

from __future__ import division

# import abc
import argparse
import copy
import csv
import collections
import datetime
import math
import os
import pdb
from pprint import pprint
import random
import sys

import applied_data_science.debug
import applied_data_science.dirutility
import applied_data_science.lower_priority
import applied_data_science.pickle_utilities

# from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.accumulators
import seven.arg_type
import seven.build
import seven.Event
import seven.EventAttributes
import seven.event_readers
import seven.HpGrids
import seven.logging
import seven.make_event_attributes
import seven.models2
import seven.read_csv
import seven.WalkTestTrainOutputDirectories

pp = pprint
csv
collections
datetime


class Control(object):
    def __init__(self, arg, path, random_seed, timer):
        self.arg = arg
        self.path = path
        self.random_seed = random_seed
        self.timer = timer

    def __repr__(self):
        return 'Control(arg=%s, len(path)=%d, random_seed=%f, timer=%s)' % (
            self.arg,
            len(self.path),
            self.random_seed,
            self.timer,
        )

    def new_with_path(self, new_logical_name, new_location):
        result = copy.copy(self)
        result.path[new_logical_name] = new_location
        return result

    @classmethod
    def make_control(cls, argv):
        'return a Control'
        parser = argparse.ArgumentParser()
        parser.add_argument('test_train_output_location')
        parser.add_argument('start_predictions', type=seven.arg_type.date)
        parser.add_argument('stop_predictions', type=seven.arg_type.date)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--trace', action='store_true')

        arg = parser.parse_args(argv[1:])

        assert arg.test_train_output_location in ('dev', 'prod')
        assert arg.start_predictions <= arg.stop_predictions

        if arg.trace:
            pdb.set_trace()
        if arg.debug:
            # logging.error() and logging.critial() call pdb.set_trace() instead of raising an exception
            seven.logging.invoke_pdb = True

        random_seed = 123
        random.seed(random_seed)

        paths = seven.build.analysis_importances(
            arg.test_train_output_location,
            arg.start_predictions,
            arg.stop_predictions,
            debug=arg.debug,
            test=arg.test,
            trace=arg.trace,
        )
        applied_data_science.dirutility.assure_exists(paths['dir_out'])

        timer = Timer()

        control = cls(
            arg=arg,
            path=paths,
            random_seed=random_seed,
            timer=timer,
        )
        return control


class Importances(object):
    def __init__(self, control, security_master):
        self._control = control
        self._security_master = security_master

        # the main output of visit_...
        self._importance_rows = []  # List[ImportanceRow] from all the input files

        # these are for reporting only
        self._no_files = []
        self._rows_found = {}  # Dict[path, int]
        self._visit_counter = collections.Counter()

    def __repr__(self):
        return 'Importances(%d importance rows)' % len(self._expert_rows)

    def report_files(self):
        print '\n******************\nreport on importances.csv files'
        print 'counters'
        for k, v in self._visit_counter.iteritems():
            print '%-30s: %d' % (k, v)

        print
        print 'directories found that did no have an importances file in them'
        for path in sorted(self._no_files):
            print ' %s' % path

        print
        print 'importances files found with no rows in them'
        for path in sorted(self._rows_found.keys()):
            if self._rows_found[path] == 0:
                print ' %s' % path

    def report_mean_importance(self, path):
        importances = sorted(
            self._mean_importance(),
            key=lambda (model_family, feature_name, mean_abs_importance): (model_family, mean_abs_importance, feature_name),
            reverse=True
        )
        with open(path, 'wb') as f:
            writer = csv.DictWriter(
                f,
                ['model_family', 'feature_name', 'mean_absolute_importance'],
                lineterminator='\n',
            )
            writer.writeheader()
            for model_family, feature_name, mean_absolute_importance in importances:
                writer.writerow({
                    'model_family': model_family,
                    'feature_name': feature_name,
                    'mean_absolute_importance': mean_absolute_importance,
                })

    def report_mean_importance_by_date(self, path):
        importances = sorted(
            self._mean_importance_by_date(),
            key=lambda (date, model_family, feature_name, mean_abs_importance): (date, model_family, mean_abs_importance, feature_name),
            reverse=True
        )
        with open(path, 'wb') as f:
            writer = csv.DictWriter(
                f,
                ['date', 'model_family', 'feature_name', 'mean_absolute_importance'],
                lineterminator='\n',
            )
            writer.writeheader()
            for date, model_family, feature_name, mean_absolute_importance in importances:
                writer.writerow({
                    'date': date,
                    'model_family': model_family,
                    'feature_name': feature_name,
                    'mean_absolute_importance': mean_absolute_importance,
                })

    def report_mean_weights_by_date(self, k, path_all, path_top_k, verbose=False):
        mean_weights_by_date = sorted(
            self._mean_weights_by_date(),
            key=lambda (date, expert, mean_weight): (date, mean_weight, expert),
            reverse=True,
        )
        if verbose:
            print '\n******************\nreort on mean weights of importances by date'
        with open(path_all, 'wb') as f_all:
            with open(path_top_k, 'wb') as f_top_k:
                header = ['date', 'expert', 'mean_weight']
                writer_all = csv.DictWriter(
                    f_all,
                    header,
                    lineterminator='\n',
                )
                writer_top_k = csv.DictWriter(
                    f_top_k,
                    header,
                    lineterminator='\n',
                )
                writer_all.writeheader()
                writer_top_k.writeheader()
                current_date = datetime.date(1, 1, 1)
                lines_since_new_cusip = 0
                for date, expert, mean_weight in mean_weights_by_date:
                    if date != current_date:
                        lines_since_new_cusip = 0
                        current_date = date
                    lines_since_new_cusip += 1
                    if verbose:
                        print '%9s %-30s %10.6f' % (date, expert, mean_weight)
                    row = {
                        'date': date,
                        'expert': expert,
                        'mean_weight': mean_weight,
                    }
                    writer_all.writerow(row)
                    if lines_since_new_cusip <= k:
                        writer_top_k.writerow(row)

    def report_mean_weights_by_issuer(self, k, path_all, path_top_k):
        mean_weights_by_issuer = sorted(
            self._mean_weights_by_issuer(),
            key=lambda (issuer, expert, mean_weight): (issuer, mean_weight, expert),
            reverse=True,
        )
        with open(path_all, 'wb') as f_all:
            with open(path_top_k, 'wb') as f_top_k:
                header = ['issuer', 'expert', 'mean_weight']
                writer_all = csv.DictWriter(
                    f_all,
                    header,
                    lineterminator='\n',
                )
                writer_top_k = csv.DictWriter(
                    f_top_k,
                    header,
                    lineterminator='\n',
                )
                writer_all.writeheader()
                writer_top_k.writeheader()
                current_issuer = ''
                lines_since_new_issuer = 0
                for issuer, expert, mean_weight in mean_weights_by_issuer:
                    if issuer != current_issuer:
                        lines_since_new_issuer = 0
                        current_issuer = issuer
                    lines_since_new_issuer += 1
                    row = {
                        'issuer': issuer,
                        'expert': expert,
                        'mean_weight': mean_weight,
                    }
                    writer_all.writerow(row)
                    if lines_since_new_issuer <= k:
                        writer_top_k.writerow(row)

    def report_mean_weights_by_issuer_cusip(self, k, path_all, path_top_k):
        mean_weights_by_issuer_cusip = sorted(
            self._mean_weights_by_issuer_cusip(),
            key=lambda (issuer, cusip, expert, mean_weight): (issuer, cusip, mean_weight, expert),
            reverse=True,
        )
        with open(path_all, 'wb') as f_all:
            with open(path_top_k, 'wb') as f_top_k:
                header = ['issuer', 'cusip', 'expert', 'mean_weight']
                writer_all = csv.DictWriter(
                    f_all,
                    header,
                    lineterminator='\n',
                )
                writer_top_k = csv.DictWriter(
                    f_top_k,
                    header,
                    lineterminator='\n',
                )
                writer_all.writeheader()
                writer_top_k.writeheader()
                current_issuer = ''
                current_cusip = ''
                lines_since_new_cusip = 0
                for issuer, cusip, expert, mean_weight in mean_weights_by_issuer_cusip:
                    if issuer != current_issuer or cusip != current_cusip:
                        lines_since_new_cusip = 0
                        current_issuer = issuer
                        current_cusip = cusip
                    lines_since_new_cusip += 1
                    row = {
                        'issuer': issuer,
                        'cusip': cusip,
                        'expert': expert,
                        'mean_weight': mean_weight,
                    }
                    writer_all.writerow(row)
                    if lines_since_new_cusip <= k:
                        writer_top_k.writerow(row)

    def visit_test_train_output_directory(self, directory_path, invocation_parameters, verbose=True):
        'update self._importances_rows with info in {directory_path}/importances.csv'
        def date(s):
            year, month, day = s.split('-')
            return datetime.date(int(year), int(month), int(day))

        def within_start_stop_dates():
            'is the range of dates in the file within those selected by the invocation args?'
            return (
                date(invocation_parameters['start_predictions']) >= self._control.arg.start_predictions and
                date(invocation_parameters['stop_predictions']) <= self._control.arg.stop_predictions)

        self._visit_counter['directories_visited'] += 1
        if not within_start_stop_dates():
            if verbose:
                print 'skipping', directory_path
            self._visit_counter['not within start stop dates'] += 1
            return
        else:
            if verbose:
                print 'examining', directory_path
            self._visit_counter['within start stop dates'] += 1
        path = os.path.join(directory_path, 'importances.csv')
        if os.path.isfile(path):
            with open(path) as f:
                csv_reader = csv.DictReader(f)
                n_rows = 0
                for row in csv_reader:
                    n_rows += 1
                    self._importance_rows.append(ImportanceRow(row, self._security_master))
                self._rows_found[path] = n_rows
        else:
            self._no_files.append(dir)

    def _mean_importance(self):
        'return List[Tuple[model_family, feature_name, mean_absolute_importance]]'
        dd = collections.defaultdict
        total_importance = dd(lambda: collections.defaultdict(float))
        count = dd(lambda: collections.defaultdict(int))
        for row in self._importance_rows:
            total_importance[row.model_family()][row.feature_name()] += abs(row.importance())
            count[row.model_family()][row.feature_name()] += 1
        result = []
        for model_family, d in total_importance.iteritems():
            for feature_name, importance in d.iteritems():
                item = (
                    model_family,
                    feature_name,
                    total_importance[model_family][feature_name] / count[model_family][feature_name],
                )
                result.append(item)
        return result

    def _mean_importance_by_date(self):
        'return List[Tuple[date, model_family, feature_name, mean_absolute_importance]]'
        dd = collections.defaultdict
        total_importance = dd(lambda: dd(lambda: dd(float)))
        count = dd(lambda: dd(lambda: dd(int)))
        for row in self._importance_rows:
            total_importance[row.date()][row.model_family()][row.feature_name()] += abs(row.importance())
            count[row.date()][row.model_family()][row.feature_name()] += 1
        result = []
        for date, d1 in total_importance.iteritems():
            for model_family, d2 in d1.iteritems():
                for feature_name, importance in d2.iteritems():
                    item = (
                        date,
                        model_family,
                        feature_name,
                        total_importance[date][model_family][feature_name] / count[date][model_family][feature_name],
                    )
                    result.append(item)
        return result

    def _mean_weights_by_date(self):
        'return List[Tuple[date, expert, mean weight]]'
        total_weights = collections.defaultdict(lambda: collections.defaultdict(float))
        counts = collections.defaultdict(lambda: collections.defaultdict(int))
        for expert_row in self._expert_rows:
            date = expert_row.date()
            expert = expert_row.expert()
            weight = expert_row.weight()
            total_weights[date][expert] += weight
            counts[date][expert] += 1
        result = []
        for date, experts_totalweights in total_weights.iteritems():
            for expert, total_weight in experts_totalweights.iteritems():
                mean_weight = total_weight / counts[date][expert]
                result.append((date, expert, mean_weight))
        return result

    def _mean_weights_by_issuer(self):
        'return List[Tuple[issuer, expert, mean weight]]'
        total_weights = collections.defaultdict(lambda: collections.defaultdict(float))
        counts = collections.defaultdict(lambda: collections.defaultdict(int))
        for expert_row in self._expert_rows:
            issuer = expert_row.issuer()
            expert = expert_row.expert()
            weight = expert_row.weight()
            total_weights[issuer][expert] += weight
            counts[issuer][expert] += 1
        result = []
        for issuer, experts_totalweights in total_weights.iteritems():
            for expert, total_weight in experts_totalweights.iteritems():
                mean_weight = total_weight / counts[issuer][expert]
                result.append((issuer, expert, mean_weight))
        return result

    def _mean_weights_by_issuer_cusip(self):
        'return List[Tuple[issuer, cusip, expert, mean weight]]'
        total_weights = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
        counts = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
        for expert_row in self._expert_rows:
            issuer = expert_row.issuer()
            cusip = expert_row.cusip()
            expert = expert_row.expert()
            weight = expert_row.weight()
            total_weights[issuer][cusip][expert] += weight
            counts[issuer][cusip][expert] += 1
        result = []
        for issuer, d1 in total_weights.iteritems():
            for cusip, d2 in d1.iteritems():
                for expert, total_weight in d2.iteritems():
                    mean_weight = total_weight / counts[issuer][cusip][expert]
                    result.append((issuer, cusip, expert, mean_weight))
        return result

    def _sort_dict_by_value(self, d, reverse=False):
        'return List[tuples(key, value)]'
        if False:
            # usage
            for key, value in self._sorted(d):
                pass
        result = sorted(
            d.iteritems(),
            key=lambda (k, v): (v, k),
            reverse=reverse,
        )
        return result


class ImportanceRow(object):
    def __init__(self, row, security_master):
        # Note: in order by column in the csv file
        self._datetime = self._extract_datetime(row['simulated_datetime'])
        self._model_family = row['importances_model_family']
        self._feature_name = row['importances_feature_name']
        self._importance = float(row['importances_weight'])

    def __repr__(self):
        return 'ImportanceRow(%s, %s, %s, %0.6f)' % (
            self._datetime,
            self._model_family,
            self._feature_name,
            self._importance,
            )

    def cusip(self):
        return self._cusip

    def date(self):
        return self._datetime.date()

    def datetime(self):
        return self._datetime

    def feature_name(self):
        return self._feature_name

    def importance(self):
        return self._importance

    def model_family(self):
        return self._model_family

    def _extract_datetime(self, event_datetime):
        date, time = event_datetime.split(' ')
        year, month, day = date.split('-')
        hour, minute, seconds_microseconds_str = time.split(':')
        seconds_microseconds = float(seconds_microseconds_str)
        microseconds, seconds = math.modf(seconds_microseconds)
        return datetime.datetime(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(seconds),
            int(microseconds * 1e6),
            )


class SecMaster(object):
    def __init__(self, path):
        self._issuer_for_cusip = self._read(path)  # Dict[cusip, issuer]

    def get_issuer(self, row):
        return self._issuer_for_cusip[row['cusip']]

    def _read(self, path):
        'build self._table'
        print 'reading security master from %s' % path
        result = {}  # Dict[cusip, issuer]
        n_cusips = 0
        with open(path) as f:
            dict_reader = csv.DictReader(f)
            for row in dict_reader:
                if row['CUSIP'] in result:
                    print 'cusip already defined in security master'
                    pp(row)
                    seven.accumulatorslogging.critical('malformed secmaster %s' % path)
                else:
                    result[row['CUSIP']] = row['ticker']
                    n_cusips += 1
        print 'read %d records from secmaster file at %s' % (n_cusips, path)
        return result


def do_work(control):
    # applied_data_science.lower_priority.lower_priority()
    # find all the importances.csv files (they have the accuracy data on importances)
    importances = Importances(
        control,
        SecMaster(control.path['in_secmaster']),
        )

    # visit each expert.csv file and extract its content
    walker = seven.WalkTestTrainOutputDirectories.WalkTestTrainOutputDirectories(control.path['dir_in'])
    walker.walk(importances.visit_test_train_output_directory)

    importances.report_files()
    importances.report_mean_importance(control.path['out_mean_importance'])
    importances.report_mean_importance_by_date(control.path['out_mean_importance_by_date'])

    return None


def main(argv):
    control = Control.make_control(argv)
    sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    # print control
    print control.arg
    print 'done'
    return


if __name__ == '__main__':
    main(sys.argv)
