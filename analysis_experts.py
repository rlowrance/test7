'''analyze accuracy of the experts using the normalized weights from forming an ensemble model

INVOCATION
  python analyze_experts.py {test_train_output_location} [--debug] [--test] [--trace]
where
test_train_location is {midpredictor, dev} tells where to find the output of the test_train program
  dev  --> its in .../Dropbox/data/7chord/7chord-01/working/test_train/
  prod --> its in .../Dropbox/MidPredictor/output/
 --debug means to call pdb.set_trace() instead of raisinng an exception, on calls to logging.critical()
   and logging.error()
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
  python anayze_experts.py --debug

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
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--trace', action='store_true')

        arg = parser.parse_args(argv[1:])

        assert arg.test_train_output_location in ('dev', 'prod')

        if arg.trace:
            pdb.set_trace()
        if arg.debug:
            # logging.error() and logging.critial() call pdb.set_trace() instead of raising an exception
            seven.logging.invoke_pdb = True

        random_seed = 123
        random.seed(random_seed)

        paths = seven.build.analysis_experts(
            arg.test_train_output_location,
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


class Experts(object):
    def __init__(self, control):
        self._control = control

        self._expert_rows = []  # List[ExpertRow] from all the input files
        self._rows_found = {}   # Dict[path, int]
        self._no_files = []     # List[str]

    def __repr__(self):
        return 'Experts()'

    def update_with_file(self, dir, start_predictions, stop_predictions):
        'update self._experts_rows with info in {dir}/experts.csv'
        path = os.path.join(dir, 'experts.csv')
        if os.path.isfile(path):
            with open(path) as f:
                csv_reader = csv.DictReader(f)
                n_rows = 0
                for row in csv_reader:
                    n_rows += 1
                    self._expert_rows.append(ExpertRow(row))
                self._rows_found[path] = n_rows
        else:
            self._no_files.append(dir)

    def report_files(self):
        print '\n******************\nreport on experts.csv files read\n'
        print 'directories without an experts.csv file:'
        for no_file in self._no_files:
            print ' %s' % no_file

        print 'experts.csv files without any content'
        grand_total = 0
        for path, n_rows in self._rows_found.iteritems():
            if n_rows == 0:
                print ' %s' % path
            grand_total += n_rows

        print 'total number of expert accuracies found: %d' % grand_total

    def report_mean_weights_by_date(self, k, path_all, path_top_k, verbose=False):
        mean_weights_by_date = sorted(
            self._mean_weights_by_date(),
            key=lambda (cusip, expert, mean_weight): (cusip, mean_weight, expert),
            reverse=True,
        )
        if verbose:
            print '\n******************\nreort on mean weights of experts by date'
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

    def report_mean_weights(self, path, verbose=False):
        mean_weights = self._sort_dict_by_value(self._mean_weights(), reverse=False)
        if verbose:
            print '\n******************\nreort on mean weights of experts'
        with open(path, 'wb') as f:
            writer = csv.DictWriter(
                f,
                ['expert', 'mean_weight'],
                lineterminator='\n',
            )
            writer.writeheader()
            for expert, mean_weight in mean_weights:
                if verbose:
                    print '%-30s: %10.6f' % (expert, mean_weight)
                writer.writerow({
                    'expert': expert,
                    'mean_weight': mean_weight,
                })

    def report_mean_weights_by_cusip(self, k, path_all, path_top_k, verbose=False):
        mean_weights_by_cusip = sorted(
            self._mean_weights_by_cusip(),
            key=lambda (cusip, expert, mean_weight): (cusip, mean_weight, expert),
            reverse=True,
        )
        if verbose:
            print '\n******************\nreort on mean weights of experts by cusip'
        with open(path_all, 'wb') as f_all:
            with open(path_top_k, 'wb') as f_top_k:
                header = ['cusip', 'expert', 'mean_weight']
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
                current_cusip = ''
                lines_since_new_cusip = 0
                for cusip, expert, mean_weight in mean_weights_by_cusip:
                    if cusip != current_cusip:
                        lines_since_new_cusip = 0
                        current_cusip = cusip
                    lines_since_new_cusip += 1
                    if verbose:
                        print '%9s %-30s %10.6f' % (cusip, expert, mean_weight)
                    row = {
                        'cusip': cusip,
                        'expert': expert,
                        'mean_weight': mean_weight,
                    }
                    writer_all.writerow(row)
                    if lines_since_new_cusip <= k:
                        writer_top_k.writerow(row)

    def _mean_weights(self):
        'return Dict[expert, mean weight]'
        total_weights = collections.defaultdict(float)  # Dict[expert, float]
        counts = collections.defaultdict(int)           # Dict[expert, int]
        for expert_row in self._expert_rows:
            expert = expert_row.expert()
            total_weights[expert] += expert_row.weight()
            counts[expert] += 1
        result = {}  # Dict[expert, float]
        for expert, total_weight in total_weights.iteritems():
            result[expert] = total_weight / counts[expert]
        return result

    def _mean_weights_by_cusip(self):
        'return List[Tuple[cusip, expert, mean weight]]'
        total_weights = collections.defaultdict(lambda: collections.defaultdict(float))
        counts = collections.defaultdict(lambda: collections.defaultdict(int))
        for expert_row in self._expert_rows:
            cusip = expert_row.cusip()
            expert = expert_row.expert()
            weight = expert_row.weight()
            total_weights[cusip][expert] += weight
            counts[cusip][expert] += 1
        result = []
        for cusip, experts_totalweights in total_weights.iteritems():
            for expert, total_weight in experts_totalweights.iteritems():
                mean_weight = total_weight / counts[cusip][expert]
                result.append((cusip, expert, mean_weight))
        return result

    def _mean_weights_by_date(self):
        'return List[Tuple[cusip, expert, mean weight]]'
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


class ExpertRow(object):
    def __init__(self, row):
        self._cusip = row['cusip']
        self._date = self._extract_date(row['event_datetime'])
        self._expert = row['last_trained_expert_model_spec']
        self._weight = float(row['expert_normalized_weight'])

    def __repr__(self):
        return 'ExpertRow(%s, %s, %s, %0.6f)' % (
            self._expert,
            self._cusip,
            self._date,
            self._weight,
            )

    def cusip(self):
        return self._cusip

    def date(self):
        return self._date

    def expert(self):
        return self._expert

    def weight(self):
        return self._weight

    def _extract_date(self, event_datetime):
        date, time = event_datetime.split(' ')
        year, month, day = date.split('-')
        return datetime.date(
            int(year),
            int(month),
            int(day),
            )


class FindExperts(object):
    def __init__(self, experts):
        self._experts = experts

    def run(self, starting_dir):
        self._process_issuers(starting_dir)

    def _process_cusips(self, dir):
        for cusip in os.listdir(dir):
            self._process_targets(os.path.join(dir, cusip))

    def _process_hpsets(self, dir):
        for hpset in os.listdir(dir):
            self._process_start_eventss(os.path.join(dir, hpset))

    def _process_issuers(self, dir):
        for issuer in os.listdir(dir):
            self._process_cusips(os.path.join(dir, issuer))

    def _process_start_eventss(self, dir):
        for start_events in os.listdir(dir):
            self._process_start_predictionss(os.path.join(dir, start_events))

    def _process_start_predictionss(self, dir):
        for start_predictions in os.listdir(dir):
            self._process_stop_predictionss(os.path.join(dir, start_predictions), start_predictions)

    def _process_stop_predictionss(self, dir, start_predictions):
        for stop_predictions in os.listdir(dir):
            self._experts.update_with_file(
                dir=os.path.join(dir, stop_predictions),
                start_predictions=start_predictions,
                stop_predictions=stop_predictions,
            )

    def _process_targets(self, dir):
        for target in os.listdir(dir):
            self._process_hpsets(os.path.join(dir, target))


def do_work(control):
    # applied_data_science.lower_priority.lower_priority()
    # find all the experts.csv files (they have the accuracy data on experts)
    dir = control.path['dir_in']
    experts = Experts(control)
    find_experts = FindExperts(experts)
    find_experts.run(dir)

    experts.report_files()
    experts.report_mean_weights(control.path['out_mean_weights'])
    experts.report_mean_weights_by_cusip(
        k=10,
        path_all=control.path['out_mean_weights_by_cusip'],
        path_top_k=control.path['out_mean_weights_by_cusip_top_k'],
        )
    experts.report_mean_weights_by_date(
        k=10,
        path_all=control.path['out_mean_weights_by_date'],
        path_top_k=control.path['out_mean_weights_by_date_top_k'],
        )
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
