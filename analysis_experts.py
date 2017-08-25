'''analyze accuracy of the experts using the normalized weights from forming an ensemble model

INVOCATION
  python analyze_experts.py {test_train_output_location} start_predictions stop_predictions [--debug] [--test] [--trace]
where
 test_train_location is {midpredictor, dev} tells where to find the output of the test_train program
  'dev'  --> its in .../Dropbox/data/7chord/7chord-01/working/test_train/
  'prod' --> its in .../Dropbox/MidPredictor/output/
  other  --> its in {other}
 start_predictions: YYYY-MM-DD is the first date on which we attempt to test and train
 stop_predictions: YYYY-MM-DD is the last date on which we attempt to test and train
 --debug means to call pdb.set_trace() instead of raisinng an exception, on calls to logging.critical()
   and logging.error()
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
  python anayze_experts.py dev 2016-01-01 2017-12-31--debug
  python anayze_experts.py prod 2016-01-01 2016-06-30--debug
  python anayze_experts.py prod 2016-07-01 2016-12-31--debug
  python anayze_experts.py prod 2017-01-01 2017-12-31--debug
  python analyze_experts.py /home/ubuntu/data/fit_predict/test_train 2016-01-01 2017-12-31

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''



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

        assert arg.start_predictions <= arg.stop_predictions

        if arg.trace:
            pdb.set_trace()
        if arg.debug:
            # logging.error() and logging.critial() call pdb.set_trace() instead of raising an exception
            seven.logging.invoke_pdb = True

        random_seed = 123
        random.seed(random_seed)

        paths = seven.build.analysis_experts(
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


class Experts(object):
    def __init__(self, control, security_master):
        self._control = control
        self._security_master = security_master

        # the main output of visit_...
        self._expert_rows = []  # List[ExpertRow] from all the input files

        # these are for reporting only
        self._no_content = []
        self._no_files = []
        self._visit_counter = collections.Counter()

    def __repr__(self):
        return 'Experts(%d expert rows)' % len(self._expert_rows)

    def report_files(self):
        print('\n******************\nreport on visiting experts.csv files\n')
        print('counters from visiting files')
        for k, v in self._visit_counter.items():
            print('%-30s: %6d' % (k, v))

        print('\ndirectories without an experts.csv file:')
        for no_file in self._no_files:
            print(' %s' % no_file)

        print('\nexperts.csv files without any content (possibly no predictions were made)')
        for path in self._no_content:
            print(' %s' % path)

        print()
        print('# directories without an experts.csv file', len(self._no_files))
        print('# experts.csv files without any rows', len(self._no_content))

        print('\ntotal number of expert accuracies found: %d' % len(self._expert_rows))

    def report_mean_weights(self, path):
        mean_weights = sorted(
            self._mean_weights(),
            key=lambda expert_mean_weight: (expert_mean_weight[1], expert_mean_weight[0]),
            reverse=True,
        )
        with open(path, 'wb') as f:
            writer = csv.DictWriter(
                f,
                ['expert', 'mean_weight'],
                lineterminator='\n',
            )
            writer.writeheader()
            for expert, mean_weight in mean_weights:
                writer.writerow({
                    'expert': expert,
                    'mean_weight': mean_weight,
                })

    def report_mean_weights_by_date(self, k, path_all, path_top_k, verbose=False):
        mean_weights_by_date = sorted(
            self._mean_weights_by_date(),
            key=lambda date_expert_mean_weight: (date_expert_mean_weight[0], date_expert_mean_weight[2], date_expert_mean_weight[1]),
            reverse=True,
        )
        if verbose:
            print('\n******************\nreort on mean weights of experts by date')
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
                        print('%9s %-30s %10.6f' % (date, expert, mean_weight))
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
            key=lambda issuer_expert_mean_weight: (issuer_expert_mean_weight[0], issuer_expert_mean_weight[2], issuer_expert_mean_weight[1]),
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
            key=lambda issuer_cusip_expert_mean_weight: (issuer_cusip_expert_mean_weight[0], issuer_cusip_expert_mean_weight[1], issuer_cusip_expert_mean_weight[3], issuer_cusip_expert_mean_weight[2]),
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
        'update self._experts_rows with info in {directory_path}/experts.csv'
        self._visit_counter['directories visited'] += 1
        print('visiting', directory_path)
        path = os.path.join(directory_path, 'experts.csv')
        if os.path.isfile(path):
            self._visit_counter['files examined'] += 1
            with open(path) as f:
                csv_reader = csv.DictReader(f)
                n_rows = 0
                for row in csv_reader:
                    n_rows += 1
                    expert_row = ExpertRow(row, self._security_master)
                    self._expert_rows.append(expert_row)
                    self._visit_counter['rows saved'] += 1
                if n_rows == 0:
                    self._no_content.append(path)
                    self._visit_counter['experts.csv files without data rows'] += 1
        else:
            self._no_files.append(directory_path)
            self._visit_counter['directories without an expert.csv file'] += 1

    def _mean_weights(self):
        'return List[expert, mean weight]'
        total_weights = collections.defaultdict(float)  # Dict[expert, float]
        counts = collections.defaultdict(int)           # Dict[expert, int]
        for expert_row in self._expert_rows:
            expert = expert_row.expert()
            total_weights[expert] += expert_row.weight()
            counts[expert] += 1
        result = []
        for expert, total_weight in total_weights.items():
            result.append((expert, total_weight / counts[expert]))
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
        for date, experts_totalweights in total_weights.items():
            for expert, total_weight in experts_totalweights.items():
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
        for issuer, experts_totalweights in total_weights.items():
            for expert, total_weight in experts_totalweights.items():
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
        for issuer, d1 in total_weights.items():
            for cusip, d2 in d1.items():
                for expert, total_weight in d2.items():
                    mean_weight = total_weight / counts[issuer][cusip][expert]
                    result.append((issuer, cusip, expert, mean_weight))
        return result

    def _sort_dict_by_valueOLD(self, d, reverse=False):
        'return List[tuples(key, value)]'
        if False:
            # usage
            for key, value in self._sorted(d):
                pass
        result = sorted(
            iter(d.items()),
            key=lambda k_v: (k_v[1], k_v[0]),
            reverse=reverse,
        )
        return result


class ExpertRow(object):
    def __init__(self, row, security_master):
        self._cusip = row['cusip']
        self._issuer = security_master.get_issuer(row)
        self._date = self._extract_date(row['event_datetime'])
        self._expert = row['last_trained_expert_model_spec']
        self._weight = float(row['expert_normalized_weight'])

    def __repr__(self):
        return 'ExpertRow(%s, %s, %s, %s, %0.6f)' % (
            self._expert,
            self._issuer,
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

    def issuer(self):
        return self._issuer

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


class SecMaster(object):
    def __init__(self, path):
        self._issuer_for_cusip = self._read(path)  # Dict[cusip, issuer]

    def get_issuer(self, row):
        return self._issuer_for_cusip[row['cusip']]

    def _read(self, path):
        'build self._table'
        print('reading security master from %s' % path)
        result = {}  # Dict[cusip, issuer]
        n_cusips = 0
        with open(path) as f:
            dict_reader = csv.DictReader(f)
            for row in dict_reader:
                if row['CUSIP'] in result:
                    print('cusip already defined in security master')
                    pp(row)
                    seven.accumulatorslogging.critical('malformed secmaster %s' % path)
                else:
                    result[row['CUSIP']] = row['ticker']
                    n_cusips += 1
        print('read %d records from secmaster file at %s' % (n_cusips, path))
        return result


def do_work(control):
    # applied_data_science.lower_priority.lower_priority()
    # find all the experts.csv files (they have the accuracy data on experts)
    experts = Experts(
        control,
        SecMaster(control.path['in_secmaster']),
        )

    # visit each expert.csv file and extract its content
    walker = seven.WalkTestTrainOutputDirectories.WalkTestTrainOutputDirectories(control.path['dir_in'])
    walker.walk_prediction_dates_between(
        visit=experts.visit_test_train_output_directory,
        start_date=control.arg.start_predictions,
        stop_date=control.arg.stop_predictions,
    )

    experts.report_files()
    experts.report_mean_weights(control.path['out_mean_weights'])
    experts.report_mean_weights_by_date(
        k=10,
        path_all=control.path['out_mean_weights_by_date'],
        path_top_k=control.path['out_mean_weights_by_date_top_k'],
        )
    experts.report_mean_weights_by_issuer(
        k=10,
        path_all=control.path['out_mean_weights_by_issuer'],
        path_top_k=control.path['out_mean_weights_by_issuer_top_k'],
        )
    experts.report_mean_weights_by_issuer_cusip(
        k=10,
        path_all=control.path['out_mean_weights_by_issuer_cusip'],
        path_top_k=control.path['out_mean_weights_by_issuer_cusip_top_k'],
        )

    return None


def main(argv):
    control = Control.make_control(argv)
    sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
    print(control)
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.arg.test:
        print('DISCARD OUTPUT: test')
    # print control
    print(control.arg)
    print('done')
    return


if __name__ == '__main__':
    main(sys.argv)
