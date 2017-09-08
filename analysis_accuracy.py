'''analyze accuracy of the ensemble predictions

INVOCATION
  python analyze_accuracy.py {test_train_output_location} start_predictions stop_predictions [--debug] [--test] [--trace]
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
  python anayze_accuracy.py dev 2016-01-01 2017-12-31--debug
  python analyze_accuracy.py /home/ubuntu/data/fit_predict/test_train 2016-01-01 2017-12-31

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''


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
from typing import List

import seven.accumulators
import seven.arg_type
import seven.build
import seven.debug
import seven.dirutility
import seven.Event
import seven.EventAttributes
import seven.event_readers
import seven.HpGrids
import seven.Logger
import seven.logging
import seven.lower_priority
import seven.make_event_attributes
import seven.models2
import seven.read_csv
import seven.pickle_utilities
import seven.Timer
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

        paths = seven.build.analysis_accuracy(
            arg.test_train_output_location,
            arg.start_predictions,
            arg.stop_predictions,
            debug=arg.debug,
            test=arg.test,
            trace=arg.trace,
        )
        seven.dirutility.assure_exists(paths['dir_out'])

        timer = seven.Timer.Timer()

        control = cls(
            arg=arg,
            path=paths,
            random_seed=random_seed,
            timer=timer,
        )
        return control


def make_datetime(s: str) -> datetime.datetime:
    date, time = s.split(' ')
    year, month, day = date.split('-')
    hour, minute, seconds_with_fraction = time.split(':')
    seconds = int(float(seconds_with_fraction))
    microseconds = int((float(seconds_with_fraction) - seconds) * 1000000)
    return datetime.datetime(
        int(year),
        int(month),
        int(day),
        int(hour),
        int(minute),
        seconds,
        microseconds,
        )


class Signals:
    'pull information out of signal.csv files'
    def __init__(self):
        self.infos = []  # type: List[Dict]
        self.counter = collections.Counter()
        self.issuer = {}  # map cusip -> issuer
        self.no_prediction_actual_pairs = []
        self._rtts = ('B', 'S')  # reclassified trade types

    def visit_test_train_output_directory(self, directory_path, invocation_parameters):
        'extract info from signals.csv and post it to self._signal_infos'
        self.counter['directories visited'] += 1
        print('visiting', directory_path)
        path = os.path.join(directory_path, 'signal.csv')
        if os.path.isfile(path):
            self.counter['files examined'] += 1
            with open(path) as f:
                csv_reader = csv.DictReader(f)
                found_prediction_actual_pair = {}
                for rtt in self._rtts:
                    found_prediction_actual_pair[rtt] = False
                last_prediction = collections.defaultdict(dict)
                for row in csv_reader:
                    # save predictions
                    for rtt in self._rtts:
                        column_name = 'prediction_%s' % rtt
                        if len(row[column_name]) > 0:
                            last_prediction[rtt] = float(row[column_name])
                    # handle actuals
                    for rtt in self._rtts:
                        column_name = 'actual_%s' % rtt
                        if len(row[column_name]) > 0:
                            if rtt in last_prediction:
                                actual = float(row[column_name])
                                predicted = float(last_prediction[rtt])
                                error = actual - predicted
                                issuer = invocation_parameters['issuer']
                                cusip = invocation_parameters['cusip']
                                self.infos.append({
                                    'issuer': issuer,
                                    'cusip': cusip,
                                    'datetime': make_datetime(row['simulated_datetime']),
                                    'rtt': rtt,
                                    'squared_error': error * error,
                                })
                                self.issuer[cusip] = issuer
                                found_prediction_actual_pair[rtt] = True
                for rtt in self._rtts:
                    if not found_prediction_actual_pair[rtt]:
                        self.counter['signal.csv file missing prediction-actual for %s' % rtt] += 1
                        self.no_prediction_actual_pairs.append(path)
        else:
            self.counter['directories without a signal.csv file'] += 1


def make_rmse_groups(infos):
    def make_rmse(v: List[float]) -> float:
        return math.sqrt(sum(v) / len(v))

    def make_by_key(squared_errors_by_key):
        result = {}
        for k, v in squared_errors_by_key.items():
            result[k] = make_rmse(v)
        return result

    squared_errors_overall = []
    squared_errors_by_cusip = collections.defaultdict(list)
    squared_errors_by_date = collections.defaultdict(list)
    squared_errors_by_rtt = collections.defaultdict(list)
    squared_errors_by_trade_hour = collections.defaultdict(list)
    for info in infos:
        squared_error = info['squared_error']
        squared_errors_overall.append(squared_error)
        squared_errors_by_cusip[info['cusip']].append(squared_error)
        squared_errors_by_date[info['datetime'].date()].append(squared_error)
        squared_errors_by_rtt[info['rtt']].append(squared_error)
        squared_errors_by_trade_hour[info['datetime'].hour].append(squared_error)
    # determine number of trades for each cusip
    squared_errors_by_n_trades = collections.defaultdict(list)
    for cusip, squared_errors in squared_errors_by_cusip.items():
        squared_errors_by_n_trades[len(squared_errors)].extend(squared_errors)

    return {
        'overall': make_rmse(squared_errors_overall),
        'by_cusip': make_by_key(squared_errors_by_cusip),
        'by_n_trades': make_by_key(squared_errors_by_n_trades),
        'by_date': make_by_key(squared_errors_by_date),
        'by_rtt': make_by_key(squared_errors_by_rtt),
        'by_trade_hour': make_by_key(squared_errors_by_trade_hour),
    }


def write_by_cusip(by_cusip, issuer_dict, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['issuer', 'cusip', 'rmse'],
            lineterminator='\n',
        )
        writer.writeheader()
        # sort by issuer, then by cusip
        for issuer in sorted(issuer_dict.values()):
            for cusip in sorted(by_cusip.keys()):
                rmse = by_cusip[cusip]
                if issuer_dict[cusip] == issuer:
                    writer.writerow({
                        'issuer': issuer_dict[cusip],
                        'cusip': cusip,
                        'rmse': rmse,
                    })


def write_by_date(by_date, issuer_dict, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['date', 'rmse'],
            lineterminator='\n',
        )
        writer.writeheader()
        for date in sorted(by_date.keys()):
            rmse = by_date[date]
            writer.writerow({
                'date': date,
                'rmse': rmse,
            })


def write_by_n_trades(by_n_trades, issuer_dict, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['n_trades', 'rmse'],
            lineterminator='\n',
        )
        writer.writeheader()
        for n_trades in sorted(by_n_trades.keys()):
            rmse = by_n_trades[n_trades]
            writer.writerow({
                'n_trades': n_trades,
                'rmse': rmse,
            })


def write_by_rtt(by_rtt, issuer_dict, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['reclassified_trade_type', 'rmse'],
            lineterminator='\n',
        )
        writer.writeheader()
        for rtt, rmse in by_rtt.items():
            writer.writerow({
                'reclassified_trade_type': rtt,
                'rmse': rmse,
            })


def write_by_trade_hour(by_trade_hour, issuer_dict, path):
    pdb.set_trace()
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['trade_hour', 'rmse'],
            lineterminator='\n',
        )
        writer.writeheader()
        for trade_hour in sorted(by_trade_hour.keys()):
            rmse = by_trade_hour[trade_hour]
            writer.writerow({
                'trade_hour': trade_hour,
                'rmse': rmse,
            })


def write_overall(rmse_overall, issuer_dict, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['rmse'],
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerow({
            'rmse': rmse_overall,
        })


def do_work(control):
    # applied_data_science.lower_priority.lower_priority()
    signals = Signals()

    # visit each expert.csv file and extract its content
    walker = seven.WalkTestTrainOutputDirectories.WalkTestTrainOutputDirectories(control.path['dir_in'])
    walker.walk_prediction_dates_between(
        visit=signals.visit_test_train_output_directory,
        start_date=control.arg.start_predictions,
        stop_date=control.arg.stop_predictions,
    )
    print('paths without prediction-actual pairs')
    for path in signals.no_prediction_actual_pairs:
        print(path)

    print('')
    print('counters from signals')
    for k in sorted(signals.counter):
        print('%50s: %d' % (k, signals.counter[k]))
    print('created %s prediction-actual pairs' % len(signals.infos))

    rmse_groups = make_rmse_groups(signals.infos)
    write_by_trade_hour(rmse_groups['by_trade_hour'], signals.issuer, control.path['out_rmse_by_trade_hour'])

    write_overall(rmse_groups['overall'], signals.issuer, control.path['out_rmse_overall'])

    write_by_cusip(rmse_groups['by_cusip'], signals.issuer, control.path['out_rmse_by_cusip'])
    write_by_date(rmse_groups['by_date'], signals.issuer, control.path['out_rmse_by_date'])
    write_by_n_trades(rmse_groups['by_n_trades'], signals.issuer, control.path['out_rmse_by_n_trades'])
    write_by_rtt(rmse_groups['by_rtt'], signals.issuer, control.path['out_rmse_by_rtt'])

    return None


def main(argv):
    control = Control.make_control(argv)
    sys.stdout = seven.Logger.Logger(control.path['out_log'])  # now print statements also write to the log file
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
