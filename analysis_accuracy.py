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
from typing import Dict, List

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


def make_date(s: str) -> datetime.date:
    year, month, day = s.split('-')
    return datetime.date(
        int(year),
        int(month),
        int(day),
    )


def make_datetime(s: str) -> datetime.datetime:
    date, time = s.split(' ')
    d = make_date(date)
    year, month, day = date.split('-')
    hour, minute, seconds_with_fraction = time.split(':')
    seconds = int(float(seconds_with_fraction))
    microseconds = int((float(seconds_with_fraction) - seconds) * 1000000)
    return datetime.datetime(
        d.year,
        d.month,
        d.day,
        int(hour),
        int(minute),
        seconds,
        microseconds,
        )


class SignalRow:
    def __init__(self, row):
        self._row = copy.copy(row)

    def actual(self, rtt) -> float:
        return float(self._value('actual', rtt))

    def date(self):
        return self.datetime().date()

    def datetime(self) -> datetime.datetime:
        return make_datetime(self._row['simulated_datetime'])

    def event_cusip(self) -> str:
        return self._row['event_cusip']

    def event_source(self) -> str:
        return self._row['event_source']

    def event_source_identifier(self) -> str:
        return self._row['event_source_identifier']

    def prediction(self, rtt) -> float:
        return float(self._value('prediction', rtt))

    def has_actual(self, rtt) -> bool:
        return self._has('actual', rtt)

    def has_prediction(self, rtt) -> bool:
        return self._has('prediction', rtt)

    def _column_name(self, field_name, rtt):
        return '%s_%s' % (field_name, rtt)

    def _has(self, field_name, rtt):
        return len(self._row[self._column_name(field_name, rtt)]) > 0

    def _value(self, field_name, rtt):
        return self._row[self._column_name(field_name, rtt)]


class Signals:
    'pull information out of signal.csv files'
    def __init__(self, start_predictions, stop_predictions):
        self.start_predictions = start_predictions
        self.stop_predictions = stop_predictions

        self.infos = {}  # type: Dick[id, Dict[attribute_name, attribute_value]]
        self.counter = collections.Counter()
        self.issuer = {}  # Dict[cusip, issuer]
        self.no_prediction_actual_pairs = []

        self._rtts = ('B', 'S')  # reclassified trade types

    def visit_test_train_output_directory(self, directory_path, invocation_parameters):
        'extract info from signals.csv and post it to self.infos'
        verbose = True

        def vp(*xs):
            if verbose:
                print(*xs)

        self._count('directories visited')
        print('visiting', directory_path)
        path = os.path.join(directory_path, 'signal.csv')
        self.issuer[invocation_parameters['cusip']] = invocation_parameters['issuer']
        if os.path.isfile(path):
            self._count('files examined')
            file_result = self._visit_signal_file(
                path,
                invocation_parameters,
            )
            for k, v in file_result.items():
                if k in self.infos:
                    print('duplicate result', k)
                    pdb.set_trace()
                self.infos[k] = v
        else:
            pdb.set_trace()
            self._count('directories without a signal.csv file')
        return None

    def _count(self, id):
        self.counter[id] += 1

    def _visit_signal_file(self, path, invocation_parameters) -> Dict:
        'return Dict[Tuple[trade_datetime, cusip, event_source_identifier], Dict of errors]'
        def squared_error(prediction, actual):
            error = prediction - actual
            return error * error

        start_predictions = make_date(invocation_parameters['start_predictions'])
        stop_predictions = make_date(invocation_parameters['stop_predictions'])

        def has_any_prediction(r):
            return r.has_prediction('B') or r.has_prediction('S')

        def has_relevant_date(r):
            r_date = r.datetime().date()
            return start_predictions <= r_date <= stop_predictions

        def has_query_cusip(r):
            return r.event_cusip() == invocation_parameters['cusip']

        def is_relevant(r):
            return has_relevant_date(r) and has_query_cusip(r)

        def trace_event_for_cusip(r):
            return r.event_source() == 'trace' and r.event_cusip() == invocation_parameters['cusip']

        if invocation_parameters['cusip'] == '037833AJ9':
            print('found cusip', path)
            debug = True

        with open(path) as f:
            csv_reader = csv.DictReader(f)
            ensemble_prediction = collections.defaultdict(dict)
            ensemble_prediction_datetime = collections.defaultdict(dict)
            naive_prediction = collections.defaultdict(dict)
            result = {}
            debug = False and invocation_parameters['cusip'] == '037833AJ9'
            for row in csv_reader:
                self._count('rows examined')
                r = SignalRow(row)
                if trace_event_for_cusip(r):
                    rtt = 'B' if r.has_actual('B') else 'S'
                    self._count('trace event %s for cusip' % rtt)
                    if has_relevant_date(r):
                        # determine and save errors
                        if False and debug:
                            print('has relevant date')
                            pdb.set_trace()
                        msg = 'actual %s with relevant date and cusip' % rtt
                        self._count(msg)
                        d = {}
                        if rtt in ensemble_prediction:
                            if False and debug:
                                print('producing ensemble error')
                                pdb.set_trace()
                            self._count('errors from ensemble predictions %s' % rtt)
                            d['squared_error_ensemble'] = squared_error(ensemble_prediction[rtt], r.actual(rtt))
                            d['timedelta_ensemble'] = r.datetime() - ensemble_prediction_datetime[rtt]
                        if rtt in naive_prediction:
                            if False and debug:
                                print('producing naive error')
                                pdb.set_trace()
                            self._count('errors from naive predictions %s' % rtt)
                            d['squared_error_naive'] = squared_error(naive_prediction[rtt], r.actual(rtt))
                        if len(d) > 0:
                            # we have at least one squared error
                            # fill in the other values
                            if False and debug:
                                print('appending to result')
                                pdb.set_trace()
                            d['cusip'] = invocation_parameters['cusip']
                            d['issuer'] = invocation_parameters['issuer']
                            d['rtt'] = rtt
                            d['trade_datetime'] = r.datetime()
                            key = (d['trade_datetime'], d['cusip'], r.event_source_identifier())
                            result[key] = d
                            # always update the naive prediction from the actual
                    # update the naive prediction
                    if False and debug:
                        print('updating naive prediction')
                        pdb.set_trace()
                    msg = 'naive prediction %s' % rtt
                    self._count(msg)
                    naive_prediction[rtt] = r.actual(rtt)
                elif has_any_prediction(r):  # predictions are always for the query cusip
                    if False and debug:
                        print('found ensemble prediction')
                        pdb.set_trace()
                    rtt = 'B' if r.has_prediction('B') else 'S'
                    msg = 'ensemble prediction %s' % rtt
                    self._count(msg)
                    ensemble_prediction[rtt] = r.prediction(rtt)
                    ensemble_prediction_datetime[rtt] = r.datetime()
                else:
                    self._count('signal not a trace for the cusip nor an ensemble prediction')

            if len(result) == 0:
                print('file with no predictions', path)
                self._count('files without predictions')
            return result


def make_rmse_groups(infos, cusip_issuer):
    'return Dict[group_name, List]'
    def make_rmse(v: List[float]) -> float:
        return math.sqrt(sum(v) / len(v))

    def make_by_key(squared_errors_by_key):
        result = {}
        for k, v in squared_errors_by_key.items():
            result[k] = make_rmse(v)
        return result

    def make_by_timedelta_minutes(squared_errors_by_timedelta):
        by_minutes = collections.defaultdict(list)
        for dt, squared_errors in squared_errors_by_timedelta.items():
            by_minutes[int(dt.total_seconds() / 60)].extend(squared_errors)
        result = {}
        for minutes, squared_errors in by_minutes.items():
            result[minutes] = make_rmse(squared_errors)
        return result

    # all of these are for the ensemble models
    squared_errors_overall = []
    squared_errors_by_cusip = collections.defaultdict(list)
    squared_errors_by_issuer = collections.defaultdict(list)
    squared_errors_by_date = collections.defaultdict(list)
    squared_errors_by_rtt = collections.defaultdict(list)
    squared_errors_by_timedelta = collections.defaultdict(list)
    squared_errors_by_trade_hour = collections.defaultdict(list)

    # these are for ensemble and naive predictions for the same trace print
    # NOTE: There are many naive predictions where we do not have an ensemble prediction
    squared_errors_overall_naive = []  # type: List[float)

    # the keys are just for detecting duplicates
    for info in infos.values():  # ignore the keys
        if 'squared_error_ensemble' in info:
            squared_error = info['squared_error_ensemble']
            cusip = info['cusip']
            issuer = cusip_issuer[cusip]
            squared_errors_overall.append(squared_error)
            squared_errors_by_cusip[info['cusip']].append(squared_error)
            squared_errors_by_date[info['trade_datetime'].date()].append(squared_error)
            squared_errors_by_issuer[issuer].append(squared_error)
            squared_errors_by_rtt[info['rtt']].append(squared_error)
            squared_errors_by_timedelta[info['timedelta_ensemble']].append(squared_error)
            squared_errors_by_trade_hour[info['trade_datetime'].hour].append(squared_error)

            squared_errors_overall_naive.append(info['squared_error_naive'])

    # determine number of trades for each cusip
    squared_errors_by_n_trades = collections.defaultdict(list)
    for cusip, squared_errors in squared_errors_by_cusip.items():
        squared_errors_by_n_trades[len(squared_errors)].extend(squared_errors)

    return {
        # all these are for ensemble predictions
        'overall': make_rmse(squared_errors_overall),
        'by_cusip': make_by_key(squared_errors_by_cusip),
        'by_issuer': make_by_key(squared_errors_by_issuer),
        'by_n_trades': make_by_key(squared_errors_by_n_trades),
        'by_date': make_by_key(squared_errors_by_date),
        'by_rtt': make_by_key(squared_errors_by_rtt),
        'by_timedelta_minutes': make_by_timedelta_minutes(squared_errors_by_timedelta),
        'by_trade_hour': make_by_key(squared_errors_by_trade_hour),

        # theis for for naive predictions when there was a corresponding ensemble prediction
        'overall_naive': make_rmse(squared_errors_overall_naive),
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
        new = {}
        for cusip, rmse in by_cusip.items():
            new[(issuer_dict[cusip], cusip)] = rmse
        for issuer_cusip in sorted(new.keys()):
            issuer, cusip = issuer_cusip
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


def write_by_issuer(by_issuer, issuer_dict, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['issuer', 'rmse'],
            lineterminator='\n',
        )
        writer.writeheader()
        for issuer in sorted(by_issuer.keys()):
            rmse = by_issuer[issuer]
            writer.writerow({
                'issuer': issuer,
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


def write_by_timedelta(by_timedelta_minutes, issuer_dict, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['timedelta_minutes', 'rmse'],
            lineterminator='\n',
        )
        writer.writeheader()
        for timedelta_minutes in sorted(by_timedelta_minutes.keys()):
            rmse = by_timedelta_minutes[timedelta_minutes]
            writer.writerow({
                'timedelta_minutes': timedelta_minutes,
                'rmse': rmse,
            })


def write_by_trade_hour(by_trade_hour, issuer_dict, path):
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


def write_overall(rmse_overall_ensemble, rmse_overall_naive, issuer_dict, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['rmse_ensemble', 'rmse_naive'],
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerow({
            'rmse_ensemble': rmse_overall_ensemble,
            'rmse_naive': rmse_overall_naive,
        })


def do_work(control):
    # applied_data_science.lower_priority.lower_priority()
    signals = Signals(control.arg.start_predictions, control.arg.stop_predictions)

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

    rmse_groups = make_rmse_groups(signals.infos, signals.issuer)

    write_overall(
        rmse_overall_ensemble=rmse_groups['overall'],
        rmse_overall_naive=rmse_groups['overall_naive'],
        issuer_dict=signals.issuer,
        path=control.path['out_rmse_overall_both'],
        )

    write_by_cusip(rmse_groups['by_cusip'], signals.issuer, control.path['out_rmse_by_cusip'])
    write_by_date(rmse_groups['by_date'], signals.issuer, control.path['out_rmse_by_date'])
    write_by_issuer(rmse_groups['by_issuer'], signals.issuer, control.path['out_rmse_by_issuer'])
    write_by_timedelta(rmse_groups['by_timedelta_minutes'], signals.issuer, control.path['out_rmse_by_timedelta'])
    write_by_n_trades(rmse_groups['by_n_trades'], signals.issuer, control.path['out_rmse_by_n_trades'])
    write_by_rtt(rmse_groups['by_rtt'], signals.issuer, control.path['out_rmse_by_rtt'])
    write_by_trade_hour(rmse_groups['by_trade_hour'], signals.issuer, control.path['out_rmse_by_trade_hour'])

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
