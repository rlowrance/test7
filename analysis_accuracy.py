'''analyze accuracy of the ensemble predictions

INVOCATION
  python analyze_accuracy.py test_train_output_location start_predictions stop_predictions upstream_version feature_version
   [--debug] [--test] [--trace]
where
 test_train_location is {midpredictor, dev} tells where to find the output of the test_train program
  'dev'  --> its in .../Dropbox/data/7chord/7chord-01/working/test_train/
  'prod' --> its in .../Dropbox/MidPredictor/output/
  other  --> its in {other}
 start_predictions: YYYY-MM-DD is the first date on which we attempt to test and train
 stop_predictions: YYYY-MM-DD is the last date on which we attempt to test and train
 upstream_version: any string; it identifies the version of the data fed to the machine learning
 feature_version: any string; it identifies the version of the features used by the machine learning
 --debug means to call pdb.set_trace() instead of raisinng an exception, on calls to logging.critical()
   and logging.error()
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
  python anayze_accuracy.py dev 2016-01-01 2017-12-31 1 1 --debug
  python analyze_accuracy.py /home/ubuntu/data/fit_predict/test_train 2016-01-01 2017-12-31 1 1

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
        parser.add_argument('upstream_version', type=str)
        parser.add_argument('feature_version', type=str)
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
            arg.upstream_version,
            arg.feature_version,
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


class SquaredError:
    def __init__(self,
                 ensemble, naive,
                 cusip, issuer, rtt,
                 trade_datetime, timedelta,
                 ):
        self.ensemble = ensemble
        self.naive = naive
        self.cusip = cusip
        self.issuer = issuer
        self.rtt = rtt
        self.trade_datetime = trade_datetime
        self.timedelta = timedelta

    def __repr__(self):
        return 'SquaredError(%s, ensemble=%.2f, naive=%.2f, %s, %s, %s, %s)' % (
            self.trade_datetime,
            self.ensemble,
            self.naive,
            self.issuer,
            self.cusip,
            self.rtt,
            self.timedelta,
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

        self.squared_errors = {}  # Dict[key, SquaredError]
        self.counter = collections.Counter()
        self.cusip_to_issuer = {}  # Dict[cusip, issuer]
        self.no_prediction_actual_pairs = []

        self.excluded_high_squared_error = []
        self._key_path = {}  # use by visit_test_...
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
        self.cusip_to_issuer[invocation_parameters['cusip']] = invocation_parameters['issuer']
        if os.path.isfile(path):
            self._count('files examined')
            file_squared_errors = self._visit_signal_file(
                path,
                invocation_parameters,
            )
            for k, v in file_squared_errors.items():
                if k in self.squared_errors:
                    print('duplicate squared error result', k)
                    print('path', path)
                    print('previous path contaiinng the key', self._key_path[k])
                    pdb.set_trace()
                self._key_path[k] = path
                if v.naive > 1e6:
                    self.excluded_high_squared_error.append(v)
                else:
                    self.squared_errors[k] = v
        else:
            pdb.set_trace()
            self._count('directories without a signal.csv file')
        return None

    def _count(self, id):
        self.counter[id] += 1

    def _visit_signal_file(self, path, invocation_parameters) -> Dict:
        'return Dict[Tuple[trade_datetime, cusip, event_source_identifier], Dict of errors]'
        def make_squared_error(prediction, actual):
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
                        if rtt in ensemble_prediction:
                            if False and debug:
                                print('producing ensemble error')
                                pdb.set_trace()
                            self._count('errors from ensemble and naive predictions %s' % rtt)
                            squared_error = SquaredError(
                                ensemble=make_squared_error(ensemble_prediction[rtt], r.actual(rtt)),
                                naive=make_squared_error(naive_prediction[rtt], r.actual(rtt)),
                                cusip=invocation_parameters['cusip'],
                                issuer=invocation_parameters['issuer'],
                                rtt=rtt,
                                trade_datetime=r.datetime(),
                                timedelta=r.datetime() - ensemble_prediction_datetime[rtt],
                            )
                            # MAYBE: the key is the prediction event ID
                            key = (squared_error.trade_datetime, squared_error.cusip, r.event_source_identifier())
                            result[key] = squared_error
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


def make_rmses(squared_errors: List[SquaredError]):
    def rmse(v):
        return math.sqrt(sum(v) / len(v))

    ensemble = []
    naive = []
    for squared_error in squared_errors:
        ensemble.append(squared_error.ensemble)
        naive.append(squared_error.naive)
    return rmse(ensemble), rmse(naive)


def write_by_cusip(squared_errors, cusip_to_issuer, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['issuer', 'cusip', 'count', 'rmse_ensemble', 'rmse_naive'],
            lineterminator='\n',
        )
        writer.writeheader()
        # sort by issuer, then by cusip
        squared_errors_by_issuer_cusip = collections.defaultdict(list)
        for squared_error in squared_errors:
            cusip = squared_error.cusip
            issuer = cusip_to_issuer[cusip]
            squared_errors_by_issuer_cusip[(issuer, cusip)].append(squared_error)
        for k in sorted(squared_errors_by_issuer_cusip.keys()):
            v = squared_errors_by_issuer_cusip[k]
            ensemble, naive = make_rmses(v)
            writer.writerow({
                'issuer': k[0],
                'cusip': k[1],
                'count': len(v),
                'rmse_ensemble': ensemble,
                'rmse_naive': naive,
            })


def write_by_date(squared_errors, cusip_to_issuer, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['date', 'count', 'rmse_ensemble', 'rmse_naive'],
            lineterminator='\n',
        )
        writer.writeheader()
        squared_errors_by_date = collections.defaultdict(list)
        for squared_error in squared_errors:
            squared_errors_by_date[squared_error.trade_datetime.date()].append(squared_error)
        for k in sorted(squared_errors_by_date.keys()):
            v = squared_errors_by_date[k]
            ensemble, naive = make_rmses(v)
            writer.writerow({
                'date': k,
                'count': len(v),
                'rmse_ensemble': ensemble,
                'rmse_naive': naive,
            })


def write_by_issuer(squared_errors, cusip_to_issuer, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['issuer', 'count', 'rmse_ensemble', 'rmse_naive'],
            lineterminator='\n',
        )
        writer.writeheader()
        squared_errors_by_issuer = collections.defaultdict(list)
        for squared_error in squared_errors:
            issuer = cusip_to_issuer[squared_error.cusip]
            squared_errors_by_issuer[issuer].append(squared_error)
        for issuer in sorted(squared_errors_by_issuer.keys()):
            v = squared_errors_by_issuer[issuer]
            ensemble, naive = make_rmses(v)
            writer.writerow({
                'issuer': issuer,
                'count': len(v),
                'rmse_ensemble': ensemble,
                'rmse_naive': naive,
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


def write_by_rtt(squared_errors, cusip_to_issuer, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['reclassified_trade_type', 'count', 'rmse_ensemble', 'rmse_naive'],
            lineterminator='\n',
        )
        writer.writeheader()
        squared_errors_by_rtt = collections.defaultdict(list)
        for squared_error in squared_errors:
            rtt = squared_error.rtt
            squared_errors_by_rtt[rtt].append(squared_error)
        for rtt in sorted(squared_errors_by_rtt.keys()):
            v = squared_errors_by_rtt[rtt]
            ensemble, naive = make_rmses(v)
            writer.writerow({
                'reclassified_trade_type': rtt,
                'count': len(v),
                'rmse_ensemble': ensemble,
                'rmse_naive': naive,
            })


def write_by_timedelta_hours(squared_errors, cusip_to_issuer, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['timedelta_hours', 'count', 'rmse_ensemble', 'rmse_naive'],
            lineterminator='\n',
        )
        writer.writeheader()
        squared_errors_by_timedelta_hours = collections.defaultdict(list)
        for squared_error in squared_errors:
            hours = int(squared_error.timedelta.total_seconds() / 60 / 60)
            squared_errors_by_timedelta_hours[hours].append(squared_error)
        for hours in sorted(squared_errors_by_timedelta_hours.keys()):
            v = squared_errors_by_timedelta_hours[hours]
            ensemble, naive = make_rmses(v)
            writer.writerow({
                'timedelta_hours': hours,
                'count': len(v),
                'rmse_ensemble': ensemble,
                'rmse_naive': naive,
            })


def write_by_timedelta_minutes(squared_errors, cusip_to_issuer, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['timedelta_minutes', 'count', 'rmse_ensemble', 'rmse_naive'],
            lineterminator='\n',
        )
        writer.writeheader()
        squared_errors_by_timedelta_minutes = collections.defaultdict(list)
        for squared_error in squared_errors:
            minutes = int(squared_error.timedelta.total_seconds() / 60)
            squared_errors_by_timedelta_minutes[minutes].append(squared_error)
        for minutes in sorted(squared_errors_by_timedelta_minutes.keys()):
            v = squared_errors_by_timedelta_minutes[minutes]
            ensemble, naive = make_rmses(v)
            writer.writerow({
                'timedelta_minutes': minutes,
                'count': len(v),
                'rmse_ensemble': ensemble,
                'rmse_naive': naive,
            })


def write_by_trade_hour(squared_errors, cusip_to_issuer, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['trade_hour', 'count', 'rmse_ensemble', 'rmse_naive'],
            lineterminator='\n',
        )
        writer.writeheader()
        squared_errors_by_trade_hour = collections.defaultdict(list)
        for squared_error in squared_errors:
            trade_hour = squared_error.trade_datetime.hour
            squared_errors_by_trade_hour[trade_hour].append(squared_error)
        for trade_hour in sorted(squared_errors_by_trade_hour.keys()):
            v = squared_errors_by_trade_hour[trade_hour]
            ensemble, naive = make_rmses(v)
            writer.writerow({
                'trade_hour': trade_hour,
                'count': len(v),
                'rmse_ensemble': ensemble,
                'rmse_naive': naive,
            })


def write_invocation(arg, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['invocation_parameter', 'value'],
            lineterminator='\n',
        )
        writer.writeheader()

        def w(name, value):
            writer.writerow({
                'invocation_parameter': name,
                'value': value,
            })

        w('test_train_output_location', arg.test_train_output_location)
        w('start_predictions', arg.start_predictions)
        w('stop_prediction', arg.stop_predictions)
        w('upstream_version', arg.upstream_version)
        w('feature_version', arg.feature_version)
        w('debug', arg.debug)
        w('test', arg.test)
        w('trace', arg.trace)


def write_squared_errors(squared_errors, cusip_to_issuer, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['trade_date', 'trade_datetime',
             'issuer', 'cusip', 'reclassified_trade_type',
             'squared_error_ensemble', 'squared_error_naive',
             ],
            lineterminator='\n',
        )
        writer.writeheader()
        for squared_error in squared_errors:
            writer.writerow({
                'trade_date': squared_error.trade_datetime.date(),
                'trade_datetime': squared_error.trade_datetime,
                'issuer': cusip_to_issuer[squared_error.cusip],
                'cusip': squared_error.cusip,
                'reclassified_trade_type': squared_error.rtt,
                'squared_error_ensemble': squared_error.ensemble,
                'squared_error_naive': squared_error.naive,
            })


def write_overall(squared_errors, cusip_to_issuer, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['count', 'rmse_ensemble', 'rmse_naive'],
            lineterminator='\n',
        )
        writer.writeheader()
        ensemble, naive = make_rmses(squared_errors)
        writer.writerow({
            'count': len(squared_errors),
            'rmse_ensemble': ensemble,
            'rmse_naive': naive,
        })


def do_work(control):
    # applied_data_science.lower_priority.lower_priority()
    write_invocation(control.arg, control.path['out_invocation'])
    signals = Signals(control.arg.start_predictions, control.arg.stop_predictions)

    # visit each expert.csv file and extract its content
    walker = seven.WalkTestTrainOutputDirectories.WalkTestTrainOutputDirectories(
        control.path['dir_in'],
        control.arg.upstream_version,
        control.arg.feature_version,
    )
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

    print('\nexcluded because of high errors for the naive prediction')
    for squared_error in signals.excluded_high_squared_error:
        print(squared_error)

    # NOTE: This analysis is flawed in that it only considers actuals on the same date as the prediction

    def write(f, path):
        f(signals.squared_errors.values(), signals.cusip_to_issuer, control.path[path])

    write(write_by_cusip, 'out_rmse_by_cusip')
    write(write_by_date, 'out_rmse_by_date')
    write(write_by_issuer, 'out_rmse_by_issuer')
    write(write_by_rtt, 'out_rmse_by_rtt')
    write(write_by_timedelta_minutes, 'out_rmse_by_timedelta_minutes')
    write(write_by_timedelta_hours, 'out_rmse_by_timedelta_hours')
    write(write_by_trade_hour, 'out_rmse_by_trade_hour')
    write(write_squared_errors, 'out_squared_errors')
    write(write_overall, 'out_rmse_overall')

    return None


def main(argv):
    print('starting analysis_accuracy: argv=%s' % argv)
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
