'''convert fit_predict output to csv file

INVOCATION
  python report0_predictions.py {ticker} {cusip} {hpset} {--test} {--testinput} {--trace}  # all dates found in WORKING

EXAMPLES OF INVOCATION
 python report04_predictions.py ORCL 68389XAS4 grid3 --test  # all fit_predict ORCL 68389XAS4 directories, 100 input records each
 python report04_predictions.py ORCL 68389XAS4 grid3 # all fit_predict ORCL 68389XAS4 directories, all input records
 python report04_predictions.py ORCL 68389XAS4 grid1 # test case for fit_predict
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
    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = seven.build.report04_predictions(arg.ticker, arg.cusip, arg.hpset, test=arg.test, testinput=arg.testinput)
    pp(paths)
    if len(paths['in_predictions']) == 0:
        print arg
        print 'no predictions found'
        sys.exit(1)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=Timer(),
    )


def make_interarrival_bucket(interarrival_seconds):
    'return a string representing an interval of time'
    def min(i):
        'return number of seconds in i minutes'
        return 60.0 * i

    def hour(i):
        'return number of seconds in i hours'
        return 60.0 * 60.0 * i

    def day(i):
        'return number of seconds in i days'
        return 24.0 * 60.0 * 60.0 * i

    if interarrival_seconds <= min(5):
        return '0 - 5 min'
    if interarrival_seconds <= min(20):
        return '5 - 20 min'
    if interarrival_seconds <= hour(2):
        return '20 min - 2 h'
    if interarrival_seconds <= hour(4):
        return '2 h - 4 h'
    if interarrival_seconds <= hour(10):
        return '4 h - 10 h'
    if interarrival_seconds <= day(1):
        return '10 h - 1 D'
    if interarrival_seconds <= day(2):
        return '1 D - 2 D'
    if interarrival_seconds <= day(5):
        return '2 D - 5 D'
    return '5 D +'


def read_and_transform_predictions(control):
    'return DataFrame with absolute_error column added'
    skipped = collections.Counter()

    def skip(s):
        skipped[s] += 1

    data = collections.defaultdict(list)
    for in_file_path in control.path['in_predictions']:
        print 'processing', in_file_path
        if os.path.getsize(in_file_path) == 0:
            print 'skipping empty file: %s' % in_file_path
            continue
        with open(in_file_path, 'rb') as f:
            obj = pickle.load(f)
            n = 0
            for output_key, prediction in obj.iteritems():
                # Not obj will have zero length if the fit_predict program is writing it.
                # if any_nans(output_key):
                #     skip('nan in outputkey %s' % output_key)
                #     continue
                # if any_nans(prediction):
                #     skip('NaN in prediction %s %s' % (output_key, prediction))
                #     continue
                # copy data from fit_predict
                error = prediction.prediction - prediction.actual
                # copy columns
                data['trace_index'].append(output_key.trace_index)
                data['model_spec'].append(output_key.model_spec)
                data['effectivedatetime'].append(prediction.effectivedatetime)
                data['trade_type'].append(prediction.trade_type)
                data['quantity'].append(prediction.quantity)
                data['interarrival_seconds'].append(prediction.interarrival_seconds)
                data['interarrival_bucket'].append(make_interarrival_bucket(prediction.interarrival_seconds))
                data['actual'].append(prediction.actual)
                data['prediction'].append(prediction.prediction)
                # create columns
                data['error'].append(error)
                n += 1
                if control.arg.test and n == 100:
                    break
            print ' read %d predictions' % n
    predictions = pd.DataFrame(data=data)
    reordered = predictions[[
        'trace_index',
        'effectivedatetime',
        'model_spec',
        'quantity',
        'trade_type',
        'interarrival_seconds',
        'interarrival_bucket',
        'actual',
        'prediction',
        'error',
    ]]
    print 'retained %d predictions' % len(predictions)
    print 'skipped %d input records' % len(skipped)
    return reordered


def do_work(control):
    'produce reports'
    # produce reports on mean absolute errors
    predictions = read_and_transform_predictions(control)
    with open(control.path['out_predictions'], 'w') as f:
        predictions.to_csv(f)


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
