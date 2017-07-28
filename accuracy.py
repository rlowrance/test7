'''determine mean accuracy of each model_spec for all the predictions made on a specified date for an issuer

The mean is taken over all the query trades on the {trade_date}

INVOCATION
  python predict.py {issuer} {cusip} {target} {trade_date} {--debug} {--test} {--trace}
where
 issuer is the issuer symbol (ex: ORCL)
 cusip
 predict_date is the date of the events for which predictions were made
 --debug means to call pdp.set_trace() if the execution call logging.error or logging.critical
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

 NOTE: This invocation procedure is specific to the current paper trading environment, in which
 - models are fitted on the last day of the previous trade
 - all of today's predictions are made using that one fitted model

EXAMPLES OF INVOCATION
 python accuracy.py AAPL 037833AJ9 oasspread 2017-07-20 --debug

 Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import datetime
import math
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import applied_data_science.debug
import applied_data_science.dirutility
import applied_data_science.lower_priority
import applied_data_science.pickle_utilities

from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.arg_type
import seven.build
import seven.Cache
import seven.feature_makers
import seven.fit_predict_output
import seven.logging
import seven.models
import seven.read_csv
import seven.target_maker

pp = pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('issuer', type=seven.arg_type.issuer)
    parser.add_argument('cusip', type=seven.arg_type.cusip)
    parser.add_argument('target', type=seven.arg_type.target)
    parser.add_argument('predict_date', type=seven.arg_type.date)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()
    if arg.debug:
        # logging.error() and logging.critial() call pdb.set_trace() instead of raising an exception
        seven.logging.invoke_pdb = True

    random_seed = 123
    random.seed(random_seed)

    paths = seven.build.accuracy(arg.issuer, arg.cusip, arg.target, arg.predict_date, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=Timer(),
    )


def mean(x):
    'return mean value of a list of number'
    if len(x) == 0:
        print 'attempt to compute mean of empty list', x
        pdb.set_trace()
    else:
        return sum(x) / (1.0 * len(x))


def isnan(x):
    return x != x


def critical_if_nan(x, msg):
    'log critical message, if x is NaN'
    if isnan(x):
        seven.logging.critical('NaN value: %s' % msg)


def do_work(control):
    'write predictions from fitted models to file system'
    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    # determine errors for each model spec across training samples
    predictions = collections.defaultdict(list)  # Dict[model_spec, List[prediction]]
    all_errors = collections.defaultdict(list)  # Dict[model_spec, List[error]]
    for in_file_path in control.path['list_in_files']:
        print 'accuracy.py %s %s %s: reading %s' % (
            control.arg.issuer,
            control.arg.cusip,
            control.arg.predict_date,
            in_file_path,
        )
        df = pd.read_csv(
            in_file_path,
            index_col=[0],  # the model_spec
        )
        # NOTE: model_specs vary by file, because some models have have failed to fit
        for model_spec, row in df.iterrows():
            critical_if_nan(row['actual'], 'actual %s' % model_spec)
            critical_if_nan(row['prediction'], 'prediction %s' % model_spec)

            error = row['actual'] - row['prediction']

            all_errors[model_spec].append(error)
            predictions[model_spec].append(row['prediction'])

    # determine mean absolute errors for each model spec across training samples
    # NOTE: work element by element, so as to locate any NaN values

    mean_absolute_errors = {}  # Dict[model_spec, mean_absolute_error]
    lowest_mean_absolute_error = float('inf')
    for model_spec, errors in all_errors.iteritems():
        assert len(errors) > 0
        absolute_errors = map(lambda error: abs(error), errors)
        mean_absolute_error = sum(absolute_errors) / (1.0 * len(absolute_errors))
        critical_if_nan(mean_absolute_error, 'mean_absolute_error %s' % model_spec)

        mean_absolute_errors[model_spec] = mean_absolute_error
        if mean_absolute_error < lowest_mean_absolute_error:
            lowest_mean_absolute_error = mean_absolute_error

    print 'lowest mean absolute error', lowest_mean_absolute_error
    print
    print 'these models had that lowest mean absolute error'
    for model_spec, mean_absolute_error in mean_absolute_errors.iteritems():
        if mean_absolute_error == lowest_mean_absolute_error:
            print model_spec

    # determine weights. They should sum to 1.
    # follow Bianchi, Lugosi p. 14
    temperature = 1.0
    # make the unnormalized weights (unnormalized ==> do not sum to 1.0 for sure)
    sum_weights = 0.0
    unnormalized_weights = {}  # Dict[model_spec, float]
    for model_spec, mean_absolute_error in mean_absolute_errors.iteritems():
        if isnan(mean_absolute_error):
            seven.logging.critical('mean absolute error for model spec %s is NaN' % model_spec)
        weight = math.exp(-temperature * mean_absolute_error)
        critical_if_nan(weight, 'weight %s' % model_spec)
        sum_weights += weight
        unnormalized_weights[model_spec] = weight
    # normalize the weights by making the sum to 1.0
    normalized_weights = {}  # Dict[model_spec, float]
    for model_spec, mean_absolute_error in unnormalized_weights.iteritems():
        normalized_weight = mean_absolute_error / sum_weights
        critical_if_nan(normalized_weight, 'normalized_weight %s' % model_spec)
        normalized_weights[model_spec] = normalized_weight

    mean_normalized_weight = 1.0 / len(unnormalized_weights)
    print 'normalized weights above the mean normalized weight (%f)' % mean_normalized_weight
    for model_spec, normalized_weight in sorted(normalized_weights.items(), key=lambda x: x[1]):
        if normalized_weight > mean_normalized_weight:
            print '%30s: %f' % (model_spec, normalized_weight)

    # write the results
    with open(control.path['out_weights'], 'wb') as f:
        pickle.dump(normalized_weights, f, pickle.HIGHEST_PROTOCOL)

    df = pd.DataFrame(
        data={'weight': [weight_normalized for model_spec, weight_normalized in normalized_weights.iteritems()]},
        index=[model_spec for model_spec, weight_normalized in normalized_weights.iteritems()],
    )
    df.index.name = 'model_spec'
    df.to_csv(control.path['out_weights_csv'])

    print
    print 'normalized weights by model_spec'
    for model_spec, row in df.sort_index().iterrows():
        print '%-40s %f' % (model_spec, row[0])
    print
    print 'normalized weights by weight'
    for model_spec, row in df.sort_values('weight').iterrows():
        print '%-40s %17.15f' % (model_spec, row[0])

    return None


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
    # print control
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
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
        datetime

    main(sys.argv)
