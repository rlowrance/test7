'''determine accuracy of each model_spec for all the predictions made on a specified date for an issuer

INVOCATION
  python predict.py {issuer} {cusip} {trade_date} {--test} {--trace}
where
 issuer is the issuer symbol (ex: ORCL)
 cusip
 trade_date is the date the models were fitted.
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python accuracy.py AAPL 037833AG5 2017-06-26
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
import seven.models
import seven.read_csv
import seven.target_maker

pp = pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('issuer', type=seven.arg_type.issuer)
    parser.add_argument('cusip', type=seven.arg_type.cusip)
    parser.add_argument('trade_date', type=seven.arg_type.date)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = seven.build.accuracy(arg.issuer, arg.cusip, arg.trade_date, test=arg.test)
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


def do_work(control):
    'write predictions from fitted models to file system'
    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    # determine errors for each model spec across training samples
    predictions = collections.defaultdict(list)  # Dict[model_spec, List[prediction]]
    errors = collections.defaultdict(list)  # Dict[model_spec, List[error]]
    for in_file_path in control.path['list_in_files']:
        print 'reading', in_file_path
        df = pd.read_csv(
            in_file_path,
            index_col=[0],  # the model_spec
        )
        # NOTE: model_specs vary by file, because some models have have failed to fit
        for model_spec, row in df.iterrows():
            error = row['actual'] - row['prediction']

            errors[model_spec].append(error)
            predictions[model_spec].append(row['prediction'])

    # determine mean absolute errors for each model spec across training samples
    mean_absolute_errors = {}  # Dict[model_spec, mean_absolute_error]
    lowest_mean_absolute_error = float('inf')
    for model_spec, errors in errors.iteritems():
        assert len(errors) > 0
        absolute_errors = map(lambda error: abs(error), errors)
        mean_absolute_error = sum(absolute_errors) / (1.0 * len(absolute_errors))

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
    weights_unnormalized = {
        model_spec: math.exp(- temperature * mean_absolute_error)
        for model_spec, mean_absolute_error in mean_absolute_errors.iteritems()
    }
    sum_weights = sum([weight for model_spec, weight in weights_unnormalized.iteritems()])

    weights_normalized = {
        model_spec: weight_unnormalized / sum_weights
        for model_spec, weight_unnormalized in weights_unnormalized.iteritems()
    }

    # write the results
    with open(control.path['out_weights'], 'wb') as f:
        pickle.dump(weights_normalized, f, pickle.HIGHEST_PROTOCOL)

    df = pd.DataFrame(
        data={'weight': [weight_normalized for model_spec, weight_normalized in weights_normalized.iteritems()]},
        index=[model_spec for model_spec, weight_normalized in weights_normalized.iteritems()],
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
        print '%-40s %f' % (model_spec, row[0])

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
