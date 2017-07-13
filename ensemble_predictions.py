'''determine accuracy of each model_spec for all the predictions made on a specified date for an issuer

INVOCATION
  python ensemble-predictions.py {issuer} {cusip} {trade_date} {--test} {--trace}
where
 issuer is the issuer symbol (ex: ORCL)
 cusip
 trade_date is the date of the query transactions.
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python ensemble_predictions.py AAPL 037833AG5 2017-06-27

 Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''

from __future__ import division

import argparse
import cPickle as pickle
import datetime
import math
import os
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

    paths = seven.build.ensemble_predictions(arg.issuer, arg.cusip, arg.trade_date, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=Timer(),
    )


# def date_to_str(date):
#     'convert datetime.date to string'
#     return '%4d-%02d-%02d' % (date.year, date.month, date.day)


def mean(x):
    'return mean value of a list of number'
    if len(x) == 0:
        print 'attempt to compute mean of empty list', x
        pdb.set_trace()
    else:
        return sum(x) / (1.0 * len(x))


def make_expert_prediction(natural_query_features, synthetic_trade_type, expert_weights, fitted, trace_index):
    'return (DataFrame of len 1 containing the expert prediction, standard_deviation of the experts)'
    if synthetic_trade_type == 'natural':
        synthetic_query_features = natural_query_features.copy()
    else:
        synthetic_query_features = seven.models.synthetic_query(natural_query_features, synthetic_trade_type)

    # form weighted average of the prediction of the expert models
    ensemble_prediction = 0.0
    expert_prediction_dict = {}  # Dict[model_spec, prediction]
    for model_spec, fitted_model in fitted.iteritems():
        expert_predictions = fitted_model.predict(synthetic_query_features)
        assert len(expert_predictions) == 1
        expert_prediction = expert_predictions[0]
        expert_prediction_dict[model_spec] = expert_prediction
        if model_spec not in expert_weights:
            print 'trace_index %s did not fit model_spec %s' % (trace_index, model_spec)
            continue
        ensemble_prediction += expert_prediction * expert_weights[model_spec]

    # determine the variance in the experts' predictions relative to the ensemble prediction
    # determine variances, by this algorithm
    # source: Dennis Shasha, "weighting the variances", email Jun 23, 2017 and subsequent discussions in email
    # let m1, m2, ..., mk = the models
    # let p1, p2, ..., pk = their predictions
    # let w1, w2, ..., wk = their weights (that sum to 1)
    # let e = prediction of ensemble = (w1*p1 + w2*p2 + ... + wk*pk) = weighted mean
    # let delta_k = pk - e
    # let variance = w1 * delta_1^2 + w2 * delta_2^2 + ... + wk * delta_k^2)
    # let standard deviation = sqrt(variance)
    variance = 0.0
    for model_spec, p in expert_prediction_dict.iteritems():
        delta = p - ensemble_prediction
        variance += expert_weights[model_spec] * delta * delta
    standard_deviation = math.sqrt(variance)

    return ensemble_prediction, standard_deviation


def do_work(control):
    'write predictions from fitted models to file system'
    def write_ensemble_predictions(ensemble_predictions):
        ensemble_predictions.to_csv(control.path['out_ensemble_predictions'])
        print 'wrote %d ensemble predictions' % len(ensemble_predictions)

    # read the weights to be used for the experts
    with open(control.path['in_accuracy'], 'rb') as f:
        expert_weights = pickle.load(f)
    print 'read %d weights' % len(expert_weights)

    # read the query features
    # its possible that the file exists and is empty
    query_features_test = pd.read_csv(
        control.path['in_query_features'],
        index_col=[0],   # the issuepriceid (aka, trace_index, aka, trade_index)
    )
    if len(query_features_test) == 0:
        print 'no query features for %s %s %s' % (
            control.arg.issuer,
            control.arg.cusip,
            control.arg.trade_date,
        )
        print 'exiting after creating empty output file'
        write_ensemble_predictions(pd.DataFrame())
        sys.exit(0)
    # re-read in order to convert the dates and times to usable values
    query_features = pd.read_csv(
        control.path['in_query_features'],
        index_col=[0],   # the issuepriceid (aka, trace_index, aka, trade_index)
        parse_dates=[
            'id_p_effectivedate', 'id_p_effectivedatetime', 'id_p_effectivetime',
            'id_otr1_effectivedate', 'id_otr1_effectivedatetime', 'id_otr1_effectivetime',
        ],
    )
    print 'read %d queries to be predicted' % len(query_features)

    # read the target features
    query_targets = pd.read_csv(
        control.path['in_query_targets'],
        index_col=[0],
        parse_dates=['id_effectivedate', 'id_effectivedatetime', 'id_effectivetime']
    )
    print 'read %d targets' % len(query_targets)

    # read the fitted models
    fitted = {}  # Dict[model_spec, fitted_model]
    for dirpath, dirnames, filenames in os.walk(control.path['dir_in_fitted']):
        for filename in filenames:
            if filename == '0log.txt':
                continue
            path_in = os.path.join(dirpath, filename)
            with open(path_in, 'rb') as f:
                obj = pickle.load(f)
            model_spec = filename.split('.')[0]
            fitted[model_spec] = obj
    print 'read %d fitted models' % len(fitted)

    ensemble_predictions = pd.DataFrame()
    for trace_index in query_features.index:
        if False and trace_index == 127453431:
            print 'found it'
            pdb.set_trace()
        print 'ensemble_predictions.py %s %s %s: using features with trace_index %s' % (
            control.arg.issuer,
            control.arg.cusip,
            control.arg.trade_date,
            trace_index,
        )
        actual = query_targets.loc[trace_index]['target_oasspread']
        query_feature = query_features.loc[[trace_index]]  # must be a DataFrame
        ep = {}  # ensemble_predictinn[synthetic_trade_type]
        sd = {}  # standard_deviation_of_expert_predictions[synthetic_trade_type]
        for synthetic_trade_type in ('natural', 'B', 'D', 'S'):
            ensemble_prediction, standard_deviation = make_expert_prediction(
                natural_query_features=query_feature,
                fitted=fitted,
                synthetic_trade_type=synthetic_trade_type,
                expert_weights=expert_weights,
                trace_index=trace_index,
            )
            ep[synthetic_trade_type] = ensemble_prediction
            sd[synthetic_trade_type] = standard_deviation
        new_row = pd.DataFrame(
            data={
                'actual': actual,
                'prediction_natural': ep['natural'],
                'prediction_B': ep['B'],
                'prediction_D': ep['D'],
                'prediction_S': ep['S'],
                'error_natural': actual - ep['natural'],
                'error_B': actual - ep['B'],
                'error_D': actual - ep['D'],
                'error_S': actual - ep['S'],
                'standard_deviation_natural': sd['natural'],
                'standard_deviation_B': sd['B'],
                'standard_deviation_D': sd['D'],
                'standard_deviation_S': sd['S'],
                'natural_trade_type': query_feature['id_p_trade_type'].iloc[0],
                'reclassified_trade_type': query_feature['id_p_reclassified_trade_type'].iloc[0],
            },
            index=pd.Index(
                data=[trace_index],
                name='trace_index',
            )
        )
        ensemble_predictions = ensemble_predictions.append(new_row)

    # write the results
    # expert_predictions.to_csv(control.path['out_expert_predictions'])
    # print 'wrote %d expert predictions to csv' % len(expert_predictions)
    write_ensemble_predictions(ensemble_predictions)

    print ensemble_predictions[:10]

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
