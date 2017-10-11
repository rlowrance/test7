'''determine accuracy of each model_spec for all the predictions made on a specified date for an issuer

INVOCATION
  python ensemble-predictions.py {issuer} {cusip} {target} {predicted_event_id} {fitted_event_id} {accuracy_date}
  {--debug} {--test} {--trace}
where
 issuer is the issuer symbol (ex: ORCL)
 cusip
 target is the target (ex: oasspread)
 predicted_event_id: is the EventId of the event to be predicted. It has a reclassified trade type.
 fitted_event_id: is the EventId of a previous event. It has the same reclassified trade type as
   the predicted_event_id
 trade_date is the date of the query transactions.
 predict_date is the date of the events for which predictions were made
 accuracy_date is the date used to determine how to weight the ensemble model
 --debug means to call pdp.set_trace() if the execution call logging.error or logging.critical
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

USAGE IN THE PAPER TRADING ENVIRONMENT:

The ensemble predictions are produced for every event on day t. Thus, this program is invoked once
for every event on day t.

The fitted model used to make the ensemble predictions is the last fitted model from day t - 1. Here a day
is a trading day, not a calendar day. We use the last fitted model from the previous trading day with the idea
that we are simulating what happens if we cannot train the models as events arrive. In the case, we will
train the models are night and use those trained models the next day.

We use the most recent of yesterday's events as the model to use on day t, except in cases when that
event had other events occur at the same datetime. When that happens, the events possibly occured in some
unknown order, and hence information could have flowed from one event to another. So we don't use the most
recent model. Instead, we search backwards until we find the most recent event that had not other events at
the same datetime.

The fitted model will produce an estimate for each hyperparameter settings. These estimates are blended using
the weights given by the accuracy_date. The weights are determined from the accuracy of all the models
induced by the hyperparameters on day t - 1.

EXAMPLES OF INVOCATION
 python ensemble_predictions.py AAPL 037833AJ9 oasspread 2017-07-20-09-06-46-traceprint-127987331 2017-07-19-15-58-14-traceprint-127978656 2017-07-19 --debug
OLD EXAMPLES OF INVOCATIONS
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
    parser.add_argument('prediction_event_id', type=seven.arg_type.event_id)
    parser.add_argument('fitted_event_id', type=seven.arg_type.event_id)
    parser.add_argument('accuracy_date', type=seven.arg_type.date)
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

    paths = seven.build.ensemble_predictions(
        arg.issuer,
        arg.cusip,
        arg.target,
        arg.prediction_event_id,
        arg.fitted_event_id,
        arg.accuracy_date,
        test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=Timer(),
    )


def meanOLD(x):
    'return mean value of a list of number'
    if len(x) == 0:
        print 'attempt to compute mean of empty list', x
        pdb.set_trace()
    else:
        return sum(x) / (1.0 * len(x))


def do_work(control):
    'write predictions from fitted models to file system'
    def write_ensemble_predictions(ensemble_predictions):
        ensemble_predictions.to_csv(control.path['out_ensemble_predictions'])
        print 'wrote %d ensemble predictions' % len(ensemble_predictions)

    event_info = seven.EventInfo.EventInfo(
        control.arg.issuer,
        control.arg.cusip,
    )
    prediction_event_id = seven.EventId.EventId.from_str(control.arg.prediction_event_id)
    reclassified_trade_type = event_info.reclassified_trade_type(prediction_event_id)

    # read the weights to be used for the experts
    expert_weights_df, err = seven.read_csv.accuracy(control.path['in_accuracy %s' % reclassified_trade_type])
    if err is not None:
        seven.logging.critical('cannot read accurcy file: %s' % err)
        os.exit(1)
    print 'read %d weights' % len(expert_weights_df)
    expert_weight = {}  # Dict[model_spec, float]
    for model_spec, row in expert_weights_df.iterrows():
        expert_weight[model_spec] = row['normalized_weight']

    # read the query features
    # its possible that the file exists and is empty
    prediction_features, err = seven.read_csv.features_targets(control.path['in_prediction_features'])
    if err is not None:
        seven.logging.critical('cannot read prediction features file: %s' % err)
        os.exit(1)
    assert len(prediction_features) == 1
    print 'read %d prediction features' % len(prediction_features)

    # read the fitted models
    fitted = {}  # Dict[model_spec, fitted_model]
    dir_in_fitted = control.path['dir_in_fitted']
    for item_name in os.listdir(control.path['dir_in_fitted']):
        if item_name.endswith('.pickle'):
            item_path = os.path.join(dir_in_fitted, item_name)
            if os.path.isfile(item_path):
                # the file holds a fitted model
                with open(item_path, 'rb') as f:
                    obj = pickle.load(f)
                model_spec = item_name.split('.')[0]
                fitted[model_spec] = obj
    print 'read %d fitted models' % len(fitted)

    # the ensemble prediction is the weighted average of the predictions from the fitted models
    ensemble_prediction = 0.0
    expert_prediction = {}  # Dict[model_spec, float]
    count = 0
    print 'expert model_spec --> weight --> expert prediction, if weight > 0.01'
    for model_spec, fitted_model in fitted.iteritems():
        count += 1
        # print 'predicting with expert %d of %d: %s' % (
        #     count,
        #     len(fitted),
        #     model_spec,
        # )
        expert_predictions = fitted_model.predict(prediction_features)
        assert len(expert_predictions) == 1
        the_expert_prediction = expert_predictions[0]
        weight = expert_weight[model_spec]
        ensemble_prediction += weight * the_expert_prediction
        expert_prediction[model_spec] = the_expert_prediction
        if weight > 0.01:
            print 'expert %30s %8.6f %f' % (model_spec, weight, the_expert_prediction)
    print 'ensemble prediction', ensemble_prediction

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
    for model_spec, p in expert_prediction.iteritems():
        delta = p - ensemble_prediction
        variance += expert_weight[model_spec] * delta * delta
    standard_deviation = math.sqrt(variance)
    print 'weighted standard deviation', standard_deviation

    # write the result as a CSV file
    assert control.arg.target == 'oasspread'
    assert len(prediction_features) == 1
    result = pd.DataFrame(
        data={
            'actual': prediction_features.iloc[0]['id_p_oasspread'],
            'ensemble_prediction': ensemble_prediction,
            'weighted_standard_deviation': standard_deviation,
            'reclassified_trade_type': reclassified_trade_type,
        },
        index=pd.Index(
            data=[control.arg.prediction_event_id],
            name='event_id'
        )
    )
    result.to_csv(control.path['out_ensemble_predictions'])
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
