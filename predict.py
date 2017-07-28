'''predict one query trade (identified by the predicted_trade_id) to all fitted models (identified by fitted_trade_id)

INVOCATION
  python predict.py {issuer} {cusip} {target} {predicted_event_id} {fitted_event_id} {--test} {--trace}
where
 issuer is the issuer symbol (ex: ORCL)
 hpset in {gridN} defines the hyperparameter set
 effective_date: YYYY-MM-DD is the date of the trade
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python predict.py AAPL 037833AJ9 oasspread 2017-07-20-09-06-46-traceprint-127987331 2017-07-19-15-58-14-traceprint-127978656
OLD INVOCATION EXAMPLES (obsolete)
 python predict.py AAPL 127084044 127076037  # on 2016-06-26 at 10:38:21 and 09:06:23
 python fit_predict.py ORCL 68389XAS4 grid3 2016-11-01  # production
 python fit_predict.py ORCL 68389XAS4 grid1 2016-11-01  # grid1 is a small mesh
 python fit_predict.py ORCL 68389XAS4 grid1 2013-08-01 --test  # 1 CUSIP, 15 trades
 python fit_predict.py ORCL 68389XAR6 grid1 2016-11-01  --test  # BUT no predictions, as XAR6 is usually missing oasspreads
 py fit_predict.py ORCL 68389XBM6 grid1 2016-11-01 --test # runs quickly

See build.py for input and output files.

An earlier version of this program did checkpoint restart, but this version does not.

Operation deployment.

See the file fit_predict.org for an explanation of the design of this program.

IDEA FOR FUTURE:
- invoke with trade number on the day as well.
  - Input files: training data (which is the query data) with all the features built out.
  - Output files: the predictions for each model spec for the particular trade. Maybe also the fitted model.

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import datetime
import gc
import numpy as np
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
    parser.add_argument('predicted_event_id', type=seven.arg_type.event_id)
    parser.add_argument('fitted_event_id', type=seven.arg_type.event_id)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cache', action='store_true')
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

    paths = seven.build.predict(
        arg.issuer,
        arg.cusip,
        arg.target,
        arg.predicted_event_id,
        arg.fitted_event_id,
        test=arg.test,
        )
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    with open(seven.build.traceinfo(arg.issuer)['out_by_trace_index'], 'rb') as f:
        traceinfo_by_trade_id = pickle.load(f)

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=Timer(),
        traceinfo_by_trade_id=traceinfo_by_trade_id,
    )


def date_to_str(date):
    'convert datetime.date to string'
    return '%4d-%02d-%02d' % (date.year, date.month, date.day)


def do_work(control):
    'write predictions from fitted models to file system'
    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()
    target_field_name = 'id_p_%s' % control.arg.target
    query_path = control.path['in_prediction_event']
    query_head, query_filename = os.path.split(query_path)
    query_reclassified_trade_type = query_filename.split('.')[1]
    print 'query_path:', query_path
    print 'query reclassified trade type:', query_reclassified_trade_type
    query_features, err = seven.read_csv.features_targets(control.path['in_prediction_event'])
    if err is not None:
        err = 'error from seven.read_csv.features_targets: %s' % err
        seven.logging.critical(err)
    assert len(query_features) == 1

    actual = query_features.iloc[0][target_field_name]
    result = pd.DataFrame()
    counter = collections.Counter()
    for fitted_path in control.path['list_in_fitted']:
        fitted_head, fitted_filename = os.path.split(fitted_path)
        print 'predicting with', fitted_filename
        fitted_reclassified_trade_type = query_filename.split('.')[1]
        assert fitted_reclassified_trade_type == fitted_reclassified_trade_type
        with open(fitted_path, 'rb') as f:
            fitted_model = pickle.load(f)
        counter['predictions attempted']

        predictions = fitted_model.predict(query_features)  # : Union(np.ndarray, List[pd.Series])
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 1

        prediction = predictions[0]
        seven.logging.error_if_nan(actual, 'actual')
        seven.logging.error_if_nan(prediction, 'prediction')

        new_row = pd.DataFrame(
            data={
                'prediction': prediction,
                'actual': actual,
            },
            index=pd.Series([str(fitted_model.model_spec)], name='model_spec'),
        )
        result = result.append(new_row)
        counter['predictions made']
        gc.collect()

    print 'counters'
    for k, v in counter.iteritems():
        print '%60s: %d' % (k, v)
    print
    print 'made %d predictions' % len(result)
    result.to_csv(control.path['out_predictions'])

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
