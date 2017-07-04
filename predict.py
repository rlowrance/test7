'''predict one query trade (identified by the predicted_trade_id) to all fitted models (identified by fitted_trade_id)

INVOCATION
  python predict.py {issuer} {cusip} {predicted_trade_id} {fitted_trade_id} {--test} {--trace}
where
 issuer is the issuer symbol (ex: ORCL)
 hpset in {gridN} defines the hyperparameter set
 effective_date: YYYY-MM-DD is the date of the trade
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
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
'''

from __future__ import division

import argparse
import cPickle as pickle
import datetime
import gc
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
    parser.add_argument('predicted_trade_id', type=seven.arg_type.trade_id)
    parser.add_argument('fitted_trade_id', type=seven.arg_type.trade_id)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = seven.build.predict(arg.issuer, arg.predicted_trade_id, arg.fitted_trade_id, test=arg.test)
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


def read_query(control):
    'return (features:Series, targets: Series)'
    def get_cusip_date(trade_id):
        info = control.traceinfo_by_trade_id[int(trade_id)]
        cusip = info['cusip']
        date = date_to_str(info['effective_date'])
        return cusip, date

    prediction_cusip, prediction_date = get_cusip_date(control.arg.predicted_trade_id)

    def get_query(what):
        csv = seven.read_csv.working(
            'features_targets',
            control.arg.issuer,
            prediction_cusip,
            prediction_date,
            '%s.csv' % what,
        )
        predicted_trade_id = int(control.arg.predicted_trade_id)
        if predicted_trade_id in csv.index:
            result = csv.loc[[predicted_trade_id]]
            return result
        else:
            print 'prediction_trade_id %s not in csv file' % predicted_trade_id
            print 'found these trade_id instead', csv.index
            print what
            pdb.set_trace()

    return get_query('features'), get_query('targets')


def do_work(control):
    'write predictions from fitted models to file system'
    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    query_features, query_targets = read_query(control)
    actual = query_targets.iloc[0]['target_oasspread']
    result = pd.DataFrame()
    for dirpath, dirnames, filenames in os.walk(control.path['dir_in']):
        print dirpath
        for modelspec in filenames:  # the filenames are the model_spec_str values used to fit the models
            if modelspec == '0log.txt':
                continue  # skip log file
            print 'predicting with fitted model_spec', modelspec
            with open(os.path.join(dirpath, modelspec), 'rb') as f:
                fitted_model = pickle.load(f)
            predictions = fitted_model.predict(query_features)
            assert len(predictions) == 1
            new_row = pd.DataFrame(
                data={
                    'prediction': predictions[0],
                    'actual': actual,
                },
                index=[modelspec],
            )
            result = result.append(new_row)
            gc.collect()
    print 'write result'
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
