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
import collections
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


def synthetic_query(query_features, trade_type):
    'return query_features, but with the trade_type reset'
    result = pd.DataFrame()
    for index, row in query_features.iterrows():
        # there is only 1 row
        row['p_trade_type_is_B'] = 0
        row['p_trade_type_is_D'] = 0
        row['p_trade_type_is_S'] = 0
        row['p_trade_type_is_%s' % trade_type] = 1
        if row['id_otr1_cusip'] == row['id_p_cusip'] and row['id_otr1_effectivedatetime'] == row['id_p_effectivedatetime']:
            pdb.set_trace()
            row['otr1_trade_type_is_B'] = 0
            row['otr1_trade_type_is_D'] = 0
            row['otr1_trade_type_is_S'] = 0
            row['otr1_trade_type_is_%s' % trade_type] = 1
        result = result.append(row)
    return result


def do_work(control):
    'write predictions from fitted models to file system'
    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    query_features, query_targets = read_query(control)
    actual = query_targets.iloc[0]['target_oasspread']
    result = pd.DataFrame()
    counter = collections.Counter()
    for dirpath, dirnames, filenames in os.walk(control.path['dir_in']):
        print dirpath
        for filename in filenames:  # the filenames are the model_spec_str values used to fit the models
            if filename == '0log.txt':
                continue  # skip log file
            print 'predicting with model in', filename
            with open(os.path.join(dirpath, filename), 'rb') as f:
                fitted_model = pickle.load(f)
            modelspec = filename.split('.')[0]

            # predict using the actual query data
            predictions = fitted_model.predict(query_features)
            assert len(predictions) == 1
            prediction = predictions[0]
            # pretend that the query was of each possible trade type, and predict those trade types
            synthetic_prediction = {}  # Dict[trade_type, prediction]
            all_the_same = True
            for trade_type in ('B', 'D', 'S'):
                synthetic_predictions = fitted_model.predict(synthetic_query(query_features, trade_type))
                assert len(synthetic_predictions) == 1
                one_synthetic_prediction = synthetic_predictions[0]
                synthetic_prediction[trade_type] = one_synthetic_prediction
                all_the_same = all_the_same and (one_synthetic_prediction == prediction)
            if all_the_same:
                counter['all B, D, S, and the natural predictions are the same'] += 1
            else:
                counter['some B, D, S, and the natural prediction differ'] += 1
            new_row = pd.DataFrame(
                data={
                    'prediction': prediction,
                    'actual': actual,
                    'prediction_B': synthetic_prediction['B'],
                    'prediction_D': synthetic_prediction['D'],
                    'prediction_S': synthetic_prediction['S'],
                },
                index=[modelspec], 
            )
            result = result.append(new_row)
            gc.collect()
    print 'analysis of impact of synthetic predictions'
    for k, v in counter.iteritems():
        print '%60s: %d' % (k, v)
    print
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
