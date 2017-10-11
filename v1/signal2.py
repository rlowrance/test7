'''determine the signal produced by the ensemble predictions

INVOCATION
  python signal.py {issuer} {cusip} {target} {ensemble_date}
  {--debug} {--test} {--trace}
where
 issuer is the issuer symbol (ex: ORCL)
 cusip
 target is the target (ex: oasspread)
 ensemble_date is the date used to determine how to weight the ensemble model
 --debug means to call pdp.set_trace() if the execution call logging.error or logging.critical
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python signal.py AAPL 037833AJ9 oasspread 2017-07-20 --debug

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''

from __future__ import division

import argparse
import datetime
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
    parser.add_argument('ensemble_date', type=seven.arg_type.date)
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

    paths = seven.build.signal(
        arg.issuer,
        arg.cusip,
        arg.target,
        arg.ensemble_date,
        test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=Timer(),
    )


def do_work(control):
    'write summary of ensemble predictions'
    ensemble_predictions = pd.DataFrame()
    for in_file in control.path['list_in_files']:
        print in_file
        df, err = seven.read_csv.ensemble_predictions(in_file)
        if err is not None:
            seven.logging.critical('cannot read ensemble predictions file: %s' % err)
            os.exit(1)
        assert len(df) == 1
        for index, row in df.iterrows():
            next_row = pd.DataFrame(
                data={
                    'actual': row['actual'],
                    'ensemble_prediction': row['ensemble_prediction'],
                    'reclassified_trade_type': row['reclassified_trade_type'],
                    'weighted_standard_deviation': row['weighted_standard_deviation']
                },
                index=pd.Series(
                    data=[index],
                    name='predicted_event_id'
                )
            )
            ensemble_predictions = ensemble_predictions.append(next_row)
    print 'read %d ensemble predictions' % len(ensemble_predictions)

    # build the signal, which is the predictions for each trade
    # also show the actuals
    sorted_ensemble_predictions = ensemble_predictions.sort_index()
    signal = pd.DataFrame()
    predictions = {}  # Dict[reclassified_trade_type, float]
    actuals = {}      # Dcit[relcassified_trade_type, float]
    for index, row in sorted_ensemble_predictions.iterrows():
        prediction_event_id = seven.EventId.EventId.from_str(index)
        prediction_datetime = prediction_event_id.datetime()
        reclassified_trade_type = row['reclassified_trade_type']
        predictions[reclassified_trade_type] = row['ensemble_prediction']
        actuals[reclassified_trade_type] = row['actual']
        next_row = pd.DataFrame(
            data={
                'starting_datetime': prediction_datetime,
                'reclassified_trade_type': reclassified_trade_type,
                'actual_B': actuals.get('B', None),
                'actual_S': actuals.get('S', None),
                'predicted_B': predictions.get('B', None),
                'predicted_S': predictions.get('S', None),
            },
            index=pd.Series(
                data=[index],
                name='predicted_event_id',
            ),
        )
        signal = signal.append(next_row)
    signal.to_csv(control.path['out_signal'])
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
