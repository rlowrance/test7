'''fit cusips's trades on specified date using specified models

NOTE: This program looks at all trades for the CUSIP and related OTR CUSIPs from the
beginning of time. Thus trades 10 years ago are used to predict next oasspreads today.
That is way too much history. We should fix this when we have a streaming infrastructure.
Most likely, only the last 1000 or so trades are relevant.

INVOCATION
  python fit.py {ticker} {cusip} {target} {event_id} {hpset} {--test} {--trace}
where
 ticker is the ticker symbol (ex: orcl)
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 event_id is the identifier of a trace print. It has this format
   {yyyy}-{mm}-{dd}-{hh}-{mm}-{ss}-traceprint-{issuepriceid}.csv
 hpset in {gridN} defines the hyperparameter set
 effective_date: YYYY-MM-DD is the date of the trade
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution
 --verbose means to print a lot

EXAMPLES OF INVOCATION
 python fit.py AAPL 037833AJ9 oasspread 2017-07-20-09-06-46-traceprint-127987331 grid4 # first trace_print on the date

OLD EXAMPLE INVOCATIONS
 python fit.py AAPL 037833AG5 127076037 grid4 # from 2017-06-26
 68389XAC9 grid3 2016-12-01  # production
 68389XAS4 grid3 2016-11-01  # production
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
import seven.HpGrids
import seven.models
import seven.feature_makers
import seven.fit_predict_output
import seven.read_csv
import seven.target_maker

pp = pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('issuer', type=seven.arg_type.issuer)
    parser.add_argument('cusip', type=seven.arg_type.cusip)
    parser.add_argument('target', type=seven.arg_type.target)
    parser.add_argument('event_id', type=seven.arg_type.event_id)
    parser.add_argument('hpset', type=seven.arg_type.hpset)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = seven.build.fit(arg.issuer, arg.cusip, arg.target, arg.event_id, arg.hpset, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        module='fit.py',
        path=paths,
        random_seed=random_seed,
        timer=Timer(),
    )


def make_model(model_spec, random_seed):
    'return a constructed Model instance'
    model_constructor = (
        seven.models.ModelNaive if model_spec.name == 'n' else
        seven.models.ModelElasticNet if model_spec.name == 'en' else
        seven.models.ModelRandomForests if model_spec.name == 'rf' else
        None
    )
    assert model_constructor is not None
    model = model_constructor(model_spec, random_seed)
    return model


def read_csvOLD(path, parse_dates):
    'return (DataFrame, err)'
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=parse_dates)
        return (df, None)
    except ValueError as e:
        # The file contains just "" if there were no trades on the date
        # That happens on non-trading dates, like weekends
        # pd.read_csv throws ValueError("<parse_date column_name> is not in list")
        if str(e).endswith("is not in list"):
            return (None, 'skipping empty csv at %s' % path)
        else:
            raise e


def read_features(path):
    parse_dates = [
        prefix + name
        for name in ['effectivedate', 'effectivedatetime', 'effectivetime']
        for prefix in ['id_p_', 'id_otr1_']
    ]
    df = pd.read_csv(
        path,
        index_col=['issuepriceid'],
        low_memory=False,
        parse_dates=parse_dates,
    )
    return df


def do_work(control):
    'write predictions from fitted models to file system'
    # read all the input files and create the consolidated features dataframe
    unsorted_features = pd.DataFrame()
    for filename in control.path['list_in_features']:
        df = read_features(os.path.join(control.path['dir_in'], filename))
        assert len(df) == 1
        unsorted_features = unsorted_features.append(df)
    print 'read %d feature vectors from %d input files' % (
        len(unsorted_features),
        len(control.path['list_in_features'])
    )
    sorted_features = unsorted_features.sort_values('id_p_effectivedatetime')

    # get query and build the output directory
    query = read_features(control.path['in_query'])
    assert len(query) == 1
    query_filename_base, query_reclassified_trade_type, query_suffix = control.path['in_query'].split('.')

    # build the training targets which are the oasspreads
    # the oasspreads are carried as IDs in the features
    # drop all training samples that are not for the reclassified trade type
    # TODO: iterate over several targets (oasspread, oasspreadratio)
    sorted_targets = pd.DataFrame(
        data=sorted_features['id_p_%s' % control.arg.target],
        index=sorted_features.index,
    )

    # fit and write the fitted models
    count = collections.Counter()
    grid = seven.HpGrids.construct_HpGridN(control.arg.hpset)
    for model_spec in grid.iter_model_specs():
        print 'fitting model spec', model_spec
        count['fitting attempted'] += 1
        m = make_model(model_spec, control.random_seed)
        # the try/except code is needed because the scikit-learn functions may raise
        try:
            m.fit(
                training_features=sorted_features,
                training_targets=sorted_targets,
            )
        except seven.models.ExceptionFit as e:
            print 'exception during fitting %s %s: %s' % (control.arg.trade_id, str(model_spec), e)
            count['exception during fitting: %s' % e] += 1
        path_out = os.path.join(
            control.path['dir_out'],
            '%s.%s.pickle' % (
                str(model_spec),
                query_reclassified_trade_type,
            )
        )
        with open(path_out, 'wb') as f:
            pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)
        count['fitted models written'] += 1
        gc.collect()  # keep memory usage roughly constant to enable running multiple instances

    print 'counts'
    for k in sorted(count.keys()):
        print '%30s: %6d' % (k, count[k])
    return None


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
    # pp(control.path)
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
