'''fit cusips's trades on specified date using specified models

NOTE: This program looks at all trades for the CUSIP and related OTR CUSIPs from the
beginning of time. Thus trades 10 years ago are used to predict next oasspreads today.
That is way too much history. We should fix this when we have a streaming infrastructure.
Most likely, only the last 1000 or so trades are relevant.

INVOCATION
  python fit.py {ticker} {cusip} {trade_id} {hpset} {--test} {--trace}
where
 ticker is the ticker symbol (ex: orcl)
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 hpset in {gridN} defines the hyperparameter set
 effective_date: YYYY-MM-DD is the date of the trade
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution
 --verbose means to print a lot

EXAMPLES OF INVOCATION
 python fit.py AAPL 037833AG5 127076037 grid4 # from 2017-06-26

OLD EXAMPLE INVOCATIONS
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
import seven.models
# from seven import HpGrids
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
    parser.add_argument('trade_id', type=seven.arg_type.trade_id)
    parser.add_argument('hpset', type=seven.arg_type.hpset)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = seven.build.fit(arg.issuer, arg.cusip, arg.trade_id, arg.hpset, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        module='fit.py',
        path=paths,
        random_seed=random_seed,
        timer=Timer(),
    )


def read_input_files(input_paths, parse_dates):
    'return DataFrame containing concatenation of contents of the input files'
    result = pd.DataFrame()
    for in_path in input_paths:
        try:
            csv = pd.read_csv(
                in_path,
                index_col=0,
                parse_dates=parse_dates,
                )
        except ValueError as e:
            # thrown if the csv file is empty
            # if it is empty, the columns specified in parse_dates will not be present
            print 'exception ValueError(%s)' % e
            print 'skipping empty input file %s' % in_path
            continue
        result = result.append(csv, verify_integrity=True)
    print 'read %d rows from %d input files' % (len(result), len(input_paths))
    return result


def make_model(model_spec, random_seed):
    'return a constructed Model instance'
    model_constructor = (
        seven.models.ModelNaive if model_spec.name == 'n' else
        seven.models.ModelElasticNet if model_spec.name == 'en' else
        seven.models.ModelRandomForests if model_spec.name == 'rf' else
        None
    )
    if model_constructor is None:
        print 'error: bad model_spec.name %s' % model_spec.name
        pdb.set_trace()
    model = model_constructor(model_spec, 'target_oasspread', random_seed)
    return model


def read_csv(path, parse_dates):
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


def make_features_targets(in_paths):
    'return (features:DataFrame, targets:DataFrame)'
    basic_date_feature_names = ('effectivedate', 'effectivedatetime', 'effectivetime')
    parse_dates_features = [
        'id_%s_%s' % (source, basic_date_feature_name)
        for source in ('otr1', 'p')
        for basic_date_feature_name in basic_date_feature_names
    ]
    parse_dates_targets = [
        'id_%s' % basic_date_feature_name
        for basic_date_feature_name in basic_date_feature_names
    ]
    result_features = pd.DataFrame()
    result_targets = pd.DataFrame()
    for path_features, path_targets in in_paths:
        features, err = read_csv(path_features, parse_dates_features)
        if err is not None:
            print err
            continue
        targets, err = read_csv(path_targets, parse_dates_targets)
        if err is not None:
            print err
            continue
        assert len(features) == len(targets)
        result_features = result_features.append(features)
        result_targets = result_targets.append(targets)
    return result_features, result_targets


def do_work(control):
    'write predictions from fitted models to file system'
    list_in_features = control.path['list_in_features']
    list_in_targets = control.path['list_in_targets']
    assert len(list_in_features) == len(list_in_targets)
    features, targets = make_features_targets(zip(list_in_features, list_in_targets))
    print len(features), len(targets)
    assert len(features) == len(targets)
    for index in features.index:
        assert index in targets.index

    # fit and write
    count = collections.Counter()
    list_out_fitted = control.path['fitted_file_list']
    for i, out_fitted in enumerate(list_out_fitted):
        if control.arg.verbose or i % 100 == 0:
            print 'fit.py %s %s %s %s:  %4d of %4d' % (
                control.arg.issuer,
                control.arg.cusip,
                control.arg.trade_id,
                control.arg.hpset, 
                i,
                len(list_out_fitted),
            )
            print out_fitted
        out_fitted_pieces = out_fitted.split('\\')
        out_fitted_dir = '\\'.join(out_fitted_pieces[:-1])
        out_fitted_filename = out_fitted_pieces[-1]

        applied_data_science.dirutility.assure_exists(out_fitted_dir)

        model_spec_str, type_suffix = out_fitted_filename.split('.')
        model_spec = seven.ModelSpec.ModelSpec.make_from_str(model_spec_str)
        m = make_model(model_spec, control.random_seed)
        try:
            m.fit(
                training_features=features,
                training_targets=targets,
            )
            with open(out_fitted, 'wb') as f:
                pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)
            count['fitted and wrote'] += 1
        except seven.models.ExceptionFit as e:
            print 'exception during fitting %s %s: %s' % (control.arg.trade_id, model_spec_str, e)
            count['exception during fitting: %s' % e] += 1
        gc.collect()

    print 'counts'
    for reason in sorted(count.keys()):
        print reason, count[reason]

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
