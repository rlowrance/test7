'''determine importances

INVOCATION
  python importances.py {issuer} {cusip} {target} {accuracy_date}
  {--debug} {--test} {--trace}
where
 issuer is the issuer symbol (ex: ORCL)
 cusip
 target is the target (ex: oasspread)
 accuracy_date is the date used to determine how to weight the ensemble model
 --debug means to call pdp.set_trace() if the execution call logging.error or logging.critical
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python importances.py AAPL 037833AJ9 oasspread 2017-07-20 --debug

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
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

    paths = seven.build.importances(
        arg.issuer,
        arg.cusip,
        arg.target,
        arg.accuracy_date,
        test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=Timer(),
    )


def make_accuracy_dict(accuracy_df):
    'return Dict[model_spec, float]'
    result = {}
    for index, row in accuracy_df.iterrows():
        result[index] = row['normalized_weight']
    return result


def make_importances_df(d):
    'convert d: Dict[feature_name, importance] into sorted data frame'
    df = pd.DataFrame()
    for feature_name, importance in d.iteritems():
        new_row = pd.DataFrame(
            data={
                'importance': importance,
            },
            index=pd.Series(
                data=[feature_name],
                name='feature_name'
            ),
        )
        df = df.append(new_row)
    result = df.sort_values(by='importance')
    return result


def do_work(control):
    'write predictions from fitted models to file system'
    accuracy_b, err = seven.read_csv.accuracy(control.path['in-accuracy B'])
    if err is not None:
        seven.logging.critical('unable to read accuracy csv: %s' % err)
        os.exit(1)

    accuracy_s, err = seven.read_csv.accuracy(control.path['in-accuracy S'])
    if err is not None:
        seven.logging.critical('unable to read accuracy csv: %s' % err)
        os.exit(1)

    # build accuracy dictionary
    accuracy = {}  # Dict[trade_type, Dict[model_spec, float]]
    accuracy['B'] = make_accuracy_dict(accuracy_b)
    accuracy['S'] = make_accuracy_dict(accuracy_s)

    importances = {}  # Dict[trade_type, Dict[feature_name, float]]
    model_names = ('elastic_net', 'naive', 'random_forests')
    for model_name in model_names:
        importances[model_name] = collections.defaultdict(float)
    count = 0
    for in_fit_path in control.path['list_in_fit_files']:
        count += 1
        if count % 1000 == 1:
            print 'reading fitted model %d of %d' % (
                count,
                len(control.path['list_in_fit_files']),
            )
        if control.arg.test and count > 1000:
            break
        # print in_fit_path
        with open(in_fit_path, 'rb') as f:
            model = pickle.load(f)
        model_importances = model.importances
        model_name = (
            'elastic_net' if isinstance(model, seven.models.ModelElasticNet) else
            'naive' if isinstance(model, seven.models.ModelNaive) else
            'random_forests' if isinstance(model, seven.models.ModelRandomForests) else
            None
        )
        assert model_name is not None
        if model_importances is None:
            continue
        # print model_importances
        in_fit_path_head, in_fit_path_tail = os.path.split(in_fit_path)
        model_spec, trade_type, file_suffix = in_fit_path_tail.split('.')
        for feature_name, model_feature_importance in model_importances.iteritems():
            # if model_feature_importance > 0:
            #     print feature_name, model_feature_importance
            importances[model_name][feature_name] += accuracy[trade_type][model_spec] * model_feature_importance

    for model_name in model_names:
        if len(importances[model_name]) == 0:
            continue
        df = make_importances_df(importances[model_name])
        print 'writing %d importance for model_name %s' % (len(df), model_name)
        df.to_csv(control.path['out_importances %s' % model_name])
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
