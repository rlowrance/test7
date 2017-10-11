'''create a normalized data base (as a series of csv files) for all the fit_predict output

INVOCATION
  python noirmalize.py [--test] [--trace]

EXAMPLES OF INVOCATIONS
 python normalize.py

INPUTS
 WORKING/fit_predict-{ticker}-{cusip}-{hpset}-{date}/output.pickle

OUTPUTS
 WORKING/normalize/actual_values.csv:
   unique key: id_target_index  (index into  MidPredictor/data/{ticker}.csv)
   other columns: actual_value

 WORKING/normalize/importances.csv:
   unique key: ticker target_index modelspec:str predicted_feature
   other columns: importance

 WORKING/normalize/n_training_samples
   unique key: ticker target_index
   other columns: n_training_samples

 WORKING/normalize/predicted_values.csv:
   unique key: ticker target_index modelspec:str predicted_feature
   other columns: predicted_value

NOTES:
 target_index is the common unique Pandas DataFrame index into these files:
   MidPredictor/data/{ticker}.csv           # the record is always in this file
   WORKING/features-{ticker}/{cusip}.csv    # the record may or may not be in this file
   WORKING/targets-{ticker}/{cusip}.csv     # the record is always in this file

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''

from __future__ import division

import argparse
import collections
import datetime
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import applied_data_science.dirutility
import applied_data_science.lower_priority
import applied_data_science.pickle_utilities

from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

from seven.FitPredictOutput import Id, Payload, Record

import seven.path


KeyActualValues = collections.namedtuple(
    'KeyActualValues',
    'ticker target_index',
)
KeyImportances = collections.namedtuple(
    'KeyImportances',
    'ticker target_index model_spec_str feature_name'
)
KeyNTrainingSamples = collections.namedtuple(
    'KeyNTrainingsamples',
    'ticker target_index',
)
KeyPredictedValues = collections.namedtuple(
    'KeyPredictedValues',
    'ticker target_index model_spec_str predicted_feature',
)


def make_input_file_infos(dir_working):
    'yield input file paths and directory info'
    output = collections.namedtuple(
        'Output',
        'ticker cusip hpset date filepath'
    )
    # examine each directory including the top-level directory
    for subdir, dirs, files in os.walk(dir_working):
        for dir in dirs:
            if dir.startswith('fit_predict-'):
                fit_predict, ticker, cusip, hpset, year, month, day = dir.split('-')
                date = datetime.date(int(year), int(month), int(day))
                filepath = os.path.join(subdir, dir, 'output.pickle')
                yield output(ticker, cusip, hpset, date, filepath)


class Doit(object):
    def __init__(self, test=False, me='normalize', old=False):
        self.me = me
        self.test = test
        # define directories
        self.working_dir = seven.path.working()
        out_dir = os.path.join(self.working_dir, me + ('-test' if test else ''))
        # path to files abd durectirs
        paths_in = [
            input_file_info.filepath
            for input_file_info in make_input_file_infos(self.working_dir)
        ]
        self.out_dir = out_dir
        self.out_actual_values = os.path.join(out_dir, 'actual_values.csv')
        self.out_importances = os.path.join(out_dir, 'importances.csv')
        self.out_n_training_samples = os.path.join(out_dir, 'n_training_samples.csv')
        self.out_predicted_values = os.path.join(out_dir, 'predicted_values.csv')

        self.out_log = os.path.join(out_dir, '0log.txt')
        # used by Doit tasks
        self.actions = [
            'python %s.py' % me
        ]
        self.targets = [
            self.out_actual_values,
            self.out_importances,
            self.out_n_training_samples,
            self.out_predicted_values,
            self.out_log,
        ]
        self.file_dep = [self.me + '.py']
        for path_in in paths_in:
            self.file_dep.append(path_in)

    def __str__(self):
        for k, v in self.__dict__.iteritems():
            print 'doit.%s = %s' % (k, v)
        return self.__repr__()


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])
    arg.me = parser.prog.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    doit = Doit(test=arg.test)
    applied_data_science.dirutility.assure_exists(doit.out_dir)

    return Bunch(
        arg=arg,
        doit=doit,
        random_seed=random_seed,
        timer=Timer(),
    )


class ProcessObject(object):
    def __init__(self, ticker=None):
        self.ticker = ticker

        self.count = 0

        # outputs
        self.actual_values = {}
        self.importances = {}
        self.n_training_samples = {}
        self.predicted_values = {}

    def process(self, obj):
        'mututal output instance variables'
        def make_key_actual_values():
            return KeyActualValues(
                ticker=self.ticker,
                target_index=id.target_index,
            )

        def make_key_importances():
            return KeyImportances(
                ticker=self.ticker,
                target_index=id.target_index,
                model_spec_str=str(id.model_spec),
                feature_name=feature_name,
            )

        def make_key_n_training_samples():
            return KeyNTrainingSamples(
                ticker=self.ticker,
                target_index=id.target_index,
            )

        def make_key_predicted_values():
            return KeyPredictedValues(
                ticker=self.ticker,
                target_index=id.target_index,
                model_spec_str=str(id.model_spec),
                predicted_feature=id.predicted_feature,
            )

        verbose = False
        self.count += 1
        if verbose or self.count % 1000 == 1:
            print self.count, obj
        id = obj.id
        payload = obj.payload

        self.actual_values[make_key_actual_values()] = payload.actual_value
        self.n_training_samples[make_key_n_training_samples()] = payload.n_training_samples
        self.predicted_values[make_key_predicted_values()] = payload.predicted_value

        for feature_name, feature_importance in payload.importances.iteritems():
            self.importances[make_key_importances()] = feature_importance


def on_EOFError(e):
    print 'EOFError', e
    return


def on_ValueError(e):
    pdb.set_trace()
    print 'ValueError', e
    if e.args[0] == 'insecure string pickle':
        return
    else:
        raise e


def make_actual_values(all):
    ticker = []
    target_index = []
    actual_value = []
    for k, v in all.iteritems():
        ticker.append(k.ticker)
        target_index.append(k.target_index)
        actual_value.append(v)
    return pd.DataFrame(
        data={
            'ticker': ticker,
            'target_index': target_index,
            'actual_value': actual_value,
        },
    )


def make_importances(all):
    ticker = []
    target_index = []
    model_spec_str = []
    feature_name = []
    importance = []
    for k, v in all.iteritems():
        ticker.append(k.ticker)
        target_index.append(k.target_index)
        model_spec_str.append(k.model_spec_str)
        feature_name.append(k.feature_name)
        importance.append(v)
    return pd.DataFrame(
        data={
            'ticker': ticker,
            'target_index': target_index,
            'model_spec_str': model_spec_str,
            'feature_name': feature_name,
            'importance': importance,
        },
    )


def make_n_training_samples(all):
    ticker = []
    target_index = []
    n_training_samples = []
    for k, v in all.iteritems():
        ticker.append(k.ticker)
        target_index.append(k.target_index)
        n_training_samples.append(v)
    return pd.DataFrame(
        data={
            'ticker': ticker,
            'target_index': target_index,
            'n_training_samples': n_training_samples,
        },
    )


def make_predicted_values(all):
    ticker = []
    target_index = []
    model_spec_str = []
    predicted_feature = []
    predicted_value = []
    for k, v in all.iteritems():
        ticker.append(k.ticker)
        target_index.append(k.target_index)
        model_spec_str.append(k.model_spec_str)
        predicted_feature.append(k.predicted_feature)
        predicted_value.append(v)
    return pd.DataFrame(
        data={
            'ticker': ticker,
            'target_index': target_index,
            'model_spec_str': model_spec_str,
            'predicted_feature': predicted_feature,
            'predicted_value': predicted_value,
        },
    )


def do_work(control):
    'create csv file that summarizes all actual and predicted prices'
    # BODY STARTS HERE
    # determine training and testing transactions
    applied_data_science.lower_priority.lower_priority()  # try to give priority to interactive tasks

    all_actual_values = {}
    all_importances = {}
    all_n_training_samples = {}
    all_predicted_values = {}
    for input_file_info in make_input_file_infos(control.doit.working_dir):
        # read input file record by record
        process_object = ProcessObject(
            ticker=input_file_info.ticker,
        )
        applied_data_science.pickle_utilities.unpickle_file(
            path=input_file_info.filepath,
            process_unpickled_object=process_object.process,
            on_EOFError=on_EOFError,
            on_ValueError=on_ValueError,
        )
        print 'processed %d results from %s' % (process_object.count, input_file_info.filepath)
        all_actual_values.update(process_object.actual_values)
        all_importances.update(process_object.importances)
        all_n_training_samples.update(process_object.n_training_samples)
        all_predicted_values.update(process_object.predicted_values)
    # write the output csv files
    make_actual_values(all_actual_values).to_csv(control.doit.out_actual_values)
    make_importances(all_importances).to_csv(control.doit.out_importances)
    make_n_training_samples(all_n_training_samples).to_csv(control.doit.out_n_training_samples)
    make_predicted_values(all_predicted_values).to_csv(control.doit.out_predicted_values)
    return None


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.doit.out_log)  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    print control
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
        Id()
        Payload()
        Record()

    main(sys.argv)
