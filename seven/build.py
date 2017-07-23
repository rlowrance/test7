'''determine how to build a program: sources, targets, commands

Each function takes these args:
positional1, position2, ... : positional arguments on the command line
executable=String           : name of the executable, without the py suffix; ex: fit_predict
test=                       : whether the output goes to the test directory

Each function returns a dictionary with these keys:
'dir_out': path to the output directory, which depends on the positional args and whether test=True is present
'in*':  path to an input file
'out*':  path to an output file
'executable':  name of executable: example: fit_predict
'dep*:  another dependency to the build process; the 'in*" items are also dependencies
'command': invocation command

There is one such function for each program.

MAINTENANCE NOTES:
1. Pandas is not compatible with scons. This module is read by scons scripts to determine the DAG
that governs the build. You must find a way to avoid using Pandas.

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''

from __future__ import division

import collections
import copy
import cPickle as pickle
import datetime
import os
import pdb
import pprint
import sys
import unittest

# imports from seven/
import HpGrids
import path


pp = pprint.pprint

representative_orcl_cusip = '68389XAS4'


def lookup_issuer(isin):
    'return issuer for the isin (CUSIP) or fail'
    raise LookupError('isin %s: cannot determine issuer' % isin)


def make_scons(paths):
    'return Dict with items sources, targets, command, based on the paths'
    # these are the values needed by sconc to control the DAG used to build the data
    # def select(f):
    #     return [v for k, v in paths.items() if f(k)]

    # result = {
    #     'commands': [paths['command']],
    #     'sources': select(lambda k: k.startswith('in_') or k.startswith('executable') or k.startswith('dep_')),
    #     'targets': select(lambda k: k.startswith('out_')),
    # }

    sources = []
    targets = []
    for k, v in paths.iteritems():
        if k.startswith('in_') or k.startswith('executable') or k.startswith('dep_'):
            sources.append(v)
        elif k.startswith('out_'):
            targets.append(v)
        elif k.startswith('list_in'):
            for item in v:
                sources.append(item)
        elif k.startswith('list_out'):
            for item in v:
                targets.append(item)
        else:
            pass

    result = {
        'commands': [paths['command']],
        'sources': sources,
        'targets': targets,
    }

    return result

########################################################################################
# utitlity functions
########################################################################################


def as_datetime_date(x):
    if isinstance(x, datetime.date):
        return x
    if isinstance(x, str):
        year, month, day = x.split('-')
        return datetime.date(int(year), int(month), int(day))
    print 'type not handled', type(x), x
    pdb.set_trace()


########################################################################################
# program-specific builds
########################################################################################


def accuracy(issuer, cusip, trade_date, executable='accuracy', test=False):
    dir_working = path.working()
    dir_out = os.path.join(
        dir_working,
        executable,
        cusip,
        trade_date,
    )

    # retrieve the infos for the cusip and trade_date
    traceinfo_path = traceinfo(issuer)['out_by_trade_date']
    with open(traceinfo_path, 'rb') as f:
        traceinfos = pickle.load(f)  # dictionary
    trade_date_datetime_date = as_datetime_date(trade_date)
    if trade_date_datetime_date not in traceinfos:
        print 'error: trade_date %s is not in the traceinfo file' % trade_date
        pdb.set_trace()
    infos_date = traceinfos[trade_date_datetime_date]  # all the trades on the date
    infos_date_cusip = filter(
        lambda info: info['cusip'] == cusip,
        infos_date
    )
    if len(infos_date_cusip) == 0:
        print 'no traceinfo for invocation paramaters', issuer, cusip, trade_date
        pdb.set_trace()

    # deterime all input files, which are the files with the predictions
    list_in_files = []
    for info in infos_date_cusip:  # for each relevant trace print
        predicted_trade_id = info['issuepriceid']
        dir_trade_id = os.path.join(dir_working, 'predict', str(predicted_trade_id))
        if not os.path.isdir(dir_trade_id):
            print 'accuracy', issuer, cusip, trade_date
            print 'ERROR: directory not present:', dir_trade_id
            print 'relevant info'
            pp(info)
            pdb.set_trace()
        for dirpath, dirnames, filenames in os.walk(dir_trade_id):
            assert len(dirnames) == 1
            path_data = os.path.join(dir_trade_id, dirnames[0], 'predictions.csv')
            list_in_files.append(path_data)
            break  # we expect only one subdirectory

    result = {
        'list_in_files': list_in_files,

        'out_weights': os.path.join(dir_out, 'weights.pickle'),  # Dict[model_spec, weight]
        'out_weights_csv': os.path.join(dir_out, 'weights.csv'),  # columns: model_spec, weight
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s %s %s' % (executable, issuer, cusip, trade_date)
    }
    return result


class TestAccuracy(unittest.TestCase):
    def test(self):
        x = accuracy('AAPL', '037833AG5', '2017-06-26')
        self.assertTrue(isinstance(x, dict))


def buildinfo(executable='buildinfo', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = path.working()
    dir_out_base = os.path.join(
        dir_working,
        executable,
    )
    dir_out = dir_out_base + '-test' if test else dir_out_base
    dir_in = os.path.join(path.midpredictor(), 'automatic_feeds')

    list_in_trace = []
    for dirpath, dirnams, filenames in os.walk(dir_in):
        for filename in filenames:
            if filename.startswith('trace_') and filename.endswith('.csv'):
                # for now, ignore the date-specific trace files
                filename_base = filename.split('.')[0]
                if len(filename_base.split('_')) > 2:
                    pass  # for example, trace_AAPL_2017__07_01.csv is skipped
                else:
                    filepath = os.path.join(dir_in, filename)
                    list_in_trace.append(filepath)

    result = {
        'in_secmaster': os.path.join(dir_in, 'secmaster.csv'),
        'list_in_trace': list_in_trace,

        'out_log': os.path.join(dir_out, '0log.txt'),
        'out_db': os.path.join(dir_working, 'info.sqlite'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py' % executable,
    }
    return result


def cusips(ticker, executable='cusips', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = path.working()
    dir_out = os.path.join(dir_working, '%s-%s%s' % (
        executable,
        ticker,
        ('-test' if test else ''),
        )
    )

    result = {
        'in_trace': path.input(ticker=ticker, logical_name='trace'),

        'out_counts_by_month': os.path.join(dir_out, 'counts_by_month.csv'),
        'out_cusips': os.path.join(dir_out, '%s.pickle' % ticker),  # Dict[cusip, count]
        'out_first_last': os.path.join(dir_out, 'first_last.csv'),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s' % (executable, ticker),
    }
    return result


def ensemble_predictions(issuer, cusip, trade_date, executable='ensemble_predictions', test=False):
    def get_prior_info(traceinfos, cusip, trade_date):
        'return info for the last trade of the cusip on the date just prior to the trade_date'
        prior_infos = [
            info
            for key, infos in traceinfos.iteritems()
            for info in infos
            if info['effective_date'] < as_datetime_date(trade_date)
            if info['cusip'] == cusip
        ]
        if len(prior_infos) > 0:
            infos_sorted = sorted(prior_infos, key=lambda info: info['effective_datetime'])
            return infos_sorted[-1]
        else:
            print 'build.ensemble_predictions: no prior trade for %s %s %s' % (
                issuer,
                cusip,
                trade_date,
            )
            pdb.set_trace()
            
    dir_working = path.working()
    dir_out = os.path.join(
        dir_working,
        executable,
        cusip,
        trade_date,
    )

    # determine the fitted model to use
    # it's the one fitted for trade immediately before the trade date
    # if cusip == '68389XAU9' and trade_date == '2017-07-10':
    #     print 'build.ensemble_predictions: found it'
    #     pdb.set_trace()
    traceinfo_path = traceinfo(issuer)['out_by_trade_date']
    with open(traceinfo_path, 'rb') as f:
        traceinfos_by_trade_date = pickle.load(f)  # dictionary
    prior_info = get_prior_info(traceinfos_by_trade_date, cusip, trade_date)
    
    # prior_trade_info = traceinfos_cusip_sorted[-1]
    prior_date = str(prior_info['effective_date'])
    prior_trade_id = str(prior_info['issuepriceid'])
    dir_in_fitted = os.path.join(dir_working, 'fit', issuer, cusip, prior_trade_id)

    result = {
        'in_accuracy': os.path.join(dir_working, 'accuracy', cusip, prior_date, 'weights.pickle'),
        'in_fitted': os.path.join(dir_in_fitted, '0log.txt'),  # proxy for many pickle files
        'in_query_features': os.path.join(dir_working, 'features_targets', issuer, cusip, trade_date, 'features.csv'),
        'in_query_targets': os.path.join(dir_working, 'features_targets', issuer, cusip, trade_date, 'targets.csv'),

        # 'out_expert_predictions': os.path.join(dir_out, 'expert_predictions.csv'),
        'out_ensemble_predictions': os.path.join(dir_out, 'ensemble_predictions.csv'),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_in_fitted': dir_in_fitted,  # contains a pickle file for each model spec that could be fitted
        'dir_out': dir_out,
        'command': 'python %s.py %s %s %s' % (executable, issuer, cusip, trade_date)
    }
    return result


class TestEnsemblePredictions(unittest.TestCase):
    def test(self):
        x = ensemble_predictions('AAPL', '037833AG5', '2017-06-27')
        self.assertTrue(isinstance(x, dict))


def features_targets(issuer, cusip, effective_date, executable='features_targets', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = path.working()
    dir_out = os.path.join(
        dir_working,
        '%s' % executable,
        '%s' % issuer,
        '%s' % cusip,
        '%s%s' % (effective_date, ('-test' if test else '')),
    )

    # NOTE: excludes all the files needed to buld the features
    # these are in MidPredictor/automatic_feeds and the actual files depend on the {ticker} and {cusip}
    # The dependency on map_cusip_ticker.csv is not reflected
    # issuer = GetSecurityMasterInfo.GetSecurityMasterInfo().issuer_for_cusip(cusip)
    result = {
        'in_trace': path.input(issuer, 'trace'),
        'in_traceinfo': os.path.join(dir_working, 'traceinfo', issuer, 'summary.pickle'),
        # 'in_otr': seven.path.input(issuer, 'otr'),
        # these are source code dependencies beyond the executable
        'in_feature_makers': os.path.join(path.src(), 'seven', 'feature_makers.py'),
        'in_target_maker': os.path.join(path.src(), 'seven', 'target_maker.py'),

        'out_features': os.path.join(dir_out, 'features.csv'),  # contains targets as IDs
        'out_log': os.path.join(dir_out, '0log.txt'),
        'optional_out_cache': os.path.join(dir_out, '1cache.pickle'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s %s %s' % (executable, issuer, cusip, effective_date),
    }
    return result


class Test_features_targets(unittest.TestCase):
    def test(self):
        'test completion'
        verbose = False
        Test = collections.namedtuple('Test', 'issuer cusip effective_date')
        tests = (
            Test('APPL', '037833AG5', '2017-06-26'),
        )
        for test in tests:
            issuer, cusip, effective_date = test
            b = features_targets(issuer, cusip, effective_date)
            if verbose:
                print b['command']
            self.assertTrue(True)


def fit(issuer, cusip, trade_id, hpset,
        executable='fit', test=False, infos_by_trace_index=None, verbose=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    def file_length(path):
        'return number of lines'
        with open(path, 'r') as f:
            for index, line in enumerate(f):
                pass

    dir_working = path.working()
    dir_out_base = os.path.join(
        dir_working,
        '%s' % executable,
        '%s' % issuer,
        '%s' % cusip,
        '%s' % trade_id,
    )
    dir_out = (
        dir_out_base + '-test' if test else
        dir_out_base
    )

    # NOTE: excludes all the files needed to buld the features
    # these are in MidPredictor/automatic_feeds and the actual files depend on the {ticker} and {cusip}
    # The dependency on map_cusip_ticker.csv is not reflected

    # determine all output files
    grid = HpGrids.construct_HpGridN(hpset)
    list_out_fitted = []
    max_n_trades_back = 0
    for model_spec in grid.iter_model_specs():
        if model_spec.n_trades_back is not None:  # the naive model does not have n_trades_back
            max_n_trades_back = max(max_n_trades_back, model_spec.n_trades_back)
        next_path = os.path.join(
            dir_out,
            '%s.pickle' % model_spec,
        )
        list_out_fitted.append(next_path)

    # determine all input files
    # the input files are grouped by date
    with open(traceinfo(issuer)['out_by_trace_index'], 'rb') as f:
        infos_by_trace_index = pickle.load(f)
    first_feature_date = infos_by_trace_index[int(trade_id)]['effective_date']

    # gbi = GetBuildInfo.GetBuildInfo(issuer)
    # current_date = gbi.get_effectivedate(int(trade_id))
    # TODO: adjust so that we use only info from traceinfo, not from the working file system.
    # (If we use info from the working file system, we can't build everything from scratch.)
    list_in_features = []   # will be file names working/{issuer}/{cusip}/{DATE}/features.csv
    list_in_targets = []    # will be "                                         /"
    trace_indices_to_read = set()
    while len(trace_indices_to_read) <= max_n_trades_back:
        features_targets_dir = os.path.join(
            dir_working,
            'features_targets',
            issuer,
            cusip,
            '%s' % first_feature_date,
        )
        if verbose:
            print 'have found %d trace indices to use for fitting' % len(trace_indices_to_read)
            print 'looking for common trace indices in %s' % features_targets_dir
        filepath = os.path.join(features_targets_dir, 'common_trace_indices.txt')
        if not os.path.isfile(filepath):
            print 'does not exist', filepath
            print 'HINT: try running scons to build feature sets'
            print 'build.fit: not enough features'
            print 'arguments', issuer, cusip, trade_id, hpset
            print 'hpset requires %d historic feature sets' % max_n_trades_back
            print 'found only %d feature sets' % len(trace_indices_to_read)
            print 'FIX: run features_targets.py on earlier dates, starting with %s' % current_date
            pdb.set_trace()
            sys.exit(1)
        if verbose:
            print 'fit.py will use features and targets from date %s' % current_date
        with open(filepath, 'r') as f:
            for index, trace_index in enumerate(f):
                trace_indices_to_read.add(int(trace_index[:-1]))  # drop final \n
        list_in_features.append(os.path.join(features_targets_dir, 'features.csv'))
        list_in_targets.append(os.path.join(features_targets_dir, 'targets.csv'))
        first_feature_date -= datetime.timedelta(1)  # 1 day back

    result = {
        'list_in_features': list_in_features,
        'list_in_targets': list_in_targets,

        # 'list_out_fitted': list_out_fitted,
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s %s %s %s' % (executable, issuer, cusip, trade_id, hpset),

        'max_n_trades_back': max_n_trades_back,
        'fitted_file_list': list_out_fitted,
        'first_feature_date': first_feature_date,
    }
    return result


class Test_fit(unittest.TestCase):
    def test(self):
        verbose = False
        Test = collections.namedtuple('Test', 'issuer cusip trade_id hpset')
        tests = (
            Test('AAPL', '037833AG5', '127076037', 'grid4'),
        )
        for test in tests:
            issuer, cusip, trade_id, hpset = test
            b = fit(issuer, cusip, trade_id, hpset)
            if verbose:
                print b['command']


def fit_predict(ticker, cusip, hpset, effective_date, executable='fit_predict', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = path.working()
    dir_out = os.path.join(
        dir_working,
        '%s' % executable,
        '%s' % ticker,
        '%s' % cusip,
        '%s' % hpset,
        '%s%s' % (effective_date, ('-test' if test else '')),
    )

    # NOTE: excludes all the files needed to buld the features
    # these are in MidPredictor/automatic_feeds and the actual files depend on the {ticker} and {cusip}
    result = {
        'in_trace': path.input(ticker, 'trace'),

        'out_importances': os.path.join(dir_out, 'importances.pickle'),
        'out_predictions': os.path.join(dir_out, 'predictions.pickle'),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s %s %s %s' % (executable, ticker, cusip, hpset, effective_date),
    }
    return result


def fit_predict_v2(ticker, cusip, hpset, effective_date, executable='fit_predict', test=False, syntheticactual=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = path.working()
    dir_out = os.path.join(dir_working, '%s-%s-%s-%s-%s%s%s' % (
        executable,
        ticker,
        cusip,
        hpset,
        effective_date,
        ('-test' if test else ''),
        ('-syntheticactual' if syntheticactual else ''),
    ))

    result = {
        'in_trace': path.input(ticker, 'trace'),

        'out_importances': os.path.join(dir_out, 'importances.pickle'),
        'out_predictions': os.path.join(dir_out, 'predictions.pickle'),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s %s %s %s' % (executable, ticker, cusip, hpset, effective_date),
    }
    return result


def predict(issuer, prediction_trade_id, fitted_trade_id, executable='predict', test=False):
    if False and issuer == 'AMZN':
        print 'predict', issuer, prediction_trade_id, fitted_trade_id
        pdb.set_trace()
    dir_working = path.working()
    dir_out = os.path.join(
        dir_working,
        executable,
        prediction_trade_id,
        fitted_trade_id,
    )
    traceinfo_path = traceinfo(issuer)['out_by_trace_index']
    with open(traceinfo_path, 'rb') as f:
        traceinfos = pickle.load(f)

    info = traceinfos.get(int(prediction_trade_id))
    if info is None:
        pdb.set_trace()
        raise ValueError('prediction_trade_it %s is not the trace info for issuer %s' % (
            prediction_trade_id,
            issuer,
        ))
    prediction_cusip = info['cusip']

    dir_in = os.path.join(dir_working, 'fit', issuer, prediction_cusip, fitted_trade_id)

    result = {
        'in_fitted': os.path.join(dir_in, '0log.txt'),  # proxy for all the model_spec files

        'out_predictions': os.path.join(dir_out, 'predictions.csv'),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_in': dir_in,
        'dir_out': dir_out,
        'command': 'python %s.py %s %s %s' % (executable, issuer, prediction_trade_id, fitted_trade_id)
    }
    return result


class TestPredict(unittest.TestCase):
    def test(self):
        'just test run to completion'
        predict('AAPL', '127084044', '127076037')
        self.assertTrue(True)


def report_compare_models2(ticker, cusip, hpset, executable='report_compare_models2', test=False):
    dir_working = path.working()
    dir_out = os.path.join(dir_working, '%s-%s-%s-%s%s' % (
        executable,
        ticker,
        cusip,
        hpset,
        ('-test' if test else ''),
        )
    )
    in_predictions = []
    in_importances = []
    for root, dirs, files in os.walk(dir_working):
        for dir in dirs:
            if dir.startswith('fit_predict-%s-%s-%s' % (ticker, cusip, hpset)):
                # in_files.append(os.path.join(root, dir, 'importances.csv'))
                path_predictions = os.path.join(root, dir, 'predictions.csv')
                if os.path.isfile(path_predictions):
                    in_predictions.append(path_predictions)
                path_importances = os.path.join(root, dir, 'importances.csv')
                if os.path.isfile(path_importances):
                    in_importances.append(path_importances)

    result = {
        'in_importances': in_importances,
        'in_predictions': in_predictions,

        'out_accuracy_modelspec_targetfeaturename': os.path.join(dir_out, 'accuracy_modelspec_targetfeaturename.txt'),
        'out_accuracy_modelspec': os.path.join(dir_out, 'accuracy_modelspec.txt'),
        'out_accuracy_queryindex': os.path.join(dir_out, 'accuracy_queryindex.txt'),
        'out_accuracy_targetfeaturename': os.path.join(dir_out, 'accuracy_targetfeaturename.txt'),
        'out_accuracy_targetfeaturename_modelspecstr': os.path.join(dir_out, 'accuracy_targetfeaturename_modelspecstr.txt'),
        'out_accuracy_queryindex_targetfeaturename': os.path.join(dir_out, 'accuracy_queryindex_targetfeaturename.txt'),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s %s %s' % (executable, ticker, cusip, hpset)
    }
    return result


def report03_compare_predictions(
    ticker,
    cusip,
    hpset,
    executable='report03_compare_predictions',
    test=False,
    testinput=False,
):
    dir_working = path.working()
    dir_out = os.path.join(dir_working, executable, '%s-%s-%s%s' % (
        ticker,
        cusip,
        hpset,
        ('-test' if test else ''),
        )
    )

    in_predictions = []  # build up paths to files containing the predictions
    root_dir = os.path.join(dir_working, 'fit_predict', ticker, cusip, hpset)
    for root, dirs, files in os.walk(root_dir):
        # should contain only dirs {effective_date} and {effective_date}-test
        for dir in dirs:
            # dir is {effective_datetime}[-test]
            if testinput:
                if dir.endswith('-test'):
                    in_prediction = os.path.join(root_dir, dir, 'predictions.pickle')
                    in_predictions.append(in_prediction)
            else:
                if not dir.endswith('-test'):
                    in_prediction = os.path.join(root_dir, dir, 'predictions.pickle')
                    in_predictions.append(in_prediction)

    result = {
        'in_predictions': in_predictions,

        'out_accuracy_modelspec': os.path.join(dir_out, 'accuracy_modelspec.txt'),
        'out_accuracy_modelspec_csv': os.path.join(dir_out, 'accuracy_modelspec.csv'),
        'out_accuracy_targetfeature_modelspec': os.path.join(dir_out, 'accuracy_targetfeature_modelspec.txt'),
        'out_accuracy_targetfeature_modelspec_csv': os.path.join(dir_out, 'accuracy_targetfeature_modelspec.csv'),
        'out_details': os.path.join(dir_out, 'details.txt'),
        'out_details_csv': os.path.join(dir_out, 'details.csv'),
        'out_mae_modelspec': os.path.join(dir_out, 'mae_modelspec.csv'),
        'out_log': os.path.join(dir_out, '0log.txt'),
        'out_large_absolute_errors': os.path.join(dir_out, 'large_absolute_errors.csv'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s %s %s' % (executable, ticker, cusip, hpset)
    }
    return result


def report04_predictions(ticker, cusip, hpset, executable='report04_predictions', test=False, testinput=False):
    dir_working = path.working()
    dir_out = os.path.join(dir_working, executable, '%s-%s-%s%s' % (
        ticker,
        cusip,
        hpset,
        ('-test' if test else ''),
        )
    )

    in_predictions = []  # build up paths to files containing the predictions
    root_dir = os.path.join(dir_working, 'fit_predict', ticker, cusip, hpset)
    for root, dirs, files in os.walk(root_dir):
        # should contain only dirs {effective_date} and {effective_date}-test
        for dir in dirs:
            # dir is {effective_datetime}[-test]
            if testinput:
                if dir.endswith('-test'):
                    in_prediction = os.path.join(root_dir, dir, 'predictions.pickle')
                    in_predictions.append(in_prediction)
            else:
                if not dir.endswith('-test'):
                    in_prediction = os.path.join(root_dir, dir, 'predictions.pickle')
                    in_predictions.append(in_prediction)

    result = {
        'in_predictions': in_predictions,

        'out_predictions_zip': os.path.join(dir_out, 'predictions-%s-%s-%s.zip' % (ticker, cusip, hpset)),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s %s %s' % (executable, ticker, cusip, hpset)
    }
    return result


def report05_compare_importances(ticker, cusip, hpset, n, executable='report05_compare_importances', test=False, testinput=False):
    dir_working = path.working()
    dir_out = os.path.join(dir_working, executable, '%s-%s-%s-%s%s' % (
        ticker,
        cusip,
        hpset,
        str(n),
        ('-test' if test else ''),
        )
    )

    in_importances = []  # build up paths to files containing the predictions
    root_dir = os.path.join(dir_working, 'fit_predict', ticker, cusip, hpset)
    for root, dirs, files in os.walk(root_dir):
        # should contain only dirs {effective_date} and {effective_date}-test
        for dir in dirs:
            # dir is {effective_datetime}[-test]
            if testinput:
                if dir.endswith('-test'):
                    in_importance = os.path.join(root_dir, dir, 'importances.pickle')
                    in_importances.append(in_importance)
            else:
                if not dir.endswith('-test'):
                    in_importance = os.path.join(root_dir, dir, 'importances.pickle')
                    in_importances.append(in_importance)

    out_importance_d = {}
    for index in xrange(n):
        key = 'out_importance_%d' % (index + 1)
        filename = 'importance_%04d.txt' % (index + 1)
        out_importance_d[key] = os.path.join(dir_out, filename)
        key_csv = 'out_importance_%d_csv' % (index + 1)
        filename_csv = 'importance_%04d.csv' % (index + 1)
        out_importance_d[key_csv] = os.path.join(dir_out, filename_csv)

    other_result = {
        'in_importances': in_importances,
        'in_mae_modelspec': report03_compare_predictions(ticker, cusip, hpset, test=testinput)['out_mae_modelspec'],

        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s %s %s %s' % (executable, ticker, cusip, hpset, n),
    }
    result = copy.copy(out_importance_d)
    result.update(other_result)
    return result


def traceinfo(issuer, executable='traceinfo', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = path.working()
    dir_out_base = os.path.join(
        dir_working,
        executable,
        '%s' % issuer,
    )
    dir_out = (
        dir_out_base + '-test' if test else
        dir_out_base
    )

    result = {
        'in_trace': path.input(issuer, 'trace'),

        'out_log': os.path.join(dir_out, '0log.txt'),
        'out_by_issuer_cusip': os.path.join(dir_out, 'by_issuer_cusip.pickle'),
        'out_by_trace_index': os.path.join(dir_out, 'by_trace_index.pickle'),
        'out_by_trade_date': os.path.join(dir_out, 'by_trade_date.pickle'),
        'out_summary': os.path.join(dir_out, 'summary.pickle'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s' % (executable, issuer),
    }
    return result


def traceinfo_get(issuer, trace_index, executable='traceinfo_get', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = path.working()
    dir_out_base = os.path.join(
        dir_working,
        executable,
        issuer,
        trace_index,
    )
    dir_out = (
        dir_out_base + '-test' if test else
        dir_out_base
    )

    result = {
        'in_by_trace_index': os.path.join(dir_working, 'traceinfo', issuer, 'by_trace_index.pickle'),

        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s %s' % (executable, issuer, trace_index)
    }
    return result


if __name__ == '__main__':
    unittest.main()
    if False:
        # avoid pyflakes warnings
        pdb
