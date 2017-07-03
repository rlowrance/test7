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
import GetBuildInfo
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


def buildinfo(issuer, executable='buildinfo', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = path.working()
    dir_out = os.path.join(
        dir_working,
        executable,
        '%s%s' % (issuer, ('-test' if test else '')),
    )

    # NOTE: excludes all the files needed to buld the features
    # these are in MidPredictor/automatic feeds and the actual files depend on the {ticker} and {cusip}
    # The dependency on map_cusip_ticker.csv is not reflected
    result = {
        'in_trace': path.input(issuer, 'trace'),

        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s' % (executable, issuer),
    }
    basenames = (
        'cusip_effectivedatetime_issuepriceids',
        'cusips',
        'effectivedate_issuepriceid',
        'issuepriceid_cusip',
        'issuepriceid_effectivedate',
    )
    for basename in basenames:
        result['out_' + basename] = path.input(issuer, 'buildinfo ' + basename)
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
    # these are in MidPredictor/automatic feeds and the actual files depend on the {ticker} and {cusip}
    # The dependency on map_cusip_ticker.csv is not reflected
    # issuer = GetSecurityMasterInfo.GetSecurityMasterInfo().issuer_for_cusip(cusip)
    result = {
        'in_trace': path.input(issuer, 'trace'),
        # 'in_otr': seven.path.input(issuer, 'otr'),
        # these are source code dependencies beyond the executable
        'in_feature_makers': os.path.join(path.src(), 'seven', 'feature_makers.py'),
        'in_target_maker': os.path.join(path.src(), 'seven', 'target_maker.py'),

        'out_features': os.path.join(dir_out, 'features.csv'),
        'out_targets': os.path.join(dir_out, 'targets.csv'),
        'out_trace_indices': os.path.join(dir_out, 'common_trace_indices.txt'),
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


def fit(issuer, cusip, trade_id, hpset, executable='fit', test=False):
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
    # these are in MidPredictor/automatic feeds and the actual files depend on the {ticker} and {cusip}
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
    gbi = GetBuildInfo.GetBuildInfo(issuer)
    current_date = gbi.get_effectivedate(int(trade_id))
    list_in_features = []
    list_in_targets = []
    trace_indices_to_read = set()
    while len(trace_indices_to_read) <= max_n_trades_back:
        features_targets_dir = os.path.join(
            dir_working,
            'features_targets',
            issuer,
            cusip,
            '%s' % current_date,
        )
        filepath = os.path.join(features_targets_dir, 'common_trace_indices.txt')
        if not os.path.isfile(filepath):
            pdb.set_trace()
            print 'build.fit: not enough features'
            print 'arguments', issuer, cusip, trade_id, hpset
            print 'hpset requires %d historic feature sets' % max_n_trades_back
            print 'found only %d feature sets' % len(trace_indices_to_read)
            print 'FIX: run features_targets.py on earlier dates, starting with %s' % current_date
            sys.exit(1)
        with open(filepath, 'r') as f:
            for index, trace_index in enumerate(f):
                trace_indices_to_read.add(int(trace_index[:-1]))  # drop final \n
        list_in_features.append(os.path.join(features_targets_dir, 'features.csv'))
        list_in_targets.append(os.path.join(features_targets_dir, 'targets.csv'))
        current_date -= datetime.timedelta(1)  # 1 day back

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
    # these are in MidPredictor/automatic feeds and the actual files depend on the {ticker} and {cusip}
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
    dir_working = path.working()
    dir_out = os.path.join(
        dir_working,
        'predict',
        prediction_trade_id,
        fitted_trade_id,
    )
    traceinfo_path = traceinfo(issuer)['out_by_trace_index']
    with open(traceinfo_path, 'rb') as f:
        traceinfos = pickle.load(f)
    info = traceinfos[int(prediction_trade_id)]
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
        'out_by_trace_index': os.path.join(dir_out, 'by_trace_index.pickle'),
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
