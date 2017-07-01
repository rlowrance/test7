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
import os
import pdb
import pprint
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


def fit(issuer, cusip, trade_id, hpset, executable='features_targets', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = path.working()
    dir_out = os.path.join(
        dir_working,
        '%s' % executable,
        '%s' % issuer,
        '%s' % cusip,
        '%s' % trade_id,
        '%s%s' % (hpset, ('-test' if test else '')),
    )

    # NOTE: excludes all the files needed to buld the features
    # these are in MidPredictor/automatic feeds and the actual files depend on the {ticker} and {cusip}
    # The dependency on map_cusip_ticker.csv is not reflected
    pdb.set_trace()

    # determine all output files
    grid = HpGrids.construct_HpGridN(hpset)
    list_out_fitted = []
    max_n_trades_back = 0
    for model_spec in grid.iter_model_specs():
        if model_spec.n_trades_back is not None:
            max_n_trades_back = max(max_n_trades_back, model_spec.n_trades_back)
        list_out_fitted.append(os.path.join(dir_out, trade_id, '%s.pickle' % model_spec))

    pdb.set_trace()
    # determine all input files
    # the input files are grouped by date
    print max_n_trades_back
    list_in_features_targets = {}
    needed = max_n_trades_back
    bi = GetBuildInfo.GetBuildInfo()
    current_date = get_effective_date
    while needed > 0:
        n_trades_present = make_n_trades_present()

    result = {
        'in_trace': path.input(issuer, 'trace'),

        'list_out_fitted': list_out_fitted,
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s %s %s %s' % (executable, issuer, cusip, trade_id, hpset),

        'max_n_trades_back': max_n_trades_back,
    }
    return result


class Test_fit(unittest.TestCase):
    def test(self):
        verbose = False
        Test = collections.namedtuple('Test', 'issuer cusip trade_id hpset')
        tests = (
            Test('AAPL', '037833AG5', '127076037', 'grid3'),
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


if __name__ == '__main__':
    unittest.main()
    if False:
        # avoid pyflakes warnings
        pdb
