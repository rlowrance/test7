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
import unittest

# imports from seven/
import EventId
import EventInfo
import exception
import HpGrids
import path


pp = pprint.pprint
dir_working = path.working()


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


def accuracy(issuer, cusip, target, predict_date, debug=False, executable='accuracy', test=False):
    dir_working = path.working()
    dir_out = os.path.join(
        dir_working,
        executable,
        issuer,
        cusip,
        target,
        predict_date,
    )

    def query_dirs():
        'yield path to each query directory'
        predict_dir = os.path.join(
            path.working(),
            'predict',
            issuer,
            cusip,
            target,
        )
        predict_date_dt = as_datetime_date(predict_date)  # convert str to datetime.date
        for item in os.listdir(predict_dir):
            item_path = os.path.join(predict_dir, item)
            if os.path.isdir(item_path) and predict_date_dt == EventId.EventId.from_str(item).date():
                yield item_path

    # the input files are the prediction files for the date
    # each is in this file:
    # {working}/predict/{issuer}/{cusips}/{target}/{query_event}/{fitted_event}/predictions.{trade_type}.csv
    # where {trade_type} is the reclassified trade type discovered by running the program features_targets.py
    list_in_files = []
    for query_dir in query_dirs():
        query_dir_items = os.listdir(query_dir)
        assert len(query_dir_items) == 1
        fitted_dir = query_dir_items[0]
        fitted_dir_path = os.path.join(query_dir, fitted_dir)
        # we want the file predictions.{reclassified_trade_type}.csv
        for trade_type in ('B', 'S'):
            predictions_path = os.path.join(fitted_dir_path, 'predictions.%s.csv' % trade_type)
            if os.path.isfile(predictions_path):
                list_in_files.append(predictions_path)
                break

    command = (
        'python %s.py %s %s %s %s' % (executable, issuer, cusip, target, predict_date) +
        (' --test' if test else '') +
        (' --debug' if debug else ''))

    result = {
        'list_in_files': list_in_files,

        'out_weights': os.path.join(dir_out, 'weights.pickle'),  # Dict[model_spec, weight]
        'out_weights_csv': os.path.join(dir_out, 'weights.csv'),  # columns: model_spec, weight
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': command,
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


def features_targets(issuer, cusip, effective_date, executable='features_targets', test=False, debug=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = path.working()
    dir_out_base = os.path.join(
        dir_working,
        '%s' % executable,
        '%s' % issuer,
        '%s' % cusip,
    )
    dir_out = (
        dir_out_base + '-test' if test else
        dir_out_base
    )

    # NOTE: excludes all the files needed to buld the features
    # these are in MidPredictor/automatic_feeds and the actual files depend on the {ticker} and {cusip}
    # N + 1 output files are produced:
    #  N feature files, where N is the number of trace prints for the cusip on the effective date
    #  1 log file
    # We track only the log file
    command = (
        'python %s.py %s %s %s' % (executable, issuer, cusip, effective_date) +
        (' --test' if test else '') +
        (' --debug' if debug else ''))

    result = {
        'in_trace': path.input(issuer, 'trace'),  # the trace file in automatic_feeds

        # these are source code dependencies beyond the executable
        # we track the main ones here
        'in_feature_makers': os.path.join(path.src(), 'seven', 'feature_makers.py'),
        'in_target_maker': os.path.join(path.src(), 'seven', 'target_maker.py'),

        'out_log': os.path.join(dir_out, effective_date + '-log.txt'),
        'optional_out_cache': os.path.join(dir_out, '1cache.pickle'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': command,
    }
    return result


def fit(issuer, cusip, target, event_id, hpset, executable='fit', debug=False, test=False, verbose=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    assert isinstance(issuer, str)
    assert isinstance(cusip, str)
    assert isinstance(target, str)
    assert isinstance(event_id, str)
    assert isinstance(hpset, str)
    # For some reason, the following statement is invalid
    # So I make dir_working a global variable as a work around.
    # dir_working = path.working()
    dir_in = os.path.join(
        dir_working,
        'features_targets',
        issuer,
        cusip,
    )
    dir_out_base = os.path.join(
        dir_working,
        '%s' % executable,
        '%s' % issuer,
        '%s' % cusip,
        '%s' % target,
        '%s' % event_id,
    )
    dir_out = (
        dir_out_base + '-test' if test else
        dir_out_base
    )

    # determine number of historic trades that are needed
    # we need one trade for the maximum value of the hyperparameter n_trades_back
    # the trades that are needed are those with the same reclassified trade type
    # as the query trade
    grid = HpGrids.construct_HpGridN(hpset)
    max_n_trades_back = 0
    for model_spec in grid.iter_model_specs():
        if model_spec.n_trades_back is not None:  # the naive model does not have n_trades_back
            max_n_trades_back = max(max_n_trades_back, model_spec.n_trades_back)

    # determine all input files
    # the are the max_n_trades_back just earliest files in features_targets/{ticker}/{cusip}
    query_reclassified_trade_type = None
    for possible_reclassified_trade_type in ('B', 'S'):
        path = os.path.join(dir_in, '%s.%s.csv' % (event_id, possible_reclassified_trade_type))
        if os.path.isfile(path):
            query_reclassified_trade_type = possible_reclassified_trade_type
            in_query = path
            break
    if query_reclassified_trade_type is None:
        raise exception.BuildException(
            'build.fit %s %s %s %s: event_id is not a B or D reclassified trade' % (
                issuer,
                cusip,
                event_id,
                hpset,
            )
        )

    # determine all file names that are for the query's reclassified trade type
    possible_input_filenames = []
    query_datetime = EventId.EventId.from_str(event_id).datetime()
    for dirpath, dirnames, filenames in os.walk(dir_in):
        if dirpath != dir_in:
            break  # only search the top-most directory
        for filename in filenames:
            if verbose:
                print filename
            filename_suffix = filename.split('.')[-1]
            if filename_suffix == 'csv':
                # the file contains features
                filename_base, filename_reclassified_trade_type, filename_suffix = filename.split('.')
                if filename_reclassified_trade_type != query_reclassified_trade_type:
                    continue
                input_event_id = EventId.EventId.from_str(filename_base)
                if input_event_id.datetime() < query_datetime:
                    possible_input_filenames.append(filename)
            else:
                # assume the file is a log file
                pass

    # select the most recent feature sets
    if len(possible_input_filenames) < max_n_trades_back:
        raise exception.BuildException(
            'build.fit %s %s %s %s: needs %d previous feature sets, but the file system has only %d' % (
                issuer,
                cusip,
                event_id,
                hpset,
                max_n_trades_back,
                len(possible_input_filenames),
            )
        )
    input_filenames = sorted(sorted(possible_input_filenames, reverse=True)[:max_n_trades_back])
    input_paths = [
        os.path.join(dir_in, input_filename)
        for input_filename in input_filenames
    ]

    command = (
        'python %s.py %s %s %s %s %s' % (executable, issuer, cusip, target, event_id, hpset) +
        (' --test' if test else '') +
        (' --debug' if debug else ''))

    result = {
        'list_in_features': input_paths,
        'in_query': in_query,

        # Do not specify each output file name, because fit.py may not be able
        # to fit the model. When it cannot, it only creaates a log file, not
        # an empty pickle file (as creating that seems to be a problem: How 
        # does one create an empty pickle?)
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_in': dir_in,
        'dir_out': dir_out,
        'command': command,

        'max_n_trades_back': max_n_trades_back,
        'len_possible_input_filenames': len(possible_input_filenames),
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


def predict(issuer, cusip, target, prediction_event_id, fitted_event_id, debug=False, executable='predict', test=False):
    def make_exception_message(txt):
        return '%s\n%s' % (
            'build.predict %s %s %s %s %s:' % (
                issuer,
                cusip,
                target,
                prediction_event_id,
                fitted_event_id,
            ),
            txt,
        )
    dir_working = path.working()
    dir_out = os.path.join(
        dir_working,
        executable,
        issuer,
        cusip,
        target,
        prediction_event_id,
        fitted_event_id,
    )

    # determine prediction_event_id path
    event_info = EventInfo.EventInfo(issuer, cusip)

    prediction = EventId.EventId.from_str(prediction_event_id)
    prediction_path = event_info.path_to_event(prediction)
    prediction_trade_type = event_info.reclassified_trade_type(prediction)

    # determine paths to fitted models
    fitted = EventId.EventId.from_str(fitted_event_id)
    fitted_dir_path = event_info.path_to_fitted_dir(fitted, target)
    list_in_fitted_paths = []
    for dirpath, dirnames, filenames in os.walk(fitted_dir_path):
        for filename in filenames:
            if filename.endswith('.pickle'):
                base, trade_type, suffix = filename.split('.')
                if trade_type != prediction_trade_type:
                    msg = '%s not equal to %s' % (
                        'prediction event reclassified trade type (%s)' % prediction_trade_type,
                        'fitted event reclassified trade type (%s)' % trade_type,
                    )
                    raise exception.BuildException(msg)
                else:
                    list_in_fitted_paths.append(os.path.join(dirpath, filename))
        break  # examine only the first directory
    if len(list_in_fitted_paths) == 0:
        raise exception.BUildException('no fitted pickle files for event %s' % fitted_event_id)

    command = (
        'python %s.py %s %s %s %s %s' % (executable, issuer, cusip, target, prediction_event_id, fitted_event_id) +
        (' --test' if test else '') +
        (' --debug' if debug else ''))

    result = {
        'list_in_fitted': list_in_fitted_paths,
        'in_prediction_event': prediction_path,

        'out_predictions': os.path.join(dir_out, 'predictions.%s.csv' % prediction_trade_type),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': command,

        'reclassified_trade_type': prediction_trade_type,
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
