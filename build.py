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
'''

from __future__ import division

import os
import pdb
import pprint
import sys

import seven.path
pp = pprint.pprint

representative_orcl_cusip = '68389XAS4'


def make_scons(paths):
    'return Dict with items sources, targets, command, based on the paths'
    def select(f):
        return [v for k, v in paths.items() if f(k)]

    result = {
        'commands': [paths['command']],
        'sources': select(lambda k: k.startswith('in') or k.startswith('executable') or k.startswith('dep')),
        'targets': select(lambda k: k.startswith('out')),
    }
    return result


def cusips(ticker, executable='cusips', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = seven.path.working()
    dir_out = os.path.join(dir_working, '%s-%s%s' % (
        executable,
        ticker,
        ('-test' if test else ''),
        )
    )

    result = {
        'in_trace': seven.path.input(ticker=ticker, logical_name='trace'),

        'out_cusips': os.path.join(dir_out, '%s.pickle' % ticker),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s' % (executable, ticker),
    }
    return result


def features(ticker, executable='features', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = seven.path.working()
    dir_out = os.path.join(dir_working, '%s-%s%s' % (
        executable,
        ticker,
        ('-test' if test else ''),
        )
    )
    assert ticker == 'orcl'
    representative_cusip = '68389XAC9'

    result = {
        'in_etf_agg': seven.path.input(ticker, 'etf agg'),
        'in_etf_lqa': seven.path.input(ticker, 'etf lqd'),
        'in_fund': seven.path.input(ticker, 'fund'),
        'in_security_master': seven.path.input(ticker, 'security master'),
        'in_ohlc_equity_spx': seven.path.input(ticker, 'ohlc spx'),
        'in_ohlc_equity_ticker': seven.path.input(ticker, 'ohlc ticker'),
        'in_trace': seven.path.input(ticker, 'trace'),

        'out_cusips': os.path.join(dir_out, '%s.csv' % representative_cusip),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'executable_imported': os.path.join('seven', 'feature_makers.py'),
        'dir_out': dir_out,
        'command': 'python %s.py %s' % (executable, ticker),
    }
    return result


def fit_predict(ticker, cusip, hpset, effective_date, executable='fit_predict', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = seven.path.working()
    dir_out = os.path.join(dir_working, '%s-%s-%s-%s-%s%s' % (
        executable,
        ticker,
        cusip,
        hpset,
        effective_date,
        ('-test' if test else ''),
        )
    )

    result = {
        'in_features': os.path.join(dir_working, 'features-%s' % ticker, '%s.csv' % cusip),
        'in_targets': os.path.join(dir_working, 'targets-%s' % ticker, '%s.csv' % cusip),
        # in_dependency not used by fit_predict.py
        # it's used to force the running of targets.py before fit_predict.py is run
        'in_dependency': os.path.join(dir_working, 'targets-%s' % ticker, '%s.csv' % representative_orcl_cusip),

        'out_importances': os.path.join(dir_out, 'importances.csv'),
        'out_predictions': os.path.join(dir_out, 'predictions.csv'),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s %s %s %s' % (executable, ticker, cusip, hpset, effective_date),
    }
    return result


def report_compare_models2(ticker, cusip, hpset, executable='report_compare_models2', test=False):
    dir_working = seven.path.working()
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


def report_compare_models3(ticker, cusip, hpset, executable='report_compare_models3', test=False):
    dir_working = seven.path.working()
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

        'out_accuracy_modelspec': os.path.join(dir_out, 'accuracy_modelspec.txt'),
        'out_accuracy_targetfeature_modelspec': os.path.join(dir_out, 'accuracy_targetfeature_modelspec.txt'),
        'out_details': os.path.join(dir_out, 'details.txt'),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s %s %s' % (executable, ticker, cusip, hpset)
    }
    return result


def targets(ticker, executable='targets', test=False):
    'return dict with keys in_* and out_* and executable and dir_out'
    dir_working = seven.path.working()
    dir_out = os.path.join(dir_working, '%s-%s%s' % (
        executable,
        ticker,
        ('-test' if test else ''),
        )
    )
    assert ticker == 'orcl'

    result = {
        'in_trace': seven.path.input(ticker, 'trace'),

        'out_targets': os.path.join(dir_out, '%s.csv' % representative_orcl_cusip),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'executable': '%s.py' % executable,
        'dir_out': dir_out,
        'command': 'python %s.py %s' % (executable, ticker),
    }
    return result


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pass
    main(sys.argv)
