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

import datetime
import os
import pdb
import pprint
import unittest

# imports from seven/
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


def sort_trace_file(issuer, debug=False, executable='sort_trace_file', test=False):
    dir_working = path.working()
    dir_out_base = os.path.join(
        dir_working,
        executable,
        issuer,
    )
    dir_out = (
        dir_out_base + '-test' if test else
        dir_out_base
    )
    command = (
        'python %s.py %s' % (
            executable,
            issuer,
        ) +
        (' --test' if test else '') +
        (' --debug' if debug else ''))

    result = {
        'in_trace_file': path.input(issuer, 'trace'),

        'out_sorted_trace_file': os.path.join(dir_out, 'trace_%s.csv' % issuer),
        'out_log': os.path.join(dir_out, '0log.txt'),

        'command': command,
        'dir_out': dir_out,
    }
    return result


def test_train(issuer, cusip, target, hpset,
               start_events, start_predictions, stop_predictions, 
               debug=False, executable='test_train', test=False,
               ):
    dir_working = path.working()
    dir_out_base = os.path.join(
        dir_working,
        executable,
        issuer,
        cusip,
        target,
        hpset,
        str(start_events),
        str(start_predictions),
        str(stop_predictions),
    )
    dir_out = (
        dir_out_base + '-test' if test else
        dir_out_base
    )
    dir_automatic_feeds = os.path.join(
        path.midpredictor(),
        'automatic_feeds',
    )
    command = (
        'python %s.py %s %s %s %s %s %s %s' % (
            executable,
            issuer,
            cusip,
            target,
            hpset,
            start_events,
            start_predictions,
            stop_predictions,
        ) +
        (' --test' if test else '') +
        (' --debug' if debug else ''))

    def af(file_name_base):
        return os.path.join(dir_automatic_feeds, '%s.csv' % file_name_base)

    result = {
        'in_amt_outstanding_history': af('amt_outstanding_history'),
        'in_current_coupon': af('current_coupon'),
        'in_etf_weight_of_cusip_pct_agg': af('etf_weight_of_cusip_pct_agg'),
        'in_etf_weight_of_cusip_pct_lqd': af('etf_weight_of_cusip_pct_lqd'),
        'in_etf_weight_of_issuer_pct_agg': af('etf_weight_of_issuer_pct_agg'),
        'in_etf_weight_of_issuer_pct_lqd': af('etf_weight_of_issuer_pct_lqd'),
        'in_etf_weight_of_sector_pct_agg': af('etf_weight_of_sector_pct_agg'),
        'in_etf_weight_of_sector_pct_lqd': af('etf_weight_of_sector_pct_lqd'),
        'in_fun_expected_interest_coverage': af('fun_expected_interest_coverage_%s' % issuer),
        'in_fun_gross_leverage': af('fun_gross_leverage_%s' % issuer),
        'in_ltm_ebitda': af('fun_LTM_EBITDA_%s' % issuer),
        'in_fun_mkt_cap': af('fun_mkt_cap_%s' % issuer),
        'in_fun_mkt_gross_leverage': af('fun_mkt_gross_leverage_%s' % issuer),
        'in_fun_reported_interest_coverage': af('fun_reported_interest_coverage_%s' % issuer),
        'in_fun_total_assets': af('fun_total_assets_%s' % issuer),
        'in_fun_total_debt': af('fun_total_debt_%s' % issuer),
        'in_hist_equity_prices': af('hist_EQUITY_prices'),
        'in_liq_flow_on_the_run': af('liq_flow_on_the_run_%s' % issuer),
        'in_secmaster': af('secmaster'),
        'in_trace': af('trace_%s' % issuer),

        'out_actions': os.path.join(dir_out, 'actions.csv'),
        'out_experts': os.path.join(dir_out, 'experts.csv'),
        'out_importances': os.path.join(dir_out, 'importances.csv'),
        'out_log': os.path.join(dir_out, '0log.txt'),
        'out_signal': os.path.join(dir_out, 'signal.csv'),
        'out_trace': os.path.join(dir_out, 'trace.csv'),

        'command': command,
        'dir_out': dir_out,
    }
    return result


if __name__ == '__main__':
    unittest.main()
    if False:
        # avoid pyflakes warnings
        pdb
