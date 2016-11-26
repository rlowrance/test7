'''run regression tests on seven.classify_dealer_trades

INVOCATION: python regression_test_seven_classify_dealer_trades.py args

INPUT FILES: 
 poc ms file (a portion is used and save to the cache)

OUTPUT FILES:
 0log.txt
 WORKING/ME/report.txt  report showing classifications for TICKER and MATURITY
'''
from __future__ import division

import argparse
import collections
import cPickle as pickle
import datetime
import glob
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys
import unittest

import Bunch
import ColumnsTable
import dirutility
import Logger
import Report
import seven
import Timer


def make_control(argv):
    'return a Bunch of controls'
    print 'argv', argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', action='store_true', help='if present, reused parsed input file data.cache')
    parser.add_argument('--maturity', default='2014-04-01', help='maturity in input file to test')
    parser.add_argument('--test', action='store_true', help='if present, truncated input and enable test code')
    parser.add_argument('--ticker', default='ms', help='input poc file')
    parser.add_argument('--trace', action='store_true', help='if present, call pdb.set_trace() early in run')
    arg = parser.parse_args(argv[1:])  # ignore invocation name
    arg.me = 'regression_test_seven_classify_dealer_trades'

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    # put all output in directory
    path_out_dir = dirutility.assure_exists('../data/working/' + arg.me + '/' + arg.ticker + '/')

    return Bunch.Bunch(
        arg=arg,
        path_in=seven.path('poc', arg.ticker),
        path_out_log=path_out_dir + '0log.txt',
        path_out_report=path_out_dir + 'report.txt',
        random_seed=random_seed,
        test=arg.test,
        timer=Timer.Timer(),
        )

def classify_dealer_trade_regression_test():
    'test on one day of ms trades'
    def test_ms_2012_01_03():
        expecteds = (  # all trades for ms on 2012-01-03
            # TODO: FIX, need orderid and new trade type and rule number
            # TODO: for now, check only rule 1
            ('62386808-06866', 135, 'B', 1),  # manage inventory
            ('62389116-09203', 135, None, 2),  # wash
            ('62389128-09215', 135, None, 2),
            ('62390680-10788', 120, 'B', 1),
            ('62393088-13237', 120, 'B', 1),
            ('62393415-13568', 126, 'B', 1),
            ('62393415-13568', 143, 'B', 1),
            ('62397128-17335', 120, 'B', 1),
            ('62417290-37848', 123, None, 3),       
            ('62402791-23077', 123, None, 3),       
            ('62417197-37749', 123, None, 3),
            ('62403810-24117', 120, 'B', 1),       
            ('62404592-24918', 62, None, 4),  # need a rule for this one
            ('62404499-24825', 62, 'B', 1),
            ('62406416-26773', 61, 'S', 1),       
            ('62406368-26725', 61, None, 4),  # need a rule for this one
            ('62406599-26957', 147, 'B', 1),
            ('62408563-28944', 61, None, 4),       
            ('62408447-28827', 61, 'B', 1),
            ('62408502-28883', 154, 'S', 1),
            ('62409040-29429', 138, 'S', 1),
            )
        debug = False
        ticker = 'ms'
        maturity = '2012-04-01'
        pdb.set_trace()
        orders = pd.read_csv(
            '../data/working/bds/%s/%s.csv' % (ticker, maturity),
            low_memory=False,
            index_col=0,
            nrows=100 if debug else None,
        )
        orders_date = orders[orders.effectivedate == datetime.date(2012, 1, 3)]
        # fails because effectivedate has become a string
        # need to run through orders_transform_subset
        print len(orders_date)
        for i, order in orders.iterrows():
            print i
            print order
            print order.effectivedate
        pdb.set_trace()
        fixes, remaining_orders = classify_dealer_trades(orders_date)
        print fixes
        for expected in expecteds:
            expected_id, expected_spread, expected_trade_type, expected_rule_number = expected
            print expected_id, expected_spread, expected_trade_type, expected_rule_number
            msg = None
            pdb.set_trace()

    test_ms_2012_01_03()


def do_work(control):
    'process the ticker file in the input directory, creating a description CSV and count files in the output directory'
    path = control.path_in_csv
    print 'reading', path
    orders = pd.read_csv(
        path,
        low_memory=False,
        index_col=0,
        nrows=1000 if control.test else None,
    )
    cumulative_len = 0
    for maturity in sorted(set(orders.maturity)):
        orders_maturity = orders[orders.maturity == maturity]
        orders_maturity.to_csv(control.path_out_csv_template % maturity)
        print 'ticker %s maturity %s wrote %d records' % (control.arg.ticker, maturity, len(orders_maturity))
        cumulative_len += len(orders_maturity)
    assert cumulative_len == len(orders)


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger.Logger(logfile_path=control.path_out_log)  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.test:
        print 'DISCARD OUTPUT: test'
    print control
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pass

    # unittest.main()
    main(sys.argv)



