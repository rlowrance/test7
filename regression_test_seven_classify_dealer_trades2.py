'''run regression tests on seven.classify_dealer_trades

INVOCATION: python regression_test_seven_classify_dealer_trades.py args

INPUT FILES:
 poc ms file (a portion is used and save to the cache)
 WORKING/ME/cache.csv   portion of input that is actually used

OUTPUT FILES:
 0log.txt               log file (what is printed also goes here)
 WORKING/ME/cache.csv   cached input
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
    parser.add_argument('--test', action='store_true', help='if present, truncated input and enable test code')
    parser.add_argument('--trace', action='store_true', help='if present, call pdb.set_trace() early in run')
    arg = parser.parse_args(argv[1:])  # ignore invocation name
    arg.me = 'regression_test_seven_classify_dealer_trades'

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    # put all output in directory
    ticker = 'ms'
    path_out_dir = dirutility.assure_exists('../data/working/' + arg.me + '/' + ticker + '/')

    return Bunch.Bunch(
        arg=arg,
        ticker=ticker,
        maturity='2012-04-01',
        path_cache=path_out_dir + 'cache.pickle',
        path_in=seven.path('poc', ticker),
        path_out_log=path_out_dir + '0log.txt',
        path_out_nareport_original=path_out_dir + 'nareport-original.txt',
        path_out_nareport_transformed=path_out_dir + 'nareport-transformed.txt',
        path_out_report_classify=path_out_dir + 'report-classify.txt',
        path_out_report_remaining=path_out_dir + 'report-remaining.txt',
        random_seed=random_seed,
        test=arg.test,
        timer=Timer.Timer(),
        )


def regression_test(orders, control):
    'test on one day of ms trades on the transformed orders'
    def test_ms_2012_01_03():
        expecteds = (  # all trades for ms on 2012-01-03
            # TODO: FIX, need orderid and new trade type and rule number
            # TODO: for now, check only rule 1
            ('62386808-06866', 'B'),
            ('62389116-09203', 'B'),
            ('62389128-09215', 'D'),
            ('62390680-10788', 'B'),
            ('62393088-13237', 'B'),
            ('62393415-13568', 'B'),
            ('62397128-17335', 'B'),
            ('62417290-37848', 'D'),
            ('62402791-23077', 'D'),
            ('62417197-37749', 'D'),
            ('62403810-24117', 'B'),
            ('62404592-24918', 'B'),
            ('62404499-24825', 'D'),
            ('62406416-26773', 'B'),
            ('62406368-26725', 'S'),
            ('62406599-26957', 'D'),
            ('62408563-28944', 'B'),
            ('62408447-28827', 'D'),
            ('62408502-28883', 'S'),
            ('62409040-29429', 'S'),
            )

        new_orders, remaining_orders = seven.classify_dealer_trades(orders)

        # write report showing how dealer orders were classified
        r_classify = seven.ReportClassifyDealerTrades('regression test; ticker: %s' % control.ticker)
        for i, new_order in new_orders.iterrows():
            r_classify.append_detail(
                new_order=new_order,
                is_remaining=i in remaining_orders.index)
        r_classify.write(control.path_out_report_classify)

        # write report showing remaining orders to be classified
        r_remaining = seven.ReportBuyDealerSell('remaining orders after Rule 1 applied; ticker: %s' % control.ticker)
        for i, remaining_order in remaining_orders.iterrows():
            r_remaining.append_detail(remaining_order)
        r_remaining.write(control.path_out_report_remaining)

        # check vs. expected
        for expected in expecteds:
            orderid, expected_restated_trade_type = expected
            actual_restated_trade_type = new_orders.loc[orderid].restated_trade_type
            print orderid, expected_restated_trade_type, actual_restated_trade_type
            if expected_restated_trade_type != actual_restated_trade_type:
                print 'oops'
                pdb.set_trace()
            assert expected_restated_trade_type == actual_restated_trade_type, (expected, actual_restated_trade_type)

    test_ms_2012_01_03()


def do_work(control):
    'read orders and hand them off to actual testing function'
    def read_raw_orders():
        orders_all = seven.read_orders_csv(
            path=control.path_in,
            nrows=1000 if control.test else None,
        )
        mask = (
            (orders_all.maturity == control.maturity) &
            (orders_all.effectivedate == '2012-01-03')
            )
        orders = orders_all[mask]
        orders_transformed, reports = seven.orders_transform_subset(control.ticker, orders)
        reports[0].write(control.path_out_nareport_original)
        reports[1].write(control.path_out_nareport_transformed)
        with open(control.path_cache, 'w') as f:
            pickle.dump((control, orders_transformed), f)
        return orders_transformed

    if os.path.isfile(control.path_cache):
        with open(control.path_cache, 'r') as f:
            control_prior, orders_transformed = pickle.load(f)
            if not(control_prior.ticker == control.ticker and control_prior.maturity == control.maturity):
                orders_transformed = read_raw_orders(control)
    else:
        orders_transformed = read_raw_orders()
    assert len(orders_transformed) == 39, len(orders_transformed)
    regression_test(orders_transformed, control)


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



