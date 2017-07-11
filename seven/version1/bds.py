'''depict buy, dealer, and sell trades by maturity data

INVOCATION: python bds.py TICKER [---test] [--trace] 
where
 TICKER means to process input file TICKER.csv

INPUT FILES: each *.csv file in NYU/7chord_ticker_universe_nyu_poc

OUTPUT FILES: all in directory ../working/bds/TICKER/
 0counts.txt      report showing transaction counts by trade_type for each TICKER-MATURITY
 0log.txt         whatever is printed when this program last ran
 0na.txt          report on NA values from input file
 MATURITY.csv     replica of TICKER.csv but with transactions only on the MATURITY date
 MATURITY.txt     report for ticker and bond maturity date

Written in Python 2.7

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
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
    print 'argv', argv
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])  # ignore invocation name
    arg.me = 'bds'

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    # put all output in directory
    path_out_dir = dirutility.assure_exists('../data/working/' + arg.me + '/' + arg.ticker + '/')

    return Bunch.Bunch(
        arg=arg,
        path_in_dir='../data/input/7chord_team_folder/NYU/7chord_ticker_universe_nyu_poc/',
        path_in_glob='*.csv',
        path_out_dir=path_out_dir,
        path_out_log=path_out_dir + '0log.txt',
        path_out_report_ticker_maturity_template=path_out_dir + '%s.txt',
        path_out_ticker_maturity_template_csv=path_out_dir + '%s.csv',
        path_out_report_counts=path_out_dir + '0counts.txt',
        path_out_report_na=path_out_dir + '0na.txt',
        random_seed=random_seed,
        test=arg.test,
        timer=Timer.Timer(),
        )


def create_reports_counts_maturities(orders, ticker):
    'return (dict[maturity]ReportCount, ReportCounts)'
    report_counts = seven.ReportCount()
    maturities = sorted(set(orders.maturity))
    d = {}
    for maturity in maturities:
        print 'creeating report for ticker', ticker, 'maturity', maturity
        report_ticker_maturity = seven.ReportTickerMaturity(ticker, maturity)
        subset = orders[orders.maturity == maturity]
        subset_sorted = subset.sort_values(
            by=['effectivedate', 'effectivetime'],
        )
        for i, series in subset_sorted.iterrows():
            report_ticker_maturity.add_detail(series)
            report_counts.add_detail(
                ticker=ticker,
                maturity=maturity,
                d=series)
        d[maturity] = report_ticker_maturity
    return (d, report_counts)


def do_work(control):
    'process the ticker file in the input directory, creating a description CSV and count files in the output directory'
    path = control.path_in_dir + control.arg.ticker + '.csv'
    filename = path.split('/')[-1]
    ticker = filename.split('.')[0]
    print 'reading file', filename
    orders, report_na = seven.orders_transform_subset(
        ticker,
        pd.read_csv(
            path,
            low_memory=False,
            index_col=0,
            nrows=1000 if control.test else None,
            ),
        )
    print 'head of orders'
    print orders.head()
    report_na.write(control.path_out_report_na)
    reports_maturity, report_counts = create_reports_counts_maturities(
        orders,
        ticker,
        )
    for maturity, report_maturity in reports_maturity.iteritems():
        report_maturity.write(control.path_out_report_ticker_maturity_template % maturity)
        print 'wrote maturity report for', maturity
    report_counts.write(control.path_out_report_counts)


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
