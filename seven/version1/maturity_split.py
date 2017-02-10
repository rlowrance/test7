'''split input order files by maturity date

INVOCATION: python maturity_split.py TICKER [---test] [--trace] 
where
 TICKER means to process input file TICKER.csv

INPUT FILES: TICKER.csv file in NYU/7chord_ticker_universe_nyu_poc

OUTPUT FILES: all in directory ../working/bds/TICKER/
 0counts.txt      report showing transaction counts by trade_type for each TICKER-MATURITY
 0log.txt         whatever is printed when this program last ran
 0na.txt          report on NA values from input file
 working/maturity_split/TICKER/MATURITY.csv

Written in Python 2.7
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
    parser.add_argument('ticker')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])  # ignore invocation name
    arg.me = 'maturity_split'

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    # put all output in directory
    path_out_dir = dirutility.assure_exists('../data/working/' + arg.me + '/' + arg.ticker + '/')

    return Bunch.Bunch(
        arg=arg,
        path_in_csv='../data/input/7chord_team_folder/NYU/7chord_ticker_universe_nyu_poc/' + arg.ticker + '.csv',
        path_out_csv_template=path_out_dir + '%s.csv',
        path_out_log=path_out_dir + '0log.txt',
        random_seed=random_seed,
        test=arg.test,
        timer=Timer.Timer(),
        )


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
