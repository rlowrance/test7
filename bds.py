'''depict buy, dealer, and sell trades by maturity data

INVOCATION: python bds.py 

INPUT FILES: each *.csv file in NYU/7chord_ticker_universe_nyu_poc

OUTPUT FILES:
 log.txt  whatever is printed when this program last ran
 TICKER-MATURITY.txt  report for ticker and bond maturity date

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

import Bunch
import ColumnsTable
import dirutility
import Logger
import Report
import Timer


def make_control(argv):
    print 'argv', argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])  # ignore invocation name
    arg.me = 'bds'

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    # put all output in directory
    path_out_dir = dirutility.assure_exists('../data/working/' + arg.me + '/')

    return Bunch.Bunch(
        arg=arg,
        path_in_dir='../data/input/7chord_team_folder/NYU/7chord_ticker_universe_nyu_poc/',
        path_in_glob='*.csv',
        path_out_dir=path_out_dir,
        path_out_log=path_out_dir + '0log.txt',
        path_out_report_template=path_out_dir + '%s-%s.txt',
        random_seed=random_seed,
        test=arg.test,
        timer=Timer.Timer(),
        )


def unittests():
    return


def make_python_dates(elements):
    'return new series with corresponding datetime.date values'
    result = pd.Series(
        data=[datetime.date(1, 1, 1)] * len(elements),
        )
    for i, s in enumerate(elements):
        s_split = s.split('-')
        result[i] = datetime.date(
            int(s_split[0]),
            int(s_split[1]),
            int(s_split[2]),
            )
    return result


def make_python_times(elements):
    'return new series with datetime.time values'
    result = pd.Series(
        data=[datetime.time(0, 0, 0, 0)] * len(elements),
        )
    for i, s in enumerate(elements):
        s_split = s.split(':')
        result[i] = datetime.time(
            int(s_split[0]),
            int(s_split[1]),
            int(s_split[2]),
            )
    return result


def read_transform_subset(path, nrows):
    'return just the columns we use and stored as useful types'
    trace = False
    df = pd.read_csv(
        path,
        low_memory=False,
        index_col=0,
        nrows=nrows,
        )
    transformed = pd.DataFrame(
        data={
            'effectivedate': make_python_dates(df.effectivedate),
            'effectivetime': make_python_times(df.effectivetime),
            'maturity': make_python_dates(df.maturity),
            'tradetype': df.trade_type,
            'quantity': df.quantity,
            'oasspread': df.oasspread,
        },
        index=df.index,
        )
    assert len(transformed) == len(df)
    # eliminate rows with maturity = NaN
    result = transformed.dropna(
        how='any',  # drop rows with any NA values
        )
    if trace:
        print result.head()
        print len(df), len(transformed), len(result)
        pdb.set_trace()
    return result


class BDSReport(object):
    def __init__(self, ticker, maturity):
        self.ct = ColumnsTable.ColumnsTable([
            ('effectivedate', 10, '%10s', ('effective', 'date'), 'effectivedate'),
            ('effectivetime', 10, '%10s', ('effective', 'time'), 'effectivetime'),
            ('quantity', 8, '%8d', (' ', 'quantity'), 'quantity'),
            ('oasspread_buy', 8, '%8.1f', ('oasspread', 'buy'), 'oasspread if trade_type is B'),
            ('oasspread_dealer', 8, '%8.1f', ('oasspread', 'dealer'), 'oasspread if trade_type is D'),
            ('oasspread_sell', 8, '%8.1f', ('oasspread', 'sell'), 'oasspread if trade_type is S'),
            ])
        self.report = Report.Report()
        self.report.append('Buy-Dealer-Sell Analysis for Ticker %s Maturity %s' % (ticker, maturity))
        self.report.append(' ')

    def add_detail(self, d):
        tradetype = d['tradetype']
        if tradetype == 'B':
            self.ct.append_detail(
                effectivedate=d['effectivedate'],
                effectivetime=d['effectivetime'],
                quantity=d['quantity'],
                oasspread_buy=d['oasspread'],
                )
        elif tradetype == 'D':
            self.ct.append_detail(
                effectivedate=d['effectivedate'],
                effectivetime=d['effectivetime'],
                quantity=d['quantity'],
                oasspread_dealer=d['oasspread'],
                )
        elif tradetype == 'S':
            self.ct.append_detail(
                effectivedate=d['effectivedate'],
                effectivetime=d['effectivetime'],
                quantity=d['quantity'],
                oasspread_sell=d['oasspread'],
                )
        else:
            print 'bad tradetype', tradetype
            pdb.set_trace()

    def write(self, path):
        self.ct.append_legend()
        for line in self.ct.iterlines():
            self.report.append(line)
        self.report.write(path)


def do_work(control):
    'process each file in the input directory, creating a description CSV in the output directory'
    # we may need to explicitly set column types of provide a function to convert to more usable type
    # column_types = {
    #    'issuepriceid': np.int64,
    #    }
    path = control.path_in_dir + control.path_in_glob
    for path in glob.glob(control.path_in_dir + control.path_in_glob):
        filename = path.split('/')[-1]
        ticker = filename.split('.')[0]
        print 'reading file', filename
        df = read_transform_subset(
            path,
            10 if control.test else None,
            )
        if control.test:
            print df.head()
        print 'file %s: %d records' % (filename, len(df))
        maturities = sorted(set(df.maturity))
        for maturity in maturities:
            pdb.set_trace()
            r = BDSReport(ticker, maturity)
            subset = df[df.maturity == maturity]
            subset_sorted = subset.sort_values(
                by=['effectivedate', 'effectivetime'],
                )
            for i, series in subset_sorted.iterrows():
                r.add_detail(series)
            r.write(control.path_out_report_template % (ticker, maturity))


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

    unittests()
    main(sys.argv)
