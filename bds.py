'''depict buy, dealer, and sell trades by maturity data

INVOCATION: python bds.py [---test] [--trace] --ticker TICKER
where
 TICKER means to process input file TICKER.csv

INPUT FILES: each *.csv file in NYU/7chord_ticker_universe_nyu_poc

OUTPUT FILES:
 log.txt               whatever is printed when this program last ran
 TICKER-MATURITY.txt  report for ticker and bond maturity date
 TICKER-counts.txt    report showing transaction counts by trade_type for each TICKER-MATURITY

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
    parser.add_argument('--ticker', action='store', required=True)
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
        path_out_log=path_out_dir + arg.ticker + '-0log.txt',
        path_out_report_ticker_maturity_template=path_out_dir + '%s-%s.txt',
        path_out_report_counts_template=path_out_dir + '%s-counts.txt',
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
            'trade_type': df.trade_type,
            'quantity': df.quantity,
            'oasspread': df.oasspread,
            'cusip': df.cusip,
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
    print 'file %s: read %d records, retained %d' % (path, len(df), len(result))
    return result


class CountReport(object):
    def __init__(self, verbose=False):
        self.ct = ColumnsTable.ColumnsTable([
            ('ticker', 6, '%6s', 'ticker', 'ticker'),
            ('maturity', 10, '%10s', 'maturity', 'maturity'),
            ('n_prints', 10, '%10d', 'nprints', 'number of prints (transactions)'),
            ('n_buy', 10, '%10d', 'n_buy', 'number of buy transactions'),
            ('n_dealer', 10, '%10d', 'n_dealer', 'number of dealer transactions'),
            ('n_sell', 10, '%10d', 'n_sell', 'number of sell transactions'),
            ('q_buy', 10, '%10d', 'q_buy', 'total quantity of buy transactions'),
            ('q_dealer', 10, '%10d', 'q_dealer', 'total quantity of dealer transactions'),
            ('q_sell', 10, '%10d', 'q_sell', 'total quantity of sell transactions'),
            ])
        self.report = Report.Report(
            also_print=verbose,
        )
        self.report.append('Buy-Dealer-Sell Analysis: Counts by trade_type')
        self.report.append(' ')

        self.n_prints = collections.defaultdict(int)
        self.n_buy = collections.defaultdict(int)
        self.n_dealer = collections.defaultdict(int)
        self.n_sell = collections.defaultdict(int)
        self.q_buy = collections.defaultdict(int)
        self.q_dealer = collections.defaultdict(int)
        self.q_sell = collections.defaultdict(int)

    def add_detail(self, ticker=None, maturity=None, d=None):
        'mutate self.counts; later produce actual detail lines'
        key = (ticker, maturity)
        self.n_prints[key] += 1
        trade_type = d['trade_type']
        if trade_type == 'B':
            self.n_buy[key] += 1
            self.q_buy[key] += d.quantity
        elif trade_type == 'D':
            self.n_dealer[key] += 1
            self.q_dealer[key] += d.quantity
        elif trade_type == 'S':
            self.n_sell[key] += 1
            self.q_sell[key] += d.quantity
        else:
            print 'bad trade_type', trade_type
            pdb.set_trace()

    def write(self, path):
        self._append_actual_detail_lines()
        self.ct.append_legend()
        for line in self.ct.iterlines():
            self.report.append(line)
        self.report.write(path)

    def _append_actual_detail_lines(self):
        'mutate self.ct'
        keys = sorted(self.n_prints.keys())
        for key in keys:
            ticker, maturity = key
            self.ct.append_detail(
                ticker=ticker,
                maturity=maturity,
                n_prints=self.n_prints[key],
                n_buy=self.n_buy[key],
                n_dealer=self.n_dealer[key],
                n_sell=self.n_sell[key],
                q_buy=self.q_buy[key],
                q_dealer=self.q_dealer[key],
                q_sell=self.q_sell[key],
                )


class TickerMaturityReport(object):
    def __init__(self, ticker, maturity, verbose=False):
        self.ct = ColumnsTable.ColumnsTable([
            ('cusip', 9, '%9s', ('', 'cusip'), 'cusip'),
            ('effectivedate', 10, '%10s', ('effective', 'date'), 'effectivedate'),
            ('effectivetime', 10, '%10s', ('effective', 'time'), 'effectivetime'),
            ('quantity', 8, '%8d', (' ', 'quantity'), 'quantity'),
            ('oasspread_buy', 8, '%8.1f', ('oasspread', 'buy'), 'oasspread if trade_type is B'),
            ('oasspread_dealer', 8, '%8.1f', ('oasspread', 'dealer'), 'oasspread if trade_type is D'),
            ('oasspread_sell', 8, '%8.1f', ('oasspread', 'sell'), 'oasspread if trade_type is S'),
            ])
        self.report = Report.Report(
            also_print=verbose,
        )
        self.report.append('Buy-Dealer-Sell Analysis for Ticker %s Maturity %s' % (ticker, maturity))
        self.report.append(' ')

    def add_detail(self, d):
        trade_type = d['trade_type']
        if trade_type == 'B':
            self.ct.append_detail(
                cusip=d['cusip'],
                effectivedate=d['effectivedate'],
                effectivetime=d['effectivetime'],
                quantity=d['quantity'],
                oasspread_buy=d['oasspread'],
                )
        elif trade_type == 'D':
            self.ct.append_detail(
                cusip=d['cusip'],
                effectivedate=d['effectivedate'],
                effectivetime=d['effectivetime'],
                quantity=d['quantity'],
                oasspread_dealer=d['oasspread'],
                )
        elif trade_type == 'S':
            self.ct.append_detail(
                cusip=d['cusip'],
                effectivedate=d['effectivedate'],
                effectivetime=d['effectivetime'],
                quantity=d['quantity'],
                oasspread_sell=d['oasspread'],
                )
        else:
            print 'bad trade_type', trade_type
            pdb.set_trace()

    def write(self, path):
        self.ct.append_legend()
        for line in self.ct.iterlines():
            self.report.append(line)
        self.report.write(path)


def do_work(control):
    'process the ticker file in the input directory, creating a description CSV and count files in the output directory'
    report_counts = CountReport()
    path = control.path_in_dir + control.arg.ticker + '.csv'
    filename = path.split('/')[-1]
    ticker = filename.split('.')[0]
    print 'reading file', filename
    df = read_transform_subset(
        path,
        10 if control.test else None,
    )
    if control.test:
        print df.head()
        pdb.set_trace()

    maturities = sorted(set(df.maturity))
    for maturity in maturities:
        report_ticker_maturity = TickerMaturityReport(ticker, maturity)
        subset = df[df.maturity == maturity]
        subset_sorted = subset.sort_values(
            by=['effectivedate', 'effectivetime'],
        )
        for i, series in subset_sorted.iterrows():
            report_ticker_maturity.add_detail(series)
            report_counts.add_detail(
                ticker=ticker,
                maturity=maturity,
                d=series)
        report_ticker_maturity.write(control.path_out_report_ticker_maturity_template % (ticker, maturity))
    report_counts.write(control.path_out_report_counts_template % ticker)


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
