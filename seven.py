'hold all the 7chord proprietary code'

import collections
import datetime
import pandas as pd
import pdb
import unittest

import ColumnsTable
import Report


def classify_dealer_trade(df):
    'return new DataFrame that classifies the trade_type D trades in the input dataframe'
    return None


def detect_outliers(df):
    'return vector of Bool, with True iff the trade is an outlier'
    pass


def path(name):
    'convert name to path'
    if name == '7chord-input-dir':
        return '../data/input/7chord_team_folder/NYU/7chord_ticker_universe_nyu_poc/',
    if name == 'ms':
        return path('7chord-input-dir') + 'ms.csv',
    assert False, 'unexpected name: ' + name


def orders_transform_subset(ticker, orders):
    'return (df with relevant columns with nonNA and transformed values, report on NAs)'
    def make_python_date(s):
        'return datetime.date corresponding to string s'
        s_split = s.split('-')
        return datetime.date(
            int(s_split[0]),
            int(s_split[1]),
            int(s_split[2]),
        )

    def make_python_time(s):
        'return datetime.time corresponding to string s'
        s_split = s.split(':')
        return datetime.time(
            int(s_split[0]),
            int(s_split[1]),
            int(s_split[2]),
        )

    trace = False
    if len(orders) != len(set(orders.issuepriceid)):
        print 'issuepriceid is not a unique key'
        print len(orders), len(set(orders.issuepriceid))
    orderid_list = []
    for i, series in orders.iterrows():
        orderid_list.append('%08d-%05d' % (series.issuepriceid, series.sequencenumber))
    if len(orderid_list) != len(set(orderid_list)):
        print 'is_list is not a unique key'
        print len(orderid_list), len(set(orderid_list))
    else:
        print 'orderid_list is a unique key'
    transformed = pd.DataFrame(
        data={
            # 'id': pd.Series(id_list, index=orders.index),
            'orderid': orderid_list,
            'effectivedate': orders.effectivedate.map(make_python_date, na_action='ignore'),
            'effectivetime': orders.effectivetime.map(make_python_time, na_action='ignore'),
            'maturity': orders.maturity.map(make_python_date, na_action='ignore'),
            'trade_type': orders.trade_type,
            'quantity': orders.quantity,
            'oasspread': orders.oasspread,  # TODO: take absolute value
            'cusip': orders.cusip,
        },
        index=orders.index,
        )
    assert len(transformed) == len(orders)
    # count NaN volumes by column
    r = ReportNA(ticker)
    for column in orders.columns:
        if False:  # debugging output
            print column
            if column == 'orderid':
                pdb.set_trace()
            print orders[column]
            print orders[column].isnull()
            print orders[column].isnull().sum()
            print
        r.add_detail(
            column=column,
            n_nans=orders[column].isnull().sum(),
            )
    # eliminate rows with maturity = NaN
    result = transformed.dropna(
        axis='index',
        how='any',  # drop rows with any NA values
        )
    r.append(' ')
    n_dropped = len(transformed) - len(result)
    r.append('input file contained %d record' % len(orders))
    r.append('retained %d of these records' % len(result))
    r.append('dropped %d records, because at least one column was NaN' % n_dropped)
    if trace:
        print result.head()
        print len(orders), len(transformed), len(result)
        pdb.set_trace()
    print 'file %s: read %d records, retained %d' % (path, len(orders), len(result))
    return result, r


class ReportCount(object):
    def __init__(self, verbose=False):
        self.ct = ColumnsTable.ColumnsTable([
            ('orderid', 13, '%13s', 'orderid', 'issuepriceid + sequencenumber'),
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


class ReportNA(object):
    def __init__(self, ticker, verbose=True):
        self.ct = ColumnsTable.ColumnsTable([
            ('column', 22, '%22s', 'column', 'column in input csv file'),
            ('n_nans', 7, '%7d', 'n_NaNs', 'number of NaN (missing) values in column in input csv file'),
        ])
        self.report = Report.Report(
            also_print=verbose,
        )
        self.report.append('Missing Values in Input File For Ticker %s' % ticker)
        self.report.append(' ')
        self.appended = []

    def add_detail(self, column=None, n_nans=None):
        self.ct.append_detail(
            column=column,
            n_nans=n_nans,
        )

    def append(self, line):
        self.appended.append(line)

    def write(self, path):
        self.ct.append_legend()
        for line in self.ct.iterlines():
            self.report.append(line)
        for line in self.appended:
            self.report.append(line)
        self.report.write(path)


class ReportTickerMaturity(object):
    def __init__(self, ticker, maturity, verbose=False):
        self.ct = ColumnsTable.ColumnsTable([
            ('orderid', 14, '%14s', ('unique', 'orderid'), 'issuepriceid + sequencenumber'),
            ('cusip', 9, '%9s', ('', 'cusip'), 'cusip'),
            ('effectivedate', 10, '%10s', ('effective', 'date'), 'effectivedate'),
            ('effectivetime', 10, '%10s', ('effective', 'time'), 'effectivetime'),
            ('quantity', 8, '%8d', (' ', 'quantity'), 'quantity'),
            ('oasspread_buy', 6, '%6.0f', ('dealer', 'buy'), 'oasspread if trade_type is B'),
            ('oasspread_dealer', 6, '%6.0f', ('dealer', 'dealer'), 'oasspread if trade_type is D'),
            ('oasspread_sell', 6, '%6.0f', ('dealer', 'sell'), 'oasspread if trade_type is S'),
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
                orderid=d['orderid'],
                cusip=d['cusip'],
                effectivedate=d['effectivedate'],
                effectivetime=d['effectivetime'],
                quantity=d['quantity'],
                oasspread_buy=d['oasspread'],
                )
        elif trade_type == 'D':
            self.ct.append_detail(
                orderid=d['orderid'],
                cusip=d['cusip'],
                effectivedate=d['effectivedate'],
                effectivetime=d['effectivetime'],
                quantity=d['quantity'],
                oasspread_dealer=d['oasspread'],
                )
        elif trade_type == 'S':
            self.ct.append_detail(
                orderid=d['orderid'],
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


class TestClassify_dealer_trade(unittest.TestCase):
    def test_ms_2012_01_03(self):
        expecteds = (  # all trades for ms on 2012-01-03
            ('10:45:04', 135, 'db', 'match customer'),
            ('11:12:19', 135, 'db', 'match dealer'),
            ('11:12:24', 135, 'eliminate', 'dealer wash'),
            ('11:30:04', 120, 'db', 'match customer'),
            ('11:57:56', 126, 'db', 'match customer'),
            ('12:01:52', 143, 'db', 'match customer'),
            ('12:51:09', 120, 'db', 'match customer'),
            ('13:57:50', 123, 'eliminate', 'wash multiple'),
            ('14:02:44', 123, 'elin=minate', 'wash multiple'),
            ('14:02:44', 123, 'eliminate', 'wash multiple'),
            ('14:14:25', 120, 'db', 'match customer'),
            ('14:21:04', 62, 'eliminate'),
            ('14:21:09', 62, 'eliminate', 'match customer at same price'),
            ('14:41:38', 61, 'ds', 'closer to recent sell prices than buy prices'),  # REDO these two
            ('14:41:45', 61, 'ds', 'closer to recent sell prices'),
            ('14:44:51', 147, 'db', 'match customer trade'),
            ('15:03:12', 61, 'ds', 'closer to recent sell prices'),
            ('15:03:23', 61, 'wash', 'match customer'),
            ('15:03:37', 154, 'wash', 'match customer'),
            ('15:07:46', 138, 'wash', 'match customer'),
            )
        pdb.set_trace()
        debug = True
        ticker = 'ms'
        the_date = datetime.date(1, 3, 2012)
        all_days = read_transform_subset(
            path,
            ticker,
            None,  # no report, since we are testing
            100 if debug else None,
            )
        one_day = all_days[all_days.effectivedate == the_date]
        print 'read $d records for ticker %s, retained %d for %s' % (len(all_days), ticker, len(one_day), the_date)
        actuals = classify_dealer_trade(one_day)
        for expected in expecteds:
            effectivetime, oasspread, expected_trade_type, expected_reason = expected
            # TODO: test if actual has this value (need to know structure of actual)
            self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
