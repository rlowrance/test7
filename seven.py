'hold all the 7chord proprietary code'

import collections
import datetime
import numpy as np
import pandas as pd
import pdb
import unittest

import ColumnsTable
import Report


def classify_dealer_trades(orders):
    '''return (new DataFrame, remaining_orders) that classifies the trade_type D trades in the input dataframe
    Rules:
    1S. A D trade is a buy if the nearest S trade has the same quantity and occured within SEC seconds.
    1B. A D trade is a sell if the nearest B trade has the same quantity and occured within SEC seconds.
    When 1S or 1B, eliminate the S or B trade from further consideration of the rules.

    result is a DataFrame columns trade_type, rule_number and index a subset of values in orders.index
    '''
    def rule_1(orders, max_elapsed_seconds=datetime.timedelta(0, 15 * 60)):
        'return None or (d_index, d_trade_type, other_index)'
        dealer_orders = orders[orders.trade_type == 'D']
        if len(dealer_orders) == 0:
            return None  # no dealer orders
        dealer_order = dealer_orders.iloc[0]  # process only the first dealer order
        mask1 = np.logical_or(orders.trade_type == 'B', orders.trade_type == 'S')
        mask2 = orders.quantity == dealer_order.quantity
        mask3 = (orders.effectivedatetime - dealer_order.effectivedatetime).abs() < max_elapsed_seconds
        mask = mask1 & mask2 & mask3
        matching_orders = orders[mask]
        if len(matching_orders) == 0:
            return None  # no orders match the first dealer order
        time_deltas = (matching_orders.effectivedatetime - dealer_order.effectivedatetime).abs()
        sorted_offsets = time_deltas.argsort()
        nearest_index = time_deltas.index[sorted_offsets[0]]
        matching_buy_sell = orders.loc[nearest_index]
        replacement_trade_type = 'S' if matching_buy_sell.trade_type == 'B' else 'B'
        assert replacement_trade_type != matching_buy_sell.trade_type
        return (
            dealer_order.name,
            replacement_trade_type,
            nearest_index,
            )

    def apply_one_rule(orders):
        'apply one rule and return None or one (d_index, d_trade_type, other_index):'
        for rule_index, rule in enumerate(rule_1):
            result = rule(orders)
            if result is None:
                continue
            else:
                return rule_index, result
        return None

    all_replacements = None
    orders_subset = orders.copy()
    while True:
        # mutate orders_subset and all_replacements if a rule applies
        # otherwise, give up, because all possible rules have been applied
        pdb.set_trace()
        if len(orders_subset) == 0:
            break
        rule_index, rule_result = apply_one_rule(orders_subset)
        if rule_result is None:
            break
        d_index, d_trade_type, other_index = rule_result
        new_replacement = pd.DataFrame(
            data={
                'trade_type': d_trade_type,
                'rule_number': rule_index + 1,
            },  # reclassified trade_type for a 'D' trade
            index=[d_index],
            )
        all_replacements = (
            new_replacement if all_replacements is None else
            all_replacements.append(
                new_replacement,
                verify_integrity=True,
                )
            )
        orders_subset = orders_subset.drop([d_index, other_index])

    return all_replacements, orders_subset


class ClassifyDealerTradeTest(unittest.TestCase):
    def setUp(self):
        def dt(seconds):
            return datetime.datetime(2000, 1, 1, 0, 0, seconds)
        
        orders = (  # orderid, effectivetime, quantity, trade_type, spread)
            ('a-1', dt(1), 100, 'S', 10),
            ('b-2', dt(3), 100, 'D', 20),
            ('c-3', dt(4), 100, 'B', 30),
            ('d-4', dt(5), 100, 'D', 40),
            )
        df = None
        for order in orders:
            new_df = pd.DataFrame(
                index = [order[0]],
                data = {
                    'effectivedatetime': order[1],
                    'quantity': order[2],
                    'trade_type': order[3],
                    'oasspread': order[4],
                },
                )
            df = new_df if df is None else df.append(new_df, verify_integrity=True)
        self.orders = df
        
    def test_rule_1(self):
        dealer_trades = classify_dealer_trades(self.orders)
        print dealer_trades
        expecteds = (
            ('b-2', 'S'),
            ('d-4', 'B'),
            )
        pdb.set_trace()
        self.assertTrue(len(dealer_trades) == 2)
        for i, expected in enumerate(expecteds):
            expected_index, expected_trade_type = expected
            dealer_trade = dealer_trades.iloc[i]
            self.assertEqual(expected_index, dealer_trade.name)
            self.assertEqual(expected_trade_type, dealer_trade.trade_type)
            self.assertEqual(1, dealer_trade.rule_number)


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


def detect_outliers(df):
    'return vector of Bool, with True iff the trade is an outlier'
    pass


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

    def make_python_datetime(date_s, time_s):
        year, month, day = date_s.split('-')
        hour, minute, second = time_s.split(':')
        return datetime.datetime(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second),
            )

    trace = False
    print orders.columns
    if len(orders) != len(set(orders.issuepriceid)):
        print 'issuepriceidid is not a unique key'
        print len(orders), len(set(orders.issuepriceid))
    orderid_list = map(
        lambda x, y: '%08d-%05d' % (x, y),
        orders.issuepriceid,
        orders.sequencenumber,
        )
    if len(orderid_list) != len(set(orderid_list)):
        print 'issuepriceid + sequencenumber is not a unique key'
        print len(orderid_list), len(set(orderid_list))
    else:
        print 'orderid is a unique key'
    transformed = pd.DataFrame(
        data={
            # 'id': pd.Series(id_list, index=orders.index),
            'orderid': orderid_list,
            'effectivedatetime': map(make_python_datetime, orders.effectivedate, orders.effectivetime),
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
    print 'read %d records, retained %d' % (len(orders), len(result))
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


if __name__ == '__main__':
    run_regression_tests = False
    if run_regression_tests:
        # test on known data
        classify_dealer_trade_regression_test()
    unittest.main()
