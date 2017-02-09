'hold all the 7chord proprietary code'

import collections
import datetime
import numpy as np
import pandas as pd
import pdb
import unittest

import ColumnsTable
import Report


def classify_dealer_trades(orders, verbose=True):
    '''return (restated_dealer_trades: dict[orderid] trade_type, remaining_orders: DataFrame) that classifies the trade_type D trades in the input dataframe
    Rules:
    1S. A D trade is a buy if the nearest S trade has the same quantity and occured within SEC seconds.
    1B. A D trade is a sell if the nearest B trade has the same quantity and occured within SEC seconds.
    When 1S or 1B, eliminate the S or B trade from further consideration of the rules.

    result is a DataFrame columns trade_type, rule_number and index a subset of values in orders.index
    '''
    def rule_1(
            dealer_print_index,
            orders,
            max_elapsed_seconds=datetime.timedelta(0, 60 * 60),  # 1 hour = 60 minutes
            large_quantity=5000000,
            verbose=True,
    ):
        'return None [if no match to the dealer trade] or (matched_dealer_trades: dict[orderid] trade_type, matched_bs_trades: set(other_indices))'
        def print_order(tag, order_id):
            order = orders.loc[order_id]
            print ' %20s %6s %3.0f %7d' % (tag, order.effectivetime, order.oasspread, order.quantity)
        if verbose:
            print 'start rule_1', 'len(order)', len(orders)
        dealer_order = orders.loc[dealer_print_index]
        if verbose:
            print_order('dealer order', dealer_print_index)
        if dealer_order.oasspread == 61.0 and False:
            pdb.set_trace()
        if dealer_order.quantity == large_quantity:
            return None  # dealt with in rule 2
        mask1 = (orders.trade_type == 'B') | (orders.trade_type == 'S')
        mask2 = orders.quantity == dealer_order.quantity
        mask3 = (orders.effectivedatetime - dealer_order.effectivedatetime).abs() < max_elapsed_seconds
        mask = mask1 & mask2 & mask3
        matching_orders = orders[mask]  # buy or sell orders with same quanity within the time window
        if verbose and False:
            print '%d matching orders' % len(matching_orders)
            for i, order in matching_orders.iterrows():
                print_order('matching %d' % i + 1, order.orderid)
        if len(matching_orders) == 0:
            return None  # no orders match the first dealer order
        time_deltas = (matching_orders.effectivedatetime - dealer_order.effectivedatetime).abs()
        sorted_offsets = time_deltas.argsort()
        nearest_index = time_deltas.index[sorted_offsets[0]]
        matching_buy_sell = orders.loc[nearest_index]
        if verbose:
            print_order('matching buy sell', matching_buy_sell.orderid)
        replacement_trade_type = 'S' if matching_buy_sell.trade_type == 'B' else 'B'
        assert replacement_trade_type != matching_buy_sell.trade_type
        return (
            {dealer_print_index: replacement_trade_type},
            {nearest_index},  # a set
            )

    orders_subset = orders.copy()
    restated_dealer_trades = {}
    for rule_index, rule in enumerate([rule_1]):  # for now, just one rule; later add others
        # TODO: append all_new_not_replaced to orders_subset
        if verbose:
            print 'starting rule', rule_index + 1
        for dealer_trade_index in orders_subset[orders_subset.trade_type == 'D'].index:
            if dealer_trade_index in restated_dealer_trades:
                # a previously-executed rule restated this trade
                continue
            rule_result = rule(dealer_trade_index, orders_subset)
            if verbose:
                print 'rule_result', rule_result
            if rule_result is None:
                continue  # keep the dealer print in the orders subset
            else:
                matched_dealer_trades, matched_bs_orderids = rule_result
                for matched_dealer_orderid, new_trade_type in matched_dealer_trades.iteritems():
                    orders_subset = orders_subset.drop([matched_dealer_orderid])
                for matched_bs_orderid in matched_bs_orderids:
                    orders_subset = orders_subset.drop([matched_bs_orderid])
                restated_dealer_trades.update(matched_dealer_trades)
        print 'end rule', rule_index + 1

    if verbose:
        print 'all restated dealer trades'
        for k, v in restated_dealer_trades.iteritems():
            print k, v
        print 'remaining orders'
        print orders_subset

    # create new order prints with dealer trades reclassified
    new_orders = orders.copy()
    new_orders = new_orders.assign(restated_trade_type=orders.trade_type)
    for order_id, restated_trade_type in restated_dealer_trades.iteritems():
        prior = new_orders.get_value(order_id, 'restated_trade_type')
        print order_id, prior,
        new_orders.set_value(order_id, 'restated_trade_type', restated_trade_type)
        retrieved = new_orders.get_value(order_id, 'restated_trade_type')
        print retrieved
        assert retrieved != prior


    return new_orders, orders_subset


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
    transformed_unsorted = pd.DataFrame(
        data={
            # 'id': pd.Series(id_list, index=orders.index),
            'orderid': orderid_list,
            'effectivedatetime': map(make_python_datetime, orders.effectivedate, orders.effectivetime),
            'effectivedate': (orders.effectivedate.map(make_python_date, na_action='ignore')).values,
            'effectivetime': (orders.effectivetime.map(make_python_time, na_action='ignore')).values,
            'maturity': (orders.maturity.map(make_python_date, na_action='ignore')).values,
            'trade_type': orders.trade_type.values,
            'quantity': orders.quantity.values,
            'oasspread': orders.oasspread.values,  # TODO: take absolute value
            'cusip': orders.cusip.values,
        },
        index=orderid_list,  # the .values selectors are needed above, because we change the index
        )
    if trace:
        print transformed_unsorted.head()
    transformed = transformed_unsorted.sort_values(
        by='effectivedatetime',
        ascending=True,
        )
    assert transformed.index.is_unique
    assert len(transformed) == len(orders)

    # count NaN volumes by column

    def make_reportna(df, msg):
        r = ReportNA(ticker, msg)
        for column in df.columns:
            r.add_detail(
                column=column,
                n_nans=df[column].isnull().sum(),
                )
        return r

    r_original = make_reportna(orders, 'original')
    r_transformed = make_reportna(transformed, 'transformed before rows with any NAs eliminated')

    # eliminate rows with maturity = NaN
    result = transformed.dropna(
        axis='index',
        how='any',  # drop rows with any NA values
        )
    if trace:
        print result.head()
        print len(orders), len(transformed), len(result)
        pdb.set_trace()
    print 'accepted %d records, retained %d' % (len(orders), len(result))
    return result, (r_original, r_transformed)


def path(*args):
    '''return path within file system

    examples:
     path('working')
     path('input')
     path('poc', 'ms')
    '''
    if len(args) == 1:
        request = args[0]
        data = '../data/'
        if request in ('input', 'working'):
            return data + request + '/'
        raise ValueError(args)
    if len(args) == 2:
        arg0 = args[0]
        if arg0 == 'poc':
            # use Dropbox 7chord team folder in the root of the Dropbox directory
            poc = '../../../../7chord Team Folder/NYU/7chord_ticker_universe_nyu_poc/'
            ticker = args[1]
            return poc + ticker + '.csv'
        raise ValueError(args)
    raise ValueError('too long: %s' % str(args))


def read_orders_csv(path=None, nrows=None):
    assert path is not None, 'path= must be supplied'
    return pd.read_csv(
        path,
        low_memory=False,
        index_col=0,
        nrows=nrows,
        )


ColumnSpec = collections.namedtuple(
    'ColumnSpec',
    'print_width formatter heading1 heading2 legend',
    )


all_column_specs = {  # each with a 2-row header
    'effectivedatetime': ColumnSpec(19, '%19s', 'effective', 'datetime', 'date and time of the print'),
    'orderid': ColumnSpec(14, '%14s', ' ', 'orderid', 'issuepriceid + sequencenumber'),
    'maturity': ColumnSpec(10, '%10s', ' ',  'maturity', 'maturity'),
    'n_prints': ColumnSpec(10, '%10d', ' ', 'nprints', 'number of prints (transactions)'),
    'n_buy': ColumnSpec(10, '%10d', 'number', 'buy', 'number of buy transactions'),
    'n_dealer': ColumnSpec(10, '%10d', 'number', 'dealer', 'number of dealer transactions'),
    'n_sell': ColumnSpec(10, '%10d', 'numbe', 'sell', 'number of sell transactions'),
    'oasspread_buy': ColumnSpec(9, '%9.0f', 'oasspread', 'buy', 'oasspread for dealer buy from customer'),
    'oasspread_dealer': ColumnSpec(9, '%9.0f', 'oasspread', 'dealer', 'oasspread for dealer trade with another dealer'),
    'oasspread_sell': ColumnSpec(9, '%9.0f', 'oasspread', 'sell', 'oasspread for dealer sell to customer'),
    'quantity': ColumnSpec(8, '%8d', ' ', 'quantity', 'number of bonds traded'),
    'q_buy': ColumnSpec(10, '%10d', 'quantity', 'buy', 'total quantity of buy transactions'),
    'q_dealer': ColumnSpec(10, '%10d', 'quantity', 'dealer', 'total quantity of dealer transactions'),
    'q_sell': ColumnSpec(10, '%10d', 'quantity', 'sell', 'total quantity of sell transactions'),
    'restated_trade_type': ColumnSpec(10, '%10s', 'restated', 'trade_type', 'trade_type of reclassified dealer-to-dealer trade'),
    'retained_order': ColumnSpec(7, '%7s', 'retained', 'order', 'whether order is subject to further rules'),
    'ticker': ColumnSpec(6, '%6s', ' ', 'ticker', 'ticker'),
    }


def column_def(column_name):
    assert column_name in all_column_specs, '%s not defined in all_column_specs' % column_name
    column_spec = all_column_specs[column_name]
    return [
        column_name,
        column_spec.print_width,
        column_spec.formatter,
        [column_spec.heading1, column_spec.heading2],
        column_spec.legend,
        ]


def column_defs(*column_names):
    return [
        column_def(column_name)
        for column_name in column_names
        ]


class ReportBuyDealerSell(object):
    def __init__(self, tag, verbose=True):
        self.ct = ColumnsTable.ColumnsTable(
            column_defs('orderid', 'effectivedatetime', 'quantity',
                        'oasspread_buy', 'oasspread_dealer', 'oasspread_sell',
                        )
            )
        self.report = Report.Report(
            also_print=verbose,
            )
        self.report.append(tag)
        self.report.append(' ')

    def write(self, path):
        self.ct.append_legend()
        for line in self.ct.iterlines():
            self.report.append(line)
        self.report.write(path)

    def append_detail(self, order):
        'mutate self.ct'
        self.ct.append_detail(
            orderid=order.orderid,
            effectivedatetime=order.effectivedatetime,
            quantity=order.quantity,
            oasspread_buy=order.oasspread if order.trade_type == 'B' else None,
            oasspread_dealer=order.oasspread if order.trade_type == 'D' else None,
            oasspread_sell=order.oasspread if order.trade_type == 'S' else None,
            )


class ReportClassifyDealerTrades(object):
    def __init__(self, tag, verbose=True):
        self.ct = ColumnsTable.ColumnsTable(
            column_defs('orderid', 'effectivedatetime', 'quantity',
                        'oasspread_buy', 'oasspread_dealer', 'oasspread_sell',
                        'restated_trade_type', 'retained_order')
            )
        self.report = Report.Report(
            also_print=verbose)
        self.report.append(tag)
        self.report.append(' ')

    def write(self, path):
        self.ct.append_legend()
        for line in self.ct.iterlines():
            self.report.append(line)
        self.report.write(path)

    def append_detail(self, new_order=None, is_remaining=None):
        'mutate self.ct'
        self.ct.append_detail(
            orderid=new_order.orderid,
            effectivedatetime=new_order.effectivedatetime,
            quantity=new_order.quantity,
            oasspread_buy=new_order.oasspread if new_order.trade_type == 'B' else None,
            oasspread_dealer=new_order.oasspread if new_order.trade_type == 'D' else None,
            oasspread_sell=new_order.oasspread if new_order.trade_type == 'S' else None,
            restated_trade_type=new_order.restated_trade_type if new_order.trade_type == 'D' and new_order.restated_trade_type is not None else None,
            retained_order=is_remaining,
            )


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
    def __init__(self, ticker, msg, verbose=True):
        self.ct = ColumnsTable.ColumnsTable([
            ('column', 22, '%22s', 'column', 'column in input csv file'),
            ('n_nans', 7, '%7d', 'n_NaNs', 'number of NaN (missing) values in column in input csv file'),
        ])
        self.report = Report.Report(
            also_print=verbose,
        )
        self.report.append('Missing Values in Input File %s For Ticker %s' % (msg, ticker))
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
