'''OrderImbalance{N} class definitions and unit tests

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''

from __future__ import division

import pdb
import unittest

from applied_data_science.Windowed import Windowed

from Synthetic import Synthetic


class OrderImbalance4(object):
    'from Gokul'
    def __init__(self, lookback=None, typical_bid_offer=None, proximity_cutoff=20):
        assert lookback > 0
        assert typical_bid_offer > 0
        assert 0 <= proximity_cutoff <= 100

        self.lookback = lookback
        self.typical_bid_offer = typical_bid_offer
        self.proximity_cutoff = proximity_cutoff

        self.synthetic = Synthetic(typical_bid_offer)
        self.trade_window = Windowed(lookback)
        self.running_imbalance = 0

        self.last_restated_trade_type = None

    def p(self):
        'print self'
        format = '%30s: %s'
        print format % ('lookback', self.lookback)
        print format % ('synthetic', self.synthetic)
        print format % ('trade_window', self.trade_window)
        print format % ('last_restated_trade_type', self.last_restated_trade_type)
        print format % ('running_imbalance', self.running_imbalance)

    def imbalance(self, trade_type=None, trade_quantity=None, trade_price=None, verbose=False):
        'return (order imbalance after the trade, restated trade type, err)'
        def near_mid(current_price):
            'is the current price near to the synthetic mid?'
            abs_diff = abs(current_price - self.synthetic.mid)
            if self.synthetic.mid < 0.1:
                return False  # protect against division by zero and near zero
            relative_abs_diff = abs_diff / self.synthetic.mid
            return (relative_abs_diff * 100.0) <= self.proximity_cutoff

        def treat_as(restated_trade_type):
            assert restated_trade_type in ('B', 'S')
            self.last_restated_trade_type = restated_trade_type 
            if restated_trade_type == 'B':
                self.synthetic.actual_bid(trade_price)
                self.trade_window.append(trade_quantity)
            elif restated_trade_type == 'S':
                self.synthetic.actual_offer(trade_price)
                self.trade_window.append(-trade_quantity)
            else:
                print 'internal error', restated_trade_type
                pdb.set_trace()

        assert trade_type in ('B', 'D', 'S')
        assert trade_quantity > 0
        assert trade_price > 0

        if verbose:
            self.p()
            print trade_type, trade_quantity, trade_price

        # update prior prices, using the current trade
        # accumulate trades in the window
        err = None
        if trade_type == 'B':
            treat_as('B')
        elif trade_type == 'S':
            treat_as('S')
        else:
            # classify the trade is a dealer buy or sell based on
            # how close it is to prior buys and sells
            if verbose:
                pdb.set_trace()
            assert trade_type == 'D'
            if self.synthetic.bid is None:
                err = 'no synthetic bid for D trade'
            elif trade_price <= self.synthetic.bid:
                treat_as('B')
            elif trade_price >= self.synthetic.offer:
                treat_as('S')
            elif near_mid(trade_price):
                if self.running_imbalance <= 0:
                    treat_as('B')
                else:
                    treat_as('S')
            else:
                if trade_price <= self.synthetic.mid:
                    treat_as('B')
                else:
                    treat_as('S')

        if err is None:
            self.running_balance = self.trade_window.sum()
            assert self.running_balance is not None
            assert self.last_restated_trade_type is not None
            return (self.running_imbalance, self.last_restated_trade_type, None)
        else:
            return (None, None, err)

        # self.running_imbalance = self.trade_window.sum()
        # assert self.running_imbalance is not None
        # if verbose:
        #     self.p()
        #     print self.running_imbalance
        #     pdb.set_trace()
        # if err is None:
        #     assert self.running_imbalance is not None
        #     assert self.last_restated_trade_type is not None
        # return (self.running_imbalance, self.last_restated_trade_type, err)


class TestOrderImbalance4(unittest.TestCase):
    def common_tester(self, lookback, typical_bid_offer, tests, verbose=False):
        oi = OrderImbalance4(lookback=lookback, typical_bid_offer=typical_bid_offer)
        for test in tests:
            trade_type, trade_quantity, trade_price, expected_imbalance = test
            actual_imbalance = oi.imbalance(trade_type, trade_quantity, trade_price, verbose)
            if verbose:
                oi.p()
                print expected_imbalance, actual_imbalance
                print test
                pdb.set_trace()
            self.assertEqual(expected_imbalance, actual_imbalance)

    def test_lookback_1(self):
        verbose = False
        lookback = 1
        typical_bid_offer = 2
        tests = (  # tests are applied sequentially to a single OrderImbalance instance
            ('B', 1, 101, 1),
            ('B', 2, 102, 2),
            ('S', 3, 103, -3),
            ('S', 4, 104, -4),
        )
        self.common_tester(lookback, typical_bid_offer, tests, verbose)

    def test_lookback_2(self):
        verbose = False
        lookback = 2
        typical_bid_offer = 2
        tests = (  # tests are applied sequentially to a single OrderImbalance instance
            ('B', 1, 101, 1),
            ('B', 2, 102, 3),
            ('S', 3, 103, -1),
            ('S', 4, 104, -7),
        )
        self.common_tester(lookback, typical_bid_offer, tests, verbose)

    def test_start_with_D(self):
        verbose = False
        lookback = 1
        typical_bid_offer = 2
        tests = (  # tests are applied sequentially to a single OrderImbalance instance
            ('D', 4, 101, 0),  # does not get classified, and running balance is zero
        )
        self.common_tester(lookback, typical_bid_offer, tests, verbose)

    def test_from_gokul(self):
        verbose = False
        oi = OrderImbalance4(
            lookback=100,  # the tests assume a number at least len(tests)
            typical_bid_offer=2,
            proximity_cutoff=20,
        )
        tests = (
            ('B', 1, 101, 'B', 1, 101, 102, 103),
            ('B', 2, 102, 'B', 3, 102, 103, 104),
            ('S', 3, 100, 'S', 0, 98, 99, 100),
            ('D', 4, 101, 'S', -4, 99, 100, 101),
            ('D', 5, 100, 'B', 1, 100, 101, 102),
            ('D', 6, 103, 'S', -5, 101, 102, 103),
            ('D', 7, 99, 'B', 2, 99, 100, 101),
            ('D', 8, 100.5, 'S', -6, 98.5, 99.5, 100.5),
            ('D', 9, 99.75, 'B', 3, 99.75, 100.75, 101.75),
        )
        for test in tests:
            (
                trade_type,
                trade_quantity,
                trade_price,
                expected_restated_trade_type,
                expected_imbalance,
                expected_synthetic_bid,
                expected_synthetic_mid,
                expected_synthetic_offer,
            ) = test
            actual_imbalance = oi.imbalance(trade_type, trade_quantity, trade_price, verbose)
            self.assertEqual(expected_restated_trade_type, oi.last_restated_trade_type)
            self.assertEqual(expected_imbalance, actual_imbalance)
            self.assertAlmostEqual(expected_synthetic_bid, oi.synthetic.bid)
            self.assertAlmostEqual(expected_synthetic_mid, oi.synthetic.mid)
            self.assertAlmostEqual(expected_synthetic_offer, oi.synthetic.offer)


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
