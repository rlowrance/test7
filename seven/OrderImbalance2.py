from __future__ import division

import pdb
import unittest

from MaybeNumber import MaybeNumber
from Windowed import Windowed


class OrderImbalance(object):
    def __init__(self, lookback=None, typical_bid_offer=None):
        assert lookback > 0
        assert typical_bid_offer > 0

        self.lookback = lookback
        self.typical_bid_offer = typical_bid_offer

        self.prior_bid_price = None
        self.prior_offer_price = None
        self.bid_window = Windowed(lookback)
        self.offer_window = Windowed(lookback)

    def p(self):
        'print self'
        print 'lookback', self.lookback
        print 'typical_bid_offer', self.typical_bid_offer
        print 'prior_bid_price', self.prior_bid_price
        print 'prior_offer_price', self.prior_offer_price
        print 'bid_window', self.bid_window
        print 'offer_window', self.offer_window

    def open_interest(self, trade_type=None, trade_quantity=None, trade_price=None):
        'return a MaybeNumber, possibly containing the order imbalance, which may not exist'
        def closer(x, a, b):
            'return MaybeNumber(True) iff x is closer to a than b'
            # print 'closer', x, a, b
            return abs(MaybeNumber(x) - MaybeNumber(a)) < abs(MaybeNumber(x) - MaybeNumber(b))

        assert trade_type in ('B', 'D', 'S')
        assert trade_quantity > 0
        assert trade_price > 0

        # update prior prices, using the current trade
        # accumulate trades in the window
        # print 'open_interest', trade_type, trade_quantity, trade_price
        if trade_type == 'B':
            # dealer buy, hence a bid
            self.prior_bid_price = trade_price
            self.bid_window.append(trade_quantity)
            if self.prior_offer_price is None:
                self.prior_offer_price = self.prior_bid_price + self.typical_bid_offer
        elif trade_type == 'S':
            # dealer sell, hence an offer
            self.prior_offer_price = trade_price
            self.offer_window.append(trade_quantity)
            if self.prior_bid_price is None:
                self.prior_bid_price = self.prior_offer_price - self.typical_bid_offer
        else:
            # classify the trade is a dealer buy or sell based on
            # how close it is to prior buys and sells
            assert trade_type == 'D'
            # self.p()
            # pdb.set_trace()
            if closer(trade_price, self.prior_bid_price, self.prior_offer_price):
                self.prior_bid_price = trade_price
                self.bid_window.append(trade_quantity)
            elif closer(trade_price, self.prior_offer_price, self.prior_bid_price):
                self.priof_offer_price = trade_price
                self.offer_window.append(trade_quantity)
            else:
                # the trade is equal distance from the most recent buy and sell
                # classify it based on the synthetic mid price
                synthetic_mid_price = (self.prior_bid_price + self.prior_offer_price) / 2.0
                if trade_price < synthetic_mid_price:
                    self.prior_bid_price = trade_price
                    self.bid_window.append(trade_quantity)
                else:
                    self.prior_offer_price = trade_price
                    self.offer_window.append(trade_quantity)

        # classify dealer trade as buy or sell
        # based on how close it is to prior buy and sell prices and
        # its price relative to the synthetic mid price

        result = self.bid_window.sum() - self.offer_window.sum()  # a MaybeNumber
        return result


class TestOrderImbalance(unittest.TestCase):
    def common_tester(self, lookback, typical_bid_offer, tests, debug=False):
        oi = OrderImbalance(lookback=lookback, typical_bid_offer=typical_bid_offer)
        for test in tests:
            trade_type, trade_quantity, trade_price, expected_imbalance = test
            actual_imbalance = oi.open_interest(trade_type, trade_quantity, trade_price)
            if debug:
                oi.p()
                print expected_imbalance, actual_imbalance
                print test
                pdb.set_trace()
            self.assertEqual(MaybeNumber(expected_imbalance), actual_imbalance)

    def test_make_order_imbalance_start_with_B(self):
        debug = False
        lookback = 1
        typical_bid_offer = 2
        tests = (  # tests are applied sequentially to a single OrderImbalance instance
            ('B', 1, 101, None),
            ('B', 2, 102, None),
            ('S', 3, 100, -1),
            ('D', 4, 101, 1),  # get's classified as a B
        )
        self.common_tester(lookback, typical_bid_offer, tests, debug)

    def test_make_order_imbalance_start_with_D(self):
        debug = False
        lookback = 1
        typical_bid_offer = 2
        tests = (  # tests are applied sequentially to a single OrderImbalance instance
            ('D', 4, 101, None),  # get's classified as a B
        )
        self.common_tester(lookback, typical_bid_offer, tests, debug)


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
