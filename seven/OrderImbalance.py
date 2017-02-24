'OrderImbalance{N} class definitions and unit tests'

from __future__ import division

from abc import ABCMeta, abstractmethod
import pdb
import unittest

from MaybeNumber import MaybeNumber
from Windowed import Windowed


class Synthetic(object):
    def __init__(self, initial_spread=None):
        assert initial_spread is not None
        self.spread = initial_spread

        self.bid = None
        self.offer = None
        self.mid = None

    def __str__(self):
        return 'Synthetic(%s, %s, %s)' % (self.bid, self.mid, self.offer)

    def actual_bid(self, actual):
        self.bid = actual
        self.offer = actual + self.spread
        self._update_mid()

    def actual_offer(self, actual):
        self.offer = actual
        self.bid = actual - self.spread
        self._update_mid()

    def _update_mid(self):
        self.mid = 0.5 * (self.bid + self.offer)

    def update_spread(self, new_spread):
        self.spread = new_spread


class OrderImbalance(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def imbalance(self, trade_type=None, trade_quantity=None, trade_price=None, verbose=False):
        'return the order imbalance, a number'
        pass


class OrderImbalance3(OrderImbalance):
    def __init__(self, typical_bid_offer=None):
        super(OrderImbalance3, self).__init__()
        assert typical_bid_offer > 0

        self.typical_bid_offer = MaybeNumber(typical_bid_offer)

        self.prior_bid_price = MaybeNumber(None)
        self.prior_offer_price = MaybeNumber(None)

        self.cumulatively_bought = 0
        self.cumulatively_sold = 0

    def p(self):
        'print self'
        format = '%30s: %s'
        print format % ('typical_bid_offer', self.typical_bid_offer)
        print format % ('prior_bid_price', self.prior_bid_price)
        print format % ('prior_offer_price', self.prior_offer_price)
        print format % ('cumulatively_bought', self.cumulatively_bought)
        print format % ('cumulatively_sold', self.cumulatively_sold)

    def imbalance(self, trade_type=None, trade_quantity=None, trade_price=None, verbose=False):
        'return a MaybeNumber, possibly containing the order imbalance, which may not exist'
        def make_bid_weight(dealer_price, bid_price, offer_price):
            'return weight to be given to bid as a MaybeNumber'
            def distance(a, b):
                diff = a - b
                return (diff * diff).sqrt()

            distance_to_bid = distance(dealer_price, bid_price)
            distance_to_offer = distance(dealer_price, offer_price)
            # print dealer_price, bid_price, offer_price, distance_to_bid, distance_to_offer
            return 1.0 - (distance_to_bid / (distance_to_bid + distance_to_offer))

        assert trade_type in ('B', 'D', 'S')
        assert trade_quantity > 0
        assert trade_price > 0

        if verbose:
            self.p()
            print trade_type, trade_quantity, trade_price

        # update prior prices, using the current trade
        # accumulate trades in the window
        price = MaybeNumber(trade_price)
        quantity = MaybeNumber(trade_quantity)
        if trade_type == 'B':
            # dealer buy, hence a bid
            self.prior_bid_price = MaybeNumber(price)
            self.cumulatively_bought += trade_quantity
            if self.prior_offer_price.value is None:
                self.prior_offer_price = self.prior_bid_price + self.typical_bid_offer
                assert self.prior_bid_price >= self.prior_offer_price  # because prices are OAS spreads
        elif trade_type == 'S':
            # dealer sell, hence an offer
            self.prior_offer_price = MaybeNumber(price)
            self.cumulatively_sold += quantity
            if self.prior_bid_price.value is None:
                self.prior_bid_price = self.prior_offer_price - self.typical_bid_offer
        else:
            # classify the trade is a dealer buy or sell based on
            # how close it is to prior buys and sells
            assert trade_type == 'D'
            if price.ge(self.prior_bid_price).value:
                weight_bid = MaybeNumber(1.0)
            elif price.le(self.prior_offer_price).value:
                weight_bid = MaybeNumber(0.0)
            else:
                weight_bid = make_bid_weight(price, self.prior_bid_price, self.prior_offer_price)
            if verbose:
                print 'weight_bid', weight_bid
            if weight_bid.value is not None:
                self.cumulatively_bought += weight_bid * quantity
                self.cumulatively_sold += (1.0 - weight_bid) * quantity
                # NOTE: do not update prior bid and offer prices
            else:
                # throw away the dealer trade_price because we cannot determine weight_bid
                # note that the cumulative_bought and sold will not change and that they
                # were initialized to zero, leading to an error when the first
                # trade is a D
                pass

        result = MaybeNumber(self.cumulatively_bought - self.cumulatively_sold)
        if verbose:
            self.p()
            print result
            pdb.set_trace()
        return result


class TestOrderImbalance3(unittest.TestCase):
    def common_tester(self, typical_bid_offer, tests, debug=False):
        oi = OrderImbalance3(typical_bid_offer=typical_bid_offer)
        for test in tests:
            trade_type, trade_quantity, trade_price, expected_imbalance = test
            actual_imbalance = oi.imbalance(trade_type, trade_quantity, trade_price)
            if debug:
                oi.p()
                print expected_imbalance, actual_imbalance
                print test
                pdb.set_trace()
            self.assertEqual(MaybeNumber(expected_imbalance), actual_imbalance)

    def test_make_order_imbalance_start_with_B(self):
        debug = False
        typical_bid_offer = 2
        tests = (  # tests are applied sequentially to a single OrderImbalance instance
            ('B', 1, 101, 1),
            ('B', 2, 102, 3),
            ('S', 3, 100, 0),
            ('D', 4, 101, 0),    # allocated .50-.50, since spread is (102, 100)
            ('D', 5, 100, -5),   # allocated entirely as a S
            ('D', 6, 103, 1),    # allocated entirely as B
            ('D', 7, 99, -6),    # allocated entire as S
            ('D', 8, 100.5, -10),  # allocat3ed 2 to B, 6 to S
        )
        self.common_tester(typical_bid_offer, tests, debug)

    def test_make_order_imbalance_start_with_D(self):
        debug = False
        typical_bid_offer = 2
        tests = (  # tests are applied sequentially to a single OrderImbalance instance
        )
        self.common_tester(typical_bid_offer, tests, debug)


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
        self.bid_window = Windowed(lookback)
        self.offer_window = Windowed(lookback)
        self.running_imbalance = 0

        self.last_restated_trade_type = None

    def p(self):
        'print self'
        format = '%30s: %s'
        print format % ('lookback', self.lookback)
        print format % ('synthetic', self.synthetic)
        print format % ('bid_window', self.bid_window)
        print format % ('offer_window', self.offer_window)
        print format % ('last_restated_trade_type', self.last_restated_trade_type)

    def imbalance(self, trade_type=None, trade_quantity=None, trade_price=None, verbose=True):
        'return the order imbalance after executing the trade'
        def near_mid(current_price):
            'is the current price near to the synthetic mid?'
            pdb.set_trace()
            abs_diff = abs(current_price - self.synthetic.mid)
            if self.synthetic.mid < 0.1:
                return False  # protect against division by zero and near zero
            relative_abs_diff = abs_diff / self.synthetic.mid
            return relative_abs_diff * 100.0 <= self.proximity_cutoff

        def treat_as(restated_trade_type):
            self.last_restated_trade_type = restated_trade_type
            if restated_trade_type == 'B':
                self.synthetic.actual_bid(trade_price)
                self.bid_window.append(trade_quantity)
            elif restated_trade_type == 'S':
                self.synthetic.actual_offer(trade_price)
                self.offer_window.append(trade_quantity)
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
        if trade_type == 'B':
            treat_as('B')
        elif trade_type == 'S':
            treat_as('S')
        else:
            # classify the trade is a dealer buy or sell based on
            # how close it is to prior buys and sells
            pdb.set_trace()
            assert trade_type == 'D'
            if trade_price <= self.synthetic.bid:
                treat_as('B')
            elif trade_price >= self.synthetic.offer:
                treat_as('S')
            elif near_mid(trade_price):
                if self.running_imbalance <= 0:
                    treat_as('S')
                else:
                    treat_as('B')
            else:
                if trade_price <= self.synthetic.mid:
                    treat_as('B')
                else:
                    treat_as('S')

        # classify dealer trade as buy or sell
        # based on how close it is to prior buy and sell prices and
        # its price relative to the synthetic mid price

        self.running_imbalance = self.bid_window.sum() - self.offer_window.sum()
        if verbose:
            self.p()
            print self.running_imbalance
            pdb.set_trace()
        return self.running_imbalance


class TestOrderImbalance4(unittest.TestCase):
    def common_tester(self, lookback, typical_bid_offer, tests, debug=False):
        oi = OrderImbalance4(lookback=lookback, typical_bid_offer=typical_bid_offer)
        for test in tests:
            trade_type, trade_quantity, trade_price, expected_imbalance = test
            actual_imbalance = oi.imbalance(trade_type, trade_quantity, trade_price)
            if debug:
                oi.p()
                print expected_imbalance, actual_imbalance
                print test
                pdb.set_trace()
            self.assertEqual(expected_imbalance, actual_imbalance)

    def test_make_order_imbalance_start_with_B(self):
        return
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
        return
        debug = False
        lookback = 1
        typical_bid_offer = 2
        tests = (  # tests are applied sequentially to a single OrderImbalance instance
            ('D', 4, 101, None),  # get's classified as a B
        )
        self.common_tester(lookback, typical_bid_offer, tests, debug)

    def test_from_gokul(self):
        # debug = True
        oi = OrderImbalance4(
            lookback=100,  # the tests assume a number at least len(tests)
            typical_bid_offer=2,
            proximity_cutoff=20)
        tests = (
            ('B', 1, 101, 'B', 1, 1, 101, 102, 103),
            ('B', 2, 102, 'B', 3, 3, 102, 103, 104),
            ('S', 3, 100, 'S', 0, 0, 98, 99, 100),
            ('D', 4, 101, 'S', 0, -4, 99, 100, 101),
            ('D', 5, 100, 'S', 5, -9, 100, 100, 101),
            # ('D', 6, 103, -1, -5, 101, 102, 103),
            # ('D', 7, 99, 6, 2, 99, 100, 101),
            # ('D', 8, 100.5, -10, -6, 98.5, 99.5, 100.5),
            # ('D', 9, 99.75, None, 3, 99.75, 100.75, 101.75),  # Gogkul wrote xxx as the resulting imbalance
        )
        for test in tests:
            (
                trade_type,
                trade_quantity,
                trade_price,
                expected_restated_trade_type,
                expected_imbalance_gokul,
                expected_imbalance_roy,
                expected_synthetic_bid,
                expected_synthetic_mid,
                expected_synthetic_offer,
            ) = test
            actual_imbalance = oi.imbalance(trade_type, trade_quantity, trade_price)
            self.assertEqual(expected_restated_trade_type, oi.last_restated_trade_type)
            self.assertEqual(expected_imbalance_roy, actual_imbalance)
            self.assertAlmostEqual(expected_synthetic_bid, oi.synthetic.bid)
            self.assertAlmostEqual(expected_synthetic_mid, oi.synthetic.mid)
            self.assertAlmostEqual(expected_synthetic_offer, oi.synthetic.offer)


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
