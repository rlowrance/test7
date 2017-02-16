from __future__ import division

import pdb
import unittest

from MaybeNumber import MaybeNumber


class OrderImbalance(object):
    def __init__(self, typical_bid_offer=None):
        assert typical_bid_offer > 0

        self.typical_bid_offer = MaybeNumber(typical_bid_offer)

        self.prior_bid_price = MaybeNumber(None)
        self.prior_offer_price = MaybeNumber(None)

        self.cumulatively_bought = MaybeNumber(0)
        self.cumulatively_sold = MaybeNumber(0)

    def p(self):
        'print self'
        format = '%30s: %s'
        print format % ('typical_bid_offer', self.typical_bid_offer)
        print format % ('prior_bid_price', self.prior_bid_price)
        print format % ('prior_offer_price', self.prior_offer_price)
        print format % ('cumulatively_bought', self.cumulatively_bought)
        print format % ('cumulatively_sold', self.cumulatively_sold)

    def open_interest(self, trade_type=None, trade_quantity=None, trade_price=None, verbose=False):
        'return a MaybeNumber, possibly containing the order imbalance, which may not exist'
        def make_bid_weight(dealer_price, bid_price, offer_price):
            'return weight to be given to bid as a MaybeNumber'
            def distance(a, b):
                diff = a - b
                return (diff * diff).sqrt()

            distance_to_bid = distance(dealer_price, bid_price)
            distance_to_offer = distance(dealer_price, offer_price)
            print dealer_price, bid_price, offer_price, distance_to_bid, distance_to_offer
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
            self.prior_bid_price = price
            self.cumulatively_bought += trade_quantity
            if self.prior_offer_price.value is None:
                self.prior_offer_price = self.prior_bid_price + self.typical_bid_offer
                print self.prior_bid_price, self.prior_offer_price
                assert self.prior_bid_price >= self.prior_offer_price  # because prices are OAS spreads
        elif trade_type == 'S':
            # dealer sell, hence an offer
            self.prior_offer_price = price
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

        result = self.cumulatively_bought - self.cumulatively_sold  # a MaybeNumber
        if verbose:
            self.p()
            print result
            pdb.set_trace()
        return result


class TestOrderImbalance(unittest.TestCase):
    def common_tester(self, typical_bid_offer, tests, debug=False):
        oi = OrderImbalance(typical_bid_offer=typical_bid_offer)
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


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
