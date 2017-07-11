'''
Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''
import pdb
import unittest


class Synthetic(object):
    'maintain synthetic bid, mid, and offer prices'
    def __init__(self, initial_spread=None):
        assert initial_spread >= 0.0
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
        return self

    def actual_offer(self, actual):
        self.offer = actual
        self.bid = actual - self.spread
        self._update_mid()
        return self

    def _update_mid(self):
        self.mid = 0.5 * (self.bid + self.offer)

    def update_spread(self, new_spread):
        self.spread = new_spread
        return self


class Test(unittest.TestCase):
    def test_construction_ok(self):
        tests = (
            0, 1, 10,
        )
        for test in tests:
            x = Synthetic(initial_spread=test)
            self.assertEqual(test, x.spread)

    def test_construction_bad(self):
        tests = (-1, None)
        for test in tests:
            self.assertRaises(Exception, Synthetic, test)

    def test_actual_bid(self):
        initial_spread = 2
        tests = (
            (10, 10, 11, 12),
        )
        for test in tests:
            price, expected_bid, expected_mid, expected_offer = test
            synthetic = Synthetic(initial_spread=initial_spread).actual_bid(price)
            self.assertEqual(expected_bid, synthetic.bid)
            self.assertEqual(expected_mid, synthetic.mid)
            self.assertEqual(expected_offer, synthetic.offer)

    def test_actual_offer(self):
        initial_spread = 2
        tests = (
            (10, 8, 9, 10),
        )
        for test in tests:
            price, expected_bid, expected_mid, expected_offer = test
            synthetic = Synthetic(initial_spread=initial_spread).actual_offer(price)
            self.assertEqual(expected_bid, synthetic.bid)
            self.assertEqual(expected_mid, synthetic.mid)
            self.assertEqual(expected_offer, synthetic.offer)

    def test_update_spread(self):
        tests = (
            (2, 10),
        )
        for test in tests:
            initial_spread, updated_spread = test
            synthetic = Synthetic(initial_spread=initial_spread)
            self.assertEqual(initial_spread, synthetic.spread)
            synthetic.update_spread(updated_spread)
            self.assertEqual(updated_spread, synthetic.spread)


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
