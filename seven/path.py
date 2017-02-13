'''return path as str to logical file or directory'''
import os
import unittest


def orcl_sample1_csv():
    # TODO: rename to include CUSIP
    return os.path.join(_dropbox(), 'MidPredictor', 'data', 'orcl_order_imb_sample1.csv')


def orcl_sample2_csv():
    # TODO: rename to include CUSIP
    return os.path.join(_dropbox(), 'MidPredictor', 'data', 'orcl_order_imb_sample2.csv')


def _dropbox():
    return os.path.join(_home(), 'Dropbox')


def _home():
    return os.path.join('C:\Users', 'roylo')


class TestPath(unittest.TestCase):
    def setUp(self):
        self.verbose = True

    def test_sample1(self):
        'just test for completion'
        x = orcl_sample1_csv()
        if self.verbose:
            print x

    def test_sample2(self):
        x = orcl_sample2_csv()
        if self.verbose:
            print x

if __name__ == '__main__':
    unittest.main()
