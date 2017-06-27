'''read the security master file and return specified info from it'''
import collections
import pdb
import unittest

import read_csv  # imports seven.read_csv


class GetSecurityMasterInfo(object):
    def __init__(self):
        self.df = read_csv.input(logical_name='security master')

    def issuer_for_cusip(self, cusip):
        'return (ticker:Str, None) or (None, err:Str)'
        selected = self.df.loc[cusip]  # a pd.Series
        result = selected['ticker']
        return result


class Test(unittest.TestCase):
    # these test cases depend on having run $python buildinfo.py ORCL
    def test1(self):
        Test = collections.namedtuple('Test', 'cusip issuer')
        tests = (
            Test('17275RAH5', 'CSCO'),
            Test('459200HP9', 'IBM'),
        )
        gsmi = GetSecurityMasterInfo()
        for test in tests:
            self.assertEqual(test.issuer, gsmi.issuer_for_cusip(test.cusip))


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
