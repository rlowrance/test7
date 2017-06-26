'''read buildinfo files and return specified info'''
import cPickle as pickle
import os
import pdb
import unittest

import path  # imports seven.path


class GetBuildInfo(object):
    def __init__(self):
        self.buildinfo_dir = os.path.join(
            path.working(),
            'buildinfo',
        )

    def issuer_for_cusip(self, cusip):
        'return ticker:Str or None'
        with open(os.path.join(self.buildinfo_dir, 'issuers.pickle'), 'rb') as f:
            issuers = pickle.load(f)
            result_set = issuers.get(cusip, None)
            if result_set is None:
                return None
            assert len(result_set) == 1
            for result in result_set:
                return result


class Test(unittest.TestCase):
    # these test cases depend on having run $python buildinfo.py ORCL
    def test1(self):
        tests = (
            '68389XAS4',
            '68389XBH7',
        )
        expected = 'ORCL'
        gbi = GetBuildInfo()
        for cusip in tests:
            self.assertEqual(expected, gbi.issuer_from_cusip(cusip))


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
