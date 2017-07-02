'''read buildinfo files and return specified info'''
import collections
import cPickle as pickle
import datetime
import numbers
import pdb
import unittest

import path  # imports seven.path


class GetBuildInfo(object):
    def __init__(self, issuer):
        self.issuer = issuer
        self.data = {}
        data_sets = (  # these are the base filenames created by buildinfo.py
            'cusips',
            'effectivedate_issuepriceid',
            'issuepriceid_cusip',
            'issuepriceid_effectivedate'
        )
        for data_set in data_sets:
            path_in = path.input(issuer, 'buildinfo ' + data_set)
            with open(path_in, 'rb') as f:
                obj = pickle.load(f)
            self.data[data_set] = obj

    def get_cusip(self, issuepriceid):
        assert self._is_issuepriceid(issuepriceid)
        assert isinstance(issuepriceid, numbers.Number)
        return self.data['issuepriceid_cusip'][issuepriceid]

    def get_cusips(self, issuer):
        assert issuer == self.issuer
        return self.data['cusips']

    def get_effectivedate(self, issuepriceid):
        assert self._is_issuepriceid(issuepriceid)
        return self.data['issuepriceid_effectivedate'][issuepriceid]

    def get_issuepriceids(self, effective_date):
        # effective_date: Union[str, datetime.date]
        return self.data['effectivedate_issuepriceid'][self._as_datetime_date(effective_date)]

    def get_issuer(self, cusip):
        assert cusip in self.data['cusips']
        return self.issuer

    def _is_issuepriceid(self, x):
        return isinstance(x, numbers.Number)

    def _as_datetime_date(self, x):
        if isinstance(x, datetime.date):
            return x
        elif isinstance(x, str):
            split_x = x.split('-')
            assert len(split_x) == 3
            year, month, day = split_x
            return datetime.date(int(year), int(month), int(day))
        else:
            print '_as_datetime_date: type not handled', type(x), x
            pdb.set_trace()


class Test(unittest.TestCase):
    # these test cases depend on having run $python buildinfo.py ORCL
    def test1(self):
        'test that runs to completion'
        verbose = False
        Test = collections.namedtuple('Test', 'issuer issuepriceid cusip effectivedatetime')
        tests = (
            Test('AAPL', 126132796, '037833AG5', datetime.date(2017, 6, 1)),
        )
        for issuer, issuepriceid, cusip, effectivedatetime in tests:
            gbi = GetBuildInfo(issuer)

            x = gbi.get_cusip(issuepriceid)  # from features_targets/AAPL/037833AG5/common_trace_indices.txt
            if verbose:
                print x
            self.assertEqual(cusip, x)

            x = gbi.get_cusips(issuer)
            if verbose:
                print x
            self.assertTrue(isinstance(x, set))
            self.assertTrue(len(x) > 0)
            self.assertTrue(cusip in x)

            x = gbi.get_effectivedate(issuepriceid)
            if verbose:
                print x
            self.assertEqual(effectivedatetime, x)

            x = gbi.get_issuepriceids(effectivedatetime)
            if verbose:
                print x
            self.assertTrue(isinstance(x, set))
            self.assertTrue(len(x) > 0)
            self.assertTrue(issuepriceid in x)

            x = gbi.get_issuer(cusip)
            if verbose:
                print x
            self.assertEqual(issuer, x)


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
