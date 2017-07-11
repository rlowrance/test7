'''read buildinfo files and return specified info

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''
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
            'cusip_effectivedatetime_issuepriceids',
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

    def get_effectivedatetimes(self, cusip):
        'return iterable of datetime.datetime values, when the cusip traded'
        d = self.data['cusip_effectivedatetime_issuepriceids']
        return d[cusip].keys()

    def get_issuepriceids(self, *args):
        def get_issuepriceids1(self, effective_date):
            # effective_date: Union[str, datetime.date]
            return self.data['effectivedate_issuepriceid'][self._as_datetime_date(effective_date)]

        def get_issuepriceids2(self, cusip, effective_datetime):
            assert isinstance(cusip, str)
            assert isinstance(effective_datetime, datetime.datetime)
            d = self.data['cusip_effectivedatetime_issuepriceids']
            dd = d[cusip]
            return dd[effective_datetime]

        if len(args) == 1:
            return get_issuepriceids1(self, *args)
        elif len(args) == 2:
            return get_issuepriceids2(self, *args)
        else:
            print 'unkonwn # args'
            assert len(args) > 0
            assert len(args) <= 2

    def get_issuer(self, cusip):
        assert cusip in self.data['cusips']
        return self.issuer

    def get_ntrades(self, cusip, effective_datetime):
        d = self.data['cusip_effectivedatetime_issuepriceids']
        dd = d[cusip]
        return len(dd[effective_datetime])

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
            Test('AAPL', 127203065, '037833AG5', datetime.datetime(2017, 6, 28, 9, 21, 49)),
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
            self.assertEqual(effectivedatetime.date(), x)

            x = gbi.get_effectivedatetimes(cusip)
            if verbose:
                print x
            self.assertTrue(len(x) > 1)
            self.assertTrue(isinstance(x, list))
            for item in x:
                self.assertTrue(isinstance(item, datetime.datetime))

            x = gbi.get_issuepriceids(effectivedatetime.date())
            if verbose:
                print x
            self.assertTrue(isinstance(x, set))
            self.assertTrue(len(x) > 0)
            self.assertTrue(issuepriceid in x)

            x = gbi.get_issuepriceids('2017-06-28')
            if verbose:
                print x
            self.assertTrue(len(x) > 0)

            x = gbi.get_issuepriceids(cusip, effectivedatetime)
            if verbose:
                print x
            self.assertTrue(isinstance(x, collections.Iterable))

            x = gbi.get_issuer(cusip)
            if verbose:
                print x
            self.assertEqual(issuer, x)

            x = gbi.get_ntrades(cusip, datetime.datetime(2017, 6, 28, 9, 21, 49))
            if verbose:
                print x
            self.assertEqual(3, x)  # see 0log.txt for buildinfo.py to find the value 3


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
