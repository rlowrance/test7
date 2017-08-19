'''handle fundamentals for a specified issuer

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''
import collections
import datetime
import pdb
import unittest

import read_csv  # imports seven/read_csv


class Fundamentals(object):
    def __init__(self, issuer):
        assert False, 'deprecated: use feature_makers.Fundamentals instead'
        self.issuer = issuer
        self.file_logical_names = (  # feature names are the same as the logical names
            'expected_interest_coverage',
            'gross_leverage',
            'LTM_EBITDA',
            'mkt_cap',
            'mkt_gross_leverage',
            'reported_interest_coverage',
            'total_assets',
            'total_debt',
        )
        self.data = self._read_files()  # create Dict[content_name: str, Dict[datetime.date, content_value:float]]

    def logical_names(self):
        return self.file_logical_names

    def get(self, date, logical_name):
        'return (value on or just the specified date, None) or (None, err)'
        data = self.data[logical_name]
        for sorted_date in sorted(data.keys(), reverse=True):
            if date >= sorted_date:
                result = data[sorted_date]
                return (result, None)
        return (None, 'date %s not in fundamentals for issuer %s content %s' % (date, self.issuer, logical_name))

    def file_names(self):
        'return list of file names to read'
        result = []
        for d in self.file_content:
            result.append(d._make_file_name(d))
        return result

    def _read_files(self):
        'return Dict[datetime.date, Dict[content_name:str, content_value:float]]'
        result = {}
        for logical_name in self.file_logical_names:
            result[logical_name] = self._read_file(logical_name)
        return result

    def _read_file(self, logical_name):
        df = read_csv.input(issuer=self.issuer, logical_name=logical_name)
        if len(df) == 0:
            print 'df has zero length', logical_name
            pdb.set_trace()
        result = {}
        for timestamp, row in df.iterrows():
            result[timestamp.date()] = row[0]
        return result


class Test(unittest.TestCase):
    def test_construction_and_file_layouts(self):
        'test all issuers'
        tests = ('AAPL', 'AMZN', 'CSCO', 'GOOGL', 'IBM', 'MSFT', 'ORCL')
        # tests = ('AMZN',)
        # tests = ('AAPL',)
        for test in tests:
            # just test that the construction completes
            fundamentals = Fundamentals(test)
            self.assertTrue(isinstance(fundamentals, Fundamentals))

    def test_get_ok(self):
        'test just AAPL'
        Test = collections.namedtuple('Test', 'date field expected_value')
        tests = (
            Test('2013-05-03', 'total_debt', 14460465000),
            Test('2013-05-04', 'total_debt', 14460465000),
            Test('2013-06-29', 'total_debt', 16958000000),
        )
        fundamentals = Fundamentals('AAPL')
        for test in tests:
            print test
            year, month, day = test.date.split('-')
            date = datetime.date(int(year), int(month), int(day))
            value, err = fundamentals.get(date, test.field)
            self.assertTrue(err is None)
            self.assertEqual(test.expected_value, value)

    def test_get_bad(self):
        'test just AAPL'
        Test = collections.namedtuple('Test', 'date field')
        tests = (
            Test('2013-05-02', 'total_debt'),
        )
        fundamentals = Fundamentals('AAPL')
        for test in tests:
            print test
            year, month, day = test.date.split('-')
            date = datetime.date(int(year), int(month), int(day))
            value, err = fundamentals.get(date, test.field)
            self.assertTrue(value is None)
            self.assertTrue(err is not None)


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
