'''read input csv files, specifying their layout details'''
import pandas as pd
import pdb
import unittest

import path


def input(ticker=None, logical_name=None, nrows=None):
    'read pd.DataFrame'
    assert ticker is not None

    parameters_for_ticker = {
        'etf agg': {
            'parse_dates': [0],
        },
        'etf lqd': {
            'parse_dates': [0],
        },
        'fund': {},
        'security master': {
            'parse_dates': ['issue_date', 'maturity_date'],
        },
        'trace': {
            'index_col': 'issuepriceid',
            'parse_dates': ['maturity', 'effectivedate', 'effectivetime'],
        },
    }

    if logical_name in parameters_for_ticker:
        pdb.set_trace()
        parameters = parameters_for_ticker[logical_name]
        parameters['nrows'] = nrows
        parameters['filepath_or_buffer'] = path.input(
            ticker=ticker,
            logical_name=logical_name,
        )
        print logical_name
        print parameters['filepath_or_buffer']
        df = pd.read_csv(**parameters)
        return df
    else:
        print 'error: unknown logical_name', logical_name
        pdb.set_trace()


class TestInput(unittest.TestCase):
    def setUp(self):
        self.verbose = True

    def test_return_string(self):
        verbose = True
        tests = (
            'etf agg',
            'etf lqd',
            'fund',
            'security master',
            'trace',
        )
        for test in tests:
            if verbose:
                print test
            logical_name = test
            df = input(
                ticker='orcl',
                logical_name=logical_name,
                nrows=10,
                )
            self.assertTrue(len(df) > 0)


if __name__ == '__main__':
    unittest.main()
