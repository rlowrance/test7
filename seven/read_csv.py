'''read input csv files, specifying their layout details'''
import pandas as pd
import pdb
import unittest

import path


parameters_for_ticker = {
    'etf agg': {
        'index_col': ['asof'],
        'parse_dates': ['asof'],
    },
    'etf lqd': {
        'index_col': ['asof'],
        'parse_dates': ['asof'],
    },
    # 'fund': {},  # first column is empty, second column has the date in Excel format
    'fund': {
        'parse_dates': ['date']
    },
    'map_cusip_ticker': {
        'index_col': 'cusip',
    },
    'ohlc spx': {
        'index_col': 'Date',
        'parse_dates': ['Date'],
    },
    'ohlc ticker': {
        'index_col': 'Date',
        'parse_dates': ['Date'], 
    },
    'security master': {
        'index_col': 'cusip',
        'parse_dates': ['issue_date', 'maturity_date'],
    },
    'trace': {
        'index_col': 'issuepriceid',
        'parse_dates': ['maturity', 'effectivedate', 'effectivetime'],
    },
}


def input(ticker=None, logical_name=None, nrows=None, low_memory=False, verbose=True):
    'read an input files identified by its logical name; return a pd.DataFrame'
    if logical_name in parameters_for_ticker:
        parameters = parameters_for_ticker[logical_name]
        parameters['nrows'] = nrows
        parameters['low_memory'] = low_memory
        parameters['filepath_or_buffer'] = path.input(
            ticker=ticker,
            logical_name=logical_name,
        )
        # check that the parameters are not misleading
        if 'index_col' in parameters and 'usecols' in parameters:
            print 'error: cannot read both the index colum and specific columns'
            print '(possibly a bug n scikit-learn)'
            pdb.set_trace()
        df = pd.read_csv(**parameters)
        print 'read %d rows from csv %s at path %s' % (
            len(df),
            logical_name,
            parameters['filepath_or_buffer'],
        )
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
            'ohlc ticker',
            'ohlc spx',
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
