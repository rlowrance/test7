'''read input csv files, specifying their layout details

This modules knows the column structure of the input CSV files. It uses that knowledge to
- set the index, which may be 1 or more columns, when there is a meaningful index
- parse dates from text strings in the CSV to Pandas date types

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''
import os
import pandas as pd
import pdb
import pprint
import unittest

import path

pp = pprint.pprint

parameters_for_fundamentals = {
    'index_col': ['date'],
    'parse_dates': ['date'],
}
parameters_for_weights = {
    'index_col': ['date', 'cusip'],
    'parse_dates': ['date'],
}
parameters_dict = {
    'etf agg': {   # deprecated: used only by fit_predict
        'index_col': ['asof'],
        'parse_dates': ['asof'],
    },
    'etf lqd': {   # deprecated: used only by fit_predict
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
    'otr': {
        'index_col': ['date'],
        'parse_dates': ['date'],
    },
    'security master': {
        'index_col': 'CUSIP',
        'parse_dates': ['issue_date', 'maturity_date'],
    },
    'trace': {
        'index_col': 'issuepriceid',
        'parse_dates': ['effectivedate', 'effectivetime'],
    },
    'weight cusip agg': parameters_for_weights,
    'weight cusip lqd': parameters_for_weights,
    'weight issuer agg': parameters_for_weights,
    'weight issuer lqd': parameters_for_weights,
    'expected_interest_coverage': parameters_for_fundamentals,
    'gross_leverage': parameters_for_fundamentals,
    'LTM_EBITDA': parameters_for_fundamentals,
    'mkt_cap': parameters_for_fundamentals,
    'mkt_gross_leverage': parameters_for_fundamentals,
    'reported_interest_coverage': parameters_for_fundamentals,
    'total_assets': parameters_for_fundamentals,
    'total_debt': parameters_for_fundamentals,
}


def working(*args):
    first = args[0]
    if first == 'features_targets':
        return pd.read_csv(
            os.path.join(path.working(), *args),
            parse_dates=(
                ['id_p_effectivedate', 'id_p_effectivedatetime', 'id_p_effectivetime',
                 'id_otr1_effectivedate', 'id_otr1_effectivetime', 'id_otr1_effectivedatetime']
                if args[-1] == 'features.csv' else
                ['id_effectivedate', 'id_effectivetime', 'id_effectivedatetime']
            ),
            index_col=0,
        )
    else:
        print 'bad args to read_csv.working'
        print args
        pdb.set_trace()
    return None


def input(issuer=None, logical_name=None, nrows=None, low_memory=False, verbose=True):
    'read an input files identified by its logical name; return a pd.DataFrame'
    if logical_name in parameters_dict:
        if logical_name == 'weight issuer agg':
            print 'found', logical_name
            pdb.set_trace()
        parameters = read_csv_parameters(
            issuer,
            logical_name,
            nrows=nrows,
            low_memory=low_memory,
        )

        # check that the parameters are not misleading
        if 'index_col' in parameters and 'usecols' in parameters:
            print 'error: cannot read both the index colum and specific columns'
            print '(possibly a bug in Pandas)'
            pdb.set_trace()
        df = pd.read_csv(**parameters)
        print 'read %d rows from logical name %s at path %s' % (
            len(df),
            logical_name,
            parameters['filepath_or_buffer'],
        )
        return df
    else:
        print 'error: unknown logical_name', logical_name
        pdb.set_trace()


def read_csv_parameters(issuer, logical_name, nrows=None, low_memory=False):
    'return parameters for calling pd.read_csv'
    parameters = parameters_dict[logical_name]
    parameters['nrows'] = nrows
    parameters['low_memory'] = low_memory
    parameters['filepath_or_buffer'] = path.input(issuer=issuer, logical_name=logical_name)
    return parameters


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
