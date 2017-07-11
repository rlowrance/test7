'''map a logical file name to an os path in the file system

Account for current structure of the file system
Account for operating system (use the right path separator characters)

logical file names handled:
  security master
  trace
  weight {cusip|issuer} {agg|lqd}  (we actually handle "weight * *")

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''
import os
import pdb
import sys
import unittest


def dropbox():
    return os.path.join(home(), 'Dropbox')


def home():
    # TODO: make work on unix as well as Windows
    if os.name == 'nt':
        return os.path.join('C:', r'\Users', 'roylo')
    if os.name == 'posix':  # mac and linux
        return os.path.join('/Users/roy/')
    print 'provide path to home directory of os.name', os.name
    sys.exit(1)


def input_dir():
    return os.path.join(dropbox(), 'data', '7chord', '7chord-01', 'input')


def midpredictor():
    return os.path.join(dropbox(), 'MidPredictor')


def midpredictor_data():
    return os.path.join(dropbox(), 'MidPredictor', 'data')


def src():
    return os.path.join(dropbox(), 'ads', 'clients', '7chord', '7chord-01', 'src')


def working():
    return os.path.join(dropbox(), 'data', '7chord', '7chord-01', 'working')


fundamental_file_logical_names = set([
    'expected_interest_coverage',
    'gross_leverage',
    'LTM_EBITDA',
    'mkt_cap',
    'mkt_gross_leverage',
    'reported_interest_coverage',
    'total_assets',
    'total_debt',
])


def input(issuer=None, logical_name=None):
    'return os path'
    if issuer is not None:
        def make_filename(template):
            if '%s' in template:
                if logical_name == 'fund':
                    # ticker name is all caps in the file name
                    return template % issuer.upper()
                else:
                    return template % issuer
            else:
                return template

        if isinstance(logical_name, dict):
            pdb.set_trace()
            if logical_name['kind'] == 'features':
                return os.path.join(
                    working(),
                    'features_targets',
                    issuer,
                    logical_name['cusip'],
                    logical_name['date'],
                    'features.csv',
                )
            elif logical_name['kind'] == 'targets':
                return os.path.join(
                    working(),
                    'features_targets',
                    issuer,
                    logical_name['cusip'],
                    logical_name['date'],
                    'features.csv',
                )
            else:
                print logical_name
                print 'unkown kind'
                pdb.set_trace()
        if logical_name.startswith('buildinfo '):
            filenamebase = logical_name[len('buildinfo '):]
            return os.path.join(
                working(),
                'buildinfo',
                issuer,
                filenamebase + '.pickle',
            )
        if logical_name in fundamental_file_logical_names:
            filename = 'fun_%s_%s.csv' % (logical_name, issuer)
            return os.path.join(
                midpredictor(),
                'automatic feeds',
                filename,
            )
        if logical_name == 'otr':
            return os.path.join(
                midpredictor(),
                'automatic feeds',
                'liq_flow_on_the_run_%s.csv' % issuer,
            )
        if logical_name == 'trace':
            return os.path.join(
                midpredictor(),
                'automatic feeds',
                'trace_%s.csv' % issuer,
            )
        else:
            print 'error: unknown logical_name %s for issuer %s' % (logical_name, issuer)
            pdb.set_trace()
    else:
        # if logical_name == 'trace':
        #     issuer = GetSecurityMasterInfo.GetSecurityMasterInfo().issuer_for_cusip()
        #     return os.path.join(
        #         midpredictor(),
        #         'automatic feeds',
        #         'TRACE_production.csv',
        #     )
        if logical_name == 'security master':
            return os.path.join(
                midpredictor(),
                'automatic feeds',
                'secmaster.csv',
            )
        if logical_name == 'map_cusip_ticker':
            return os.path.join(
                midpredictor(),
                'automatic feeds',
                'map_cusip_ticker.csv',
            )
        logical_name_part = logical_name.split(' ')
        if logical_name_part[0] == 'weight':
            assert len(logical_name_part) == 3
            filename = 'etf_weight_of_%s_pct_%s.csv' % (logical_name_part[1], logical_name_part[2])
            return os.path.join(
                midpredictor(),
                'automatic feeds',
                filename
            )
        else:
            print 'error: unknown logical_name %s when there is no issuer' % logical_name
            pdb.set_trace()


class TestPath(unittest.TestCase):
    def setUp(self):
        self.verbose = True

    def test_return_string(self):
        'just test that a string is returned'
        self.assertFalse()  # all these tests are for a dprecated version of this code
        verbose = False
        tests = (
            dropbox(),
            home(),
            midpredictor(),
            midpredictor_data(),
            working(),
            input('orcl', 'fund'),
            input('orcl', 'ohlc ticker'),
            input('orcl', 'ohlc spx'),
            input('orcl', 'security master'),
            input('orcl', 'trace'),
            input(None, 'weight cusip agg'),
            input(None, 'weight cusip lqd'),
            input(None, 'weight issuer agg'),
            input(None, 'weight issuer lqd'),
        )
        for test in tests:
            if verbose:
                print test
            self.assertTrue(isinstance(test, str))

    def test_correct_filename(self):
        'test that the correct filename is returned'
        # don't test that the entire path is correct
        self.assertFalse()  # all these tests are for a dprecated version of this code
        verbose = False
        tests = (
            (None, 'trace', 'TRACE_production.csv'),
            (None, 'security master', 'SecMaster_production.csv'),
            (None, 'weight cusip agg', 'etf_weight_of_cusip_pct_agg.csv'),
            (None, 'weight cusip lqd', 'etf_weight_of_cusip_pct_lqd.csv'),
            (None, 'weight issuer agg', 'etf_weight_of_issuer_pct_agg.csv'),
            (None, 'weight issuer lqd', 'etf_weight_of_issuer_pct_lqd.csv'),
        )
        for test in tests:
            issuer, logical_name, expected_filename = test
            actual_path = input(issuer=issuer, logical_name=logical_name)
            actual_filename = os.path.basename(actual_path)
            if verbose:
                print test, actual_path, actual_filename
            self.assertEqual(expected_filename, actual_filename)


if __name__ == '__main__':
    unittest.main()
