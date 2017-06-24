'''return path as str to logical file or directory'''
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


def working():
    return os.path.join(dropbox(), 'data', '7chord', '7chord-01', 'working')


def input(ticker=None, logical_name=None):
    if ticker is not None:
        def make_filename(template):
            if '%s' in template:
                if logical_name == 'fund':
                    # ticker name is all caps in the file name
                    return template % ticker.upper()
                else:
                    return template % ticker
            else:
                return template

        template_equity_ohlc = '%s_equity_ohlc.csv'
        template_eft = '%%s_etf_%s.csv'
        template_fund = '%s.csv'

        filename_templates = {
            'etf agg': ('etf', template_eft % 'agg'),
            'etf lqd': ('etf', template_eft % 'lqd'),
            'fund': ('fundamentals', template_fund),
            'ohlc spx': ('tmp-todelete', template_equity_ohlc % 'spx'),
            'ohlc ticker': ('tmp-todelete', template_equity_ohlc),
            'security master': ('secmaster', '%s_and_comps_sec_master.csv'),
            'trace': ('trace', 'nodupe_trace_%s_otr.csv'),
        }
        if logical_name in filename_templates:
            directory_name, filename_template = filename_templates[logical_name]
            return os.path.join(
                midpredictor(),
                'data',
                directory_name,
                make_filename(filename_template),
            )
        else:
            print 'error: unknown logical_name', logical_name
            pdb.set_trace()
    else:
        if logical_name == 'map_cusip_ticker':
            return os.path.join(
                midpredictor(),
                'automatic feeds',
                'map_cusip_ticker.csv',
            )
        else:
            print 'error: unknown logical_name %s when there is no ticker' % logical_name
            pdb.set_trace()


class TestPath(unittest.TestCase):
    def setUp(self):
        self.verbose = True

    def test_return_string(self):
        verbose = False
        tests = (
            dropbox(),
            home(),
            midpredictor(),
            midpredictor_data(),
            working(),
            input('orcl', 'etf agg'),
            input('orcl', 'etf lqd'),
            input('orcl', 'fund'),
            input('orcl', 'ohlc ticker'),
            input('orcl', 'ohlc spx'),
            input('orcl', 'security master'),
            input('orcl', 'trace'),
        )
        for test in tests:
            if verbose:
                print test
            self.assertTrue(isinstance(test, str))


if __name__ == '__main__':
    unittest.main()
