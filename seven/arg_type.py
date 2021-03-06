'''type verifiers for argparse

each function either
- returns an argument parsed from a string (possible the string); OR
- raises argpare.ArgumentTypeError

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''

import argparse
import datetime
import multiprocessing
import os
import pdb

if False:
    pdb


def cusip(s):
    'ref: https://en.wikipedia.org/wiki/CUSIP'
    try:
        assert len(s) == 9
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not a valid CUSIP' % s)


def cusipfile(s):
    try:
        cusip_str, suffix = s.split('.')
        return cusip(cusip_str)
    except:
        raise argparse.ArgumentTypeError('%s is not a filename of the form {cusip}.{suffix}')


def date(s):
    'return datetime.date or raise'
    year, month, day = s.split('-')
    try:
        value = datetime.date(int(year), int(month), int(day))
        return value
    except:
        raise argparse.ArgumentError('%is is not a date in form YYYY-MM-DD' % s)


def date_quarter_start(s):
    'return datetime.date contrained to be on first day of a calendar quarter or raise'
    try:
        x = date(s)
        if x.month in (1, 4, 7, 10) and x.day == 1:
            return x
        else:
            raise argparse.ArgumentError('date %s does not start on a calendar quarter' % s)
    except argparse.ArgumentError:
        raise  # re-raise the exception


def filename_csv(s):
    'file name ending with .csv'
    pieces = s.split('.')
    if pieces[-1] == 'csv':
        return s
    else:
        raise argparse.ArgumentTypeError('%s is not a filename ending in .csv' % s)


def hpset(s):
    try:
        if s.startswith('grid'):
            supposed_digits = s[4:]
            int(supposed_digits)
            return s
    except:
        raise argparse.ArgumentError('%s is not a hyperparameter set name' % s)


def isin(s):
    'ref: https://en.wikipedia.org/wiki/International_Securities_Identification_Number'
    # Components of an ISIN:
    #  2-char country code for issuing country
    #  9-char security identifier (called the NSIN = National Securities Identifying Number)
    #  1-number check digit
    try:
        assert len(s) == 12
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not 12 positions, as required for a valid ISIN' % s)


def issuer(s):
    return ticker(s)


def _in_set(s, allowed):
    'return s or raise ArgumentTypeError'
    try:
        assert s in allowed
        return s
    except:
        raise argparse.ArgumentTypeError('%s not in allowed values {%s}' % (s, allowed))


def target(s):
    return _in_set(s, ['oasspread'])


def trade_id(s):
    try:
        int(s)
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not a trace id (issuepriceid)' % s)


trace_id = trade_id


def month(s):
    's is a string of the form YYYYMM'
    try:
        s_year = s[:4]
        s_month = s[4:]
        int_year = int(s_year)
        assert 0 <= int_year <= 2016
        int_month = int(s_month)
        assert 1 <= int_month <= 12
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not a yearmonth of form YYYYMM' % s)


def n_processes(s):
    'return int value of s, if it is valid for system we are running on'
    cpu_count = multiprocessing.cpu_count()
    try:
        result = int(s)
        assert 1 <= result <= cpu_count
        return result
    except:
        raise argparse.ArgumentTypeError('%s not an itteger in [1,%d]' % (s, cpu_count))


def path_creatable(s):
    'is is a path to a file that can be created'
    try:
        # I can't get the statement below to work
        # assert os.access(s, os.W_OK)
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not a path to a creatable file' % s)


def path_existing(s):
    's is a path to an existing file or directory'
    try:
        assert os.path.exists(s)
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not a path to an existing file or dir' % s)


def positive_int(s):
    'convert s to a positive integer or raise exception'
    try:
        value = int(s)
        assert value > 0
        return value
    except:
        raise argparse.ArgumentTypeError('%s is not a positive integer' % s)


def ticker(s):
    try:
        assert len(s.split('.')) == 1
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not a ticker' % s)


def year(s):
    'convert s to integer that could be a year'
    try:
        assert len(s) == 4
        value = int(s)
        return value
    except:
        raise argparse.ArgumentTypeError('%s is not a year' % s)
