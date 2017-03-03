'''type verifiers for argparse

each function either
- returns an argument parsed from a string (possible the string); OR
- raises argpare.ArgumentTypeError
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
    year, month, day = s.split('-')
    try:
        datetime.date(int(year), int(month), int(day))
        return s
    except:
        raise argparse.ArgumentError('%is is not a date in form YYYY-MM-DD' % s)


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


def _in_set(s, allowed):
    'return s or raise ArgumentTypeError'
    try:
        assert s in allowed
        return s
    except:
        raise argparse.ArgumentTypeError('s not in allowed values {%s}' (s, allowed))


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