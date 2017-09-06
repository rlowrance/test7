'''
facade for python logging module

INVOCATION
   cd src
   python -mseven.logging

   THIS will fail with a error on the import statement
     cd src/seven
     python seven.py


Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''
import pdb
import unittest

from . import exception

invoke_pdb = False
verbose_info = True
verbose_warning = True
verbose_error = True
verbose_critical = True


def debug(*msgs):
    'report information for a debugging session'
    for msg in msgs:
        print('DEBUG', msg)
    if invoke_pdb:
        pdb.set_trace()


def info(*msgs):
    'confirm that things are working as expected'
    if verbose_info:
        for msg in msgs:
            print('INFO:', msg)


def warning(*msgs):
    'indicate something unexpected happened and the sofware is working as expected'
    # or indicate that a problem will occur in the near future (ex: disk space is low)
    if verbose_warning:
        for msg in msgs:
            print('WARNING:', msg)


def error(*msgs):
    'the software was not be able to perform some function, but could continue to run'
    if verbose_error:
        for msg in msgs:
            print('ERROR:', msg)
    if invoke_pdb:
        pdb.set_trace()
    else:
        raise exception.ErrorException(msgs)


def error_if_nan(x, *msgs):
    if x != x:
        error(*msgs)


def critical(*msgs):
    'the program may not be able to continue running'
    if verbose_critical:
        for msg in msgs:
            print('CRITICAL:', msg)
    if invoke_pdb:
        pdb.set_trace()
    else:
        raise exception.CriticalExpection(msgs)


def critical_if_nan(x, *msgs):
    if x != x:
        critical(*msgs)


class Test(unittest.TestCase):
    def test_info(self):
        info('abc')

    def test_warning(self):
        warning(
            'line 1',
            'line 2',
        )

    def test_error(self):
        with self.assertRaises(exception.ErrorException):
            error('did not perform my function')

    def test_error_if_nan(self):
        error_if_nan(123.0)
        error_if_nan(float('Inf'))
        with self.assertRaises(exception.ErrorException):
            error(float('NaN'))

    def test_critial(self):
        with self.assertRaises(exception.ErrorException):
            error('I must stop')

    def test_critial_if_nan(self):
        critical_if_nan(123.0)
        critical_if_nan(float('Inf'))
        with self.assertRaises(exception.ErrorException):
            error(float('NaN'))


if __name__ == '__main__':
    unittest.main()
