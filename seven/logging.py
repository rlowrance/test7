'''
facade for python logging module


Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''
import pdb
import unittest

import exception

invoke_pdb = False


def debug(*msgs):
    'report information for a debugging session'
    for msg in msgs:
        print 'DEBUG', msg
    if invoke_pdb:
        pdb.set_trace()


def info(*msgs):
    'confirm that things are working as expected'
    for msg in msgs:
        print 'INFO:', msg


def warning(*msgs):
    'indicate something unexpected happened and the sofware is working as expected'
    # or indicate that a problem will occur in the near future (ex: disk space is low)
    for msg in msgs:
        print 'WARNING:', msg


def error(*msgs):
    'the software was not be able to perform some function, but could continue to run'
    for msg in msgs:
        print 'ERROR:', msg
    if invoke_pdb:
        pdb.set_trace()
    else:
        raise exception.ErrorException(msgs)


def critical(*msgs):
    'the program may not be able to continue running'
    for msg in msgs:
        print 'CRITICAL:', msg
    if invoke_pdb:
        pdb.set_trace()
    else:
        raise exception.CriticalExpection(msgs)


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

    def test_critial(self):
        with self.assertRaises(exception.ErrorException):
            error('I must stop')


if __name__ == '__main__':
    unittest.main()
