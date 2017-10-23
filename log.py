'''logging module of msgs to stdout and publishing them to an exchange'''
import datetime
import pdb
import unittest


invoke_pdb = {
    'debug': True,
    'info': False,
    'warning': False,
    'error': True,
    'critical': True,
    }

# if these flags are True, logging at these levels will print
verbose = {
    'debug': True,
    'info': True,
    'warning': True,
    'error': True,
    'critical': True,
    }

channel = None  # if not None, publish to channel

caller = None  # the published messages has the caller in it
logging_exchange = None  # the exchange to publish to


def maybe_announce(level, msg):
    if verbose[level]:
        print('%s: %s' % (level.upper(), str(msg)))
    if channel is not None:
        channel.publish(
            exchange=logging_exchange,
            routing_key='logging.%s.%s.%s' % (caller, level, datetime.datetime.now()),
            body=str(msg),
            )

        
def debug(*msgs):
    'report information for a debugging session'
    for msg in msgs:
        print('DEBUG', msg)
    if invoke_pdb['debug']:
        pdb.set_trace()


def info(*msgs):
    'confirm that things are working as expected'
    for msg in msgs:
        maybe_announce('info', msg)
    if invoke_pdb['info']:
        pdb.set_trace()


def warning(*msgs):
    'indicate something unexpected happened and the sofware is working as expected'
    # or indicate that a problem will occur in the near future (ex: disk space is low)
    for msg in msgs:
        maybe_announce('warning', msg)
    if invoke_pdb['warning']:
        pdb.set_trace()


def error(*msgs):
    'the software was not be able to perform some function, but could continue to run'
    for msg in msgs:
        print('ERROR:', msg)
        maybe_announce('error', msg)
    if invoke_pdb['error']:
        pdb.set_trace()


def error_if_nan(x, *msgs):
    if x != x:
        error(*msgs)


def critical(*msgs):
    'the program may not be able to continue running'
    for msg in msgs:
        maybe_announce('critical', msg)
    if invoke_pdb['critical']:
        pdb.set_trace()


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
        error('did not perform my function')

    def test_error_if_nan(self):
        error_if_nan(123.0)
        error_if_nan(float('Inf'))

    def test_critial(self):
        critical('I must stop')

    def test_critial_if_nan(self):
        critical_if_nan(123.0)
        critical_if_nan(float('Inf'))


if __name__ == '__main__':
    unittest.main()
