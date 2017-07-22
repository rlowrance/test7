'''determine interarrival times of trace print records

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''
import collections
import datetime
import pdb
import unittest


class InterarrivalTime(object):
    def __init__(self):
        assert False, 'deprecated: use feature_makers.TracerecordInterarrivalTime instead of me'
        self.last_trace_print_effectivedatetime = None

    def interarrival_seconds(self, trace_index, trace_record):
        'return (seconds since last trade, err)'
        previous_effectivedatetime = self.last_trace_print_effectivedatetime
        current_effectivedatetime = trace_record['effectivedatetime']
        self.last_trace_print_effectivedatetime = current_effectivedatetime
        if previous_effectivedatetime is None:
            return (None, 'no prior trace record')
        else:
            interval = current_effectivedatetime - previous_effectivedatetime
            self.last_trace_print_effectivedatetime = current_effectivedatetime
            # interval: Timedelta, a subclass of datetime.timedelta
            # attributes of a datetime.timedelta are days, seconds, microseconds
            interarrival_seconds = (interval.days * 24.0 * 60.0 * 60.0) + (interval.seconds * 1.0)
            return (interarrival_seconds, None)


class TestInterarrivalTime(unittest.TestCase):
    def test1(self):
        Test = collections.namedtuple('Test', 'minute second expected_interval')
        tests = (  # (minute, second, expected_interval)
            Test(10, 0, None),
            Test(10, 0, 0),
            Test(10, 1, 1),
            Test(10, 20, 19),
            Test(10, 20, 0),
        )
        iat = InterarrivalTime()
        for test in tests:
            trace_record = {}
            trace_record['effectivedatetime'] = datetime.datetime(2016, 11, 3, 6, test.minute, test.second)
            actual_interval, err = iat.interarrival_seconds(
                trace_index=None,
                trace_record=trace_record,
            )
            if test.expected_interval is None:
                self.assertTrue(actual_interval is None)
                self.assertTrue(isinstance(err, str))  # it contains an error message
            else:
                self.assertEqual(test.expected_interval, actual_interval)
                self.assertTrue(err is None)


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
