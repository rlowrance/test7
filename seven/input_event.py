'''an Event is the input type from a stream of data

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''
import datetime
import pdb
import unittest


class Event(object):
    'a stream of data is read by an EventReader, which returns a sequence of Events'
    def __init__(self, 
                 year, month, day, hour, minute, second, microsecond,
                 source, source_identifier, 
                 payload,
                 event_feature_maker_class):
        # the fields up to the payload are required to uniquely identify the event
        assert isinstance(payload, dict)
        # NOTE: these fielda are all part of the API
        self.year = int(year)
        self.month = int(month)
        self.day = int(day)
        self.hour = int(hour)
        self.minute = int(minute)
        self.second = int(second)
        self.microsecond = int(microsecond)  # 0 <= microsecond <= 1_000_000
        self.source = source
        self.source_identifier = source_identifier
        self.payload = payload
        self.event_feature_maker_class = event_feature_maker_class

        # check for valid values by attempting to construct
        datetime.date(self.year, self.month, self.day)
        datetime.time(self.hour, self.minute, self.second, self.microsecond)

    def __repr__(self):
        reclassified_trade_type = self.maybe_reclassified_trade_type()
        return 'Event(%s, %s, %s, %s, %s, %s, %s, %s, %s, %d columns, cusip %s, %s)' % (
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            self.source,
            self.source_identifier,
            len(self.payload),
            self.cusip() if self.is_trace_print() else 'None',
            '' if reclassified_trade_type is None else reclassified_trade_type,
            )

    def _as_tuple(self):
        return (
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            self.source,
            self.source_identifier,
            len(self.payload))

    def __eq__(self, other):
        # ref: http://portingguide.readthedocs.io/en/latest/comparisons.html
        return self._as_tuple() == other._as_tuple()

    def __ge__(self, other):
        return self._as_tuple() >= other._as_tuple()

    def __gt__(self, other):
        return self._as_tuple() > other._as_tuple()

    def __le__(self, other):
        return self._as_tuple() <= other._as_tuple()

    def __lt__(self, other):
        return self._as_tuple() < other._as_tuple()

    def __ne__(self, other):
        return self._as_tuple() != other._as_tuple()

    def __hash__(self):
        return hash((
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            self.source,
            self.source_identifier,
            len(self.payload),
            ))

    def datetime(self):
        'return the date and time as a datetime.datetime object'
        return datetime.datetime(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            )

    def datetime_str(self):
        'return the date and time as a string'
        return '%04d-%02d-%02d-%02d-%02d-%-02d-%06d' % (
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
        )

    def date(self):
        'return the date as a datetime.date object'
        return datetime.date(
            self.year,
            self.month,
            self.day,
            )

    def is_trace_print(self):
        'return True or False'
        return self.source == 'trace'

    def is_trace_print_with_cusip(self, cusip):
        'return True of False'
        if self.is_trace_print():
            return self.maybe_cusip() == cusip
        else:
            return False

    def cusip(self):
        assert self.is_trace_print()
        return self.payload['cusip']

    def maybe_cusip(self):
        ' return the cusip if the source is a trace print, else return None'
        if self.is_trace_print():
            return self.payload['cusip']
        else:
            return None

    def maybe_reclassified_trade_type(self):
        'return the reclassified trade type if the event source is a trace print, else return None'
        if self.is_trace_print():
            return self.payload['reclassified_trade_type']
        else:
            return None

    def reclassified_trade_type(self):
        'return the reclassified trade type if the event source is a trace print, else raise an exception'
        assert self.is_trace_print()
        return self.payload['reclassified_trade_type']


class EventFeatures(object):
    'a dictionary-like object with feature names (str) as keys and non-None values'
    'has a dictionary with restriction on keys and values'
    def __init__(self, *args, **kwargs):
        self.value = dict(*args, **kwargs)

    def __getitem__(self, key):
        assert isinstance(key, str)
        return self.value[key]

    def __setitem__(self, key, value):
        assert isinstance(key, str)
        assert value is not None
        self.value[key] = value

    def __repr__(self):
        return 'EventFeatures(%s)' % self.value

    def __str__(self):
        return 'EventFeatures(%d items)' % len(self.value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).iteritems():
            self.value[k] = v


##############################################################
#  unit tests
##############################################################

# NOTE: These tests are obsolete
# They were written for an older version of the API

class EventIdTest(unittest.TestCase):
    def test_eq(self):
        eid1 = EventId.from_str('2017-07-23-17-55-30-123-sleep-123')
        eid2 = EventId.from_str('2017-07-23-17-55-30-123-sleep-123')
        self.assertEqual(eid1, eid2)

    def test_lt(self):
        eid1 = EventId.from_str('2015-07-23-17-55-30-123-sleep-123')
        eid2 = EventId.from_str('2017-07-23-17-55-30-123-sleep-123')
        self.assertTrue(eid1 < eid2)

    def test_hash(self):
        eid1 = EventId.from_str('2017-07-23-17-55-30-123-sleep-123')
        eid2 = EventId.from_str('2017-07-23-17-55-30-123-sleep-123a')
        self.assertEqual(hash(eid1), hash(eid1))
        self.assertEqual(hash(eid2), hash(eid2))
        self.assertNotEqual(hash(eid1), hash(eid2))
        d = {eid1: 'one', eid2: 'two'}
        self.assertTrue(eid1 in d)
        self.assertTrue(eid2 in d)

    def test_construction_from_details(self):
        event_id = EventId(
            2017,
            7,
            23,
            17,
            55,
            30,
            123,
            'sleep',
            '123',
        )
        self.assertEqual(2017, event_id.year)
        self.assertEqual(7, event_id.month)
        self.assertEqual(23, event_id.day)
        self.assertEqual(17, event_id.hour)
        self.assertEqual(55, event_id.minute)
        self.assertEqual(30, event_id.second)
        self.assertEqual(123, event_id.microsecond)
        self.assertEqual('sleep', event_id.source)
        self.assertEqual('123', event_id.source_id)

    def test_construction_from_str(self):
        s = '2017-07-23-17-55-30-123-sleep-123'
        event_id = EventId.from_str(s)
        self.assertEqual(2017, event_id.year)
        self.assertEqual(7, event_id.month)
        self.assertEqual(23, event_id.day)
        self.assertEqual(17, event_id.hour)
        self.assertEqual(55, event_id.minute)
        self.assertEqual(30, event_id.second)
        self.assertEqual(123, event_id.microsecond)
        self.assertEqual('sleep', event_id.source)
        self.assertEqual('123', event_id.source_id)

    def test_construction_bad(self):
        with self.assertRaises(Exception):
            EventId.from_str('2017-04-23-23-59-60-123-test-boo')  # cannot have 60 seconds

    def test_str(self):
        s = '2017-07-23-17-55-30-123-sleep-123'
        event_id = EventId.from_str(s)
        self.assertEqual(s, str(event_id))

    def test_datetime(self):
        s = '2017-07-23-17-55-30-123-sleep-123'
        event_id = EventId.from_str(s)
        self.assertEqual(datetime.datetime(2017, 7, 23, 17, 55, 30, 123), event_id.datetime())


class OtrCusipEventIdTest(unittest.TestCase):
    def test(self):
        e = OtrCusipEventId('2015-05-03', 'AAPL')
        self.assertTrue(isinstance(e, EventId))


class TotalDebtEventIdTest(unittest.TestCase):
    def test(self):
        e = TotalDebtEventId('2013-05-03', 'AAPL')
        self.assertTrue(isinstance(e, EventId))


class TraceEventIdTest(unittest.TestCase):
    def test(self):
        e = TraceEventId('2015-06-19', '14:54:37', 'AAPL', '100265211')
        self.assertTrue(isinstance(e, EventId))


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
