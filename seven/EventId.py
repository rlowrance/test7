'''An EventId uniquely identifies a time at which new data become available

A model can be fit at each event identified by an EventId

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''
import datetime
import pdb
import unittest


class EventId(object):
    def __init__(self, year, month, day, hour, minute, second, millisecond, source, source_id):
        self.year = int(year)
        self.month = int(month)
        self.day = int(day)
        self.hour = int(hour)
        self.minute = int(minute)
        self.second = int(second)
        self.millisecond = int(millisecond)
        self.source = source
        self.source_id = source_id
        # check for valid values by attempting to construct
        datetime.date(self.year, self.month, self.day)
        datetime.time(self.hour, self.minute, self.second, self.millisecond)

    def __str__(self):
        return '%04d-%02d-%02d-%02d-%02d-%02d-%03d-%s-%s' % (
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.millisecond,
            self.source,
            self.source_id,
        )

    def __repr__(self):
        return 'EventId(%s, %s, %s, %s, %s, %s, %s, %s, %s)' % (
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.millisecond,
            self.source,
            self.source_id,
            )

    def __eq__(self, other):
        return (
            self.year == other. year and
            self.month == other.month and
            self.day == other.day and
            self.hour == other.hour and
            self.minute == other.minute and
            self.second == other.second and
            self.millisecond == other.millisecond and
            self.source == other.source and
            self.source_id == other.source_id)

    def __hash__(self):
        return hash((
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.millisecond,
            self.source,
            self.source_id,
            ))

    @classmethod
    def from_str(cls, s):
        year, month, day, hour, minute, second, millisecond, source, source_id = s.split('-')
        return cls(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second),
            int(millisecond),
            source,
            source_id,
        )

    def datetime(self):
        'return the date and time as a datetime.datetime object'
        return datetime.datetime(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.millisecond,
            )

    def date(self):
        'return the date as a datetime.date object'
        return datetime.date(
            self.year,
            self.month,
            self.day,
            )


class EventIdTest(unittest.TestCase):
    def test_eq(self):
        eid1 = EventId.from_str('2017-07-23-17-55-30-123-sleep-123')
        eid2 = EventId.from_str('2017-07-23-17-55-30-123-sleep-123')
        self.assertEqual(eid1, eid2)

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
        self.assertEqual(123, event_id.millisecond)
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
        self.assertEqual(123, event_id.millisecond)
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


class TraceEventId(EventId):
    def __init__(self, effective_date_str, effective_time_str, issuer, issuepriceid):
        year, month, day = effective_date_str.split('-')
        hour, minute, second = effective_time_str.split(':')
        super(TraceEventId, self).__init__(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second),
            0,
            'trace_%s' % issuer,
            issuepriceid,
        )


class TestTraceEventId(unittest.TestCase):
    def test(self):
        e = TraceEventId('2015-06-19', '14:54:37', 'AAPL', '100265211')
        self.assertTrue(isinstance(e, EventId))


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
