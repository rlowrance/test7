'''An EventId uniquely identifies a time at which new data become available

A model can be fit at each event identified by an EventId

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''
import datetime
import pdb
import unittest


class EventId(object):
    def __init__(self, year, month, day, hour, minute, second, source, source_id):
        self.year = int(year)
        self.month = int(month)
        self.day = int(day)
        self.hour = int(hour)
        self.minute = int(minute)
        self.second = int(second)
        self.source = source
        self.source_id = source_id
        # check for valid values by attempting to construct
        datetime.date(self.year, self.month, self.day)
        datetime.time(self.hour, self.minute, self.second)

    def __str__(self):
        return '%04d-%02d-%02d-%02d-%02d-%02d-%s-%s' % (
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.source,
            self.source_id,
        )

    def __repr__(self):
        return 'EventId(%s, %s, %s, %s, %s, %s, %s, %s)' % (
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
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
            self.source == other.source and
            self.source_id == other.source_id)

    def __hash__(self):
        return hash((self.year, self.month, self.day, self.hour, self.minute, self.second, self.source, self.source_id))

    @classmethod
    def from_str(cls, s):
        year, month, day, hour, minute, second, source, source_id = s.split('-')
        return cls(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second),
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
            )


class EventIdTest(unittest.TestCase):
    def test_eq(self):
        eid1 = EventId.from_str('2017-07-23-17-55-30-sleep-123')
        eid2 = EventId.from_str('2017-07-23-17-55-30-sleep-123')
        self.assertEqual(eid1, eid2)

    def test_hash(self):
        eid1 = EventId.from_str('2017-07-23-17-55-30-sleep-123')
        eid2 = EventId.from_str('2017-07-23-17-55-30-sleep-123a')
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
            'sleep',
            '123',
        )
        self.assertEqual(2017, event_id.year)
        self.assertEqual(7, event_id.month)
        self.assertEqual(23, event_id.day)
        self.assertEqual(17, event_id.hour)
        self.assertEqual(55, event_id.minute)
        self.assertEqual(30, event_id.second)
        self.assertEqual('sleep', event_id.source)
        self.assertEqual('123', event_id.source_id)

    def test_construction_from_str(self):
        s = '2017-07-23-17-55-30-sleep-123'
        event_id = EventId.from_str(s)
        self.assertEqual(2017, event_id.year)
        self.assertEqual(7, event_id.month)
        self.assertEqual(23, event_id.day)
        self.assertEqual(17, event_id.hour)
        self.assertEqual(55, event_id.minute)
        self.assertEqual(30, event_id.second)
        self.assertEqual('sleep', event_id.source)
        self.assertEqual('123', event_id.source_id)

    def test_construction_bad(self):
        with self.assertRaises(Exception):
            EventId.from_str('2017-04-23-23-59-60-test-boo')  # cannot have 60 seconds

    def test_str(self):
        s = '2017-07-23-17-55-30-sleep-123'
        event_id = EventId.from_str(s)
        self.assertEqual(s, str(event_id))

    def test_datetime(self):
        s = '2017-07-23-17-55-30-sleep-123'
        event_id = EventId.from_str(s)
        self.assertEqual(datetime.datetime(2017, 7, 23, 17, 55, 30), event_id.datetime())


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
