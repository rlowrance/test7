'''An EventId uniquely identifies a time at which new data become available

A model can be fit at each event identified by an EventId

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''
import datetime
import pdb
import unittest


class EventId(object):
    def __init__(self, year, month, day, hour, minute, second, microsecond, source, source_id):
        self.year = int(year)
        self.month = int(month)
        self.day = int(day)
        self.hour = int(hour)
        self.minute = int(minute)
        self.second = int(second)
        self.microsecond = int(microsecond)  # 0 <= microsecond <= 1_000_000
        self.source = source
        self.source_id = source_id
        # check for valid values by attempting to construct
        datetime.date(self.year, self.month, self.day)
        datetime.time(self.hour, self.minute, self.second, self.microsecond)

    def __str__(self):
        return '%04d-%02d-%02d-%02d-%02d-%02d-%03d-%s-%s' % (
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
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
            self.microsecond,
            self.source,
            self.source_id,
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
            self.source_id)

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
            self.source_id,
            ))

    @classmethod
    def from_str(cls, s):
        year, month, day, hour, minute, second, microsecond, source, source_id = s.split('-')
        return cls(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second),
            int(microsecond),
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
            self.microsecond,
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


class OtrCusipEventId(EventId):
    def __init__(self, date, issuer):
        assert isinstance(date, str)
        assert isinstance(issuer, str)
        year, month, day = date.split('-')
        super(OtrCusipEventId, self).__init__(
            int(year),
            int(month),
            int(day),
            0,
            0,
            0,
            0,
            'liq_flow_on_the_run_%s' % issuer,
            date.replace('-', ''),
        )


class OtrCusipEventIdTest(unittest.TestCase):
    def test(self):
        e = OtrCusipEventId('2015-05-03', 'AAPL')
        self.assertTrue(isinstance(e, EventId))


class TotalDebtEventId(EventId):
    def __init__(self, date, issuer):
        assert isinstance(date, str)
        assert isinstance(issuer, str)
        year, month, day = date.split('-')
        super(TotalDebtEventId, self).__init__(
            int(year),
            int(month),
            int(day),
            0,
            0,
            0,
            0,
            'total_debt_%s' % issuer,
            date.replace('-', ''),
        )


class TotalDebtEventIdTest(unittest.TestCase):
    def test(self):
        e = TotalDebtEventId('2013-05-03', 'AAPL')
        self.assertTrue(isinstance(e, EventId))


class TraceEventId(EventId):
    def __init__(self, effective_date, effective_time, issuer, issuepriceid):
        assert isinstance(effective_date, str)
        assert isinstance(effective_time, str)
        assert isinstance(issuer, str)
        assert isinstance(issuepriceid, str)
        year, month, day = effective_date.split('-')
        hour, minute, second = effective_time.split(':')
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


class TraceEventIdTest(unittest.TestCase):
    def test(self):
        e = TraceEventId('2015-06-19', '14:54:37', 'AAPL', '100265211')
        self.assertTrue(isinstance(e, EventId))


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
