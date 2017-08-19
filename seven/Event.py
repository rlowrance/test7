'''an Event is the input type from a stream of data

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''
import datetime
import pdb


class Event(object):
    'a stream of data is read by an EventReader, which returns a sequence of Events'
    def __init__(self,
                 year, month, day, hour, minute, second, microsecond,
                 source, source_identifier,
                 payload,
                 make_event_attributes_class):
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
        self.make_event_attributes_class = make_event_attributes_class

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

    def oasspread(self):
        assert self.is_trace_print()
        return self.payload['oasspread']

    def reclassified_trade_type(self):
        'return the reclassified trade type if the event source is a trace print, else raise an exception'
        assert self.is_trace_print()
        return self.payload['reclassified_trade_type']


if __name__ == '__main__':
    if False:
        pdb
