'''computations on EventIds for a specific issuer and cusip

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''
import collections
import datetime
import os
import pdb

import EventId
import exception
import logging
import path


class EventInfo(object):
    def __init__(self, issuer, cusip):
        self.issuer = issuer
        self.cusip = cusip

        # build several tables of event ids
        event_ids = set()
        trade_type = {}
        path_to_event = {}
        dir_in = os.path.join(
            path.working(),
            'features_targets',
            issuer,
            cusip,
        )
        self._events_on_date = collections.defaultdict(list)  # Dict[datetime.date, List[EvenId]]
        self._events_at_datetime = collections.defaultdict(list)  # Dict[datetime.datetime, List[EventId]]
        for dirpath, dirnames, filenames in os.walk(dir_in):
            if dirpath != dir_in:
                break
            for filename in filenames:
                filename_suffix = filename.split('.')[-1]
                if filename_suffix == 'csv':
                    filename_base, reclassified_trade_type = filename.split('.')[:2]
                    event_id = EventId.EventId.from_str(filename_base)
                    self._events_on_date[event_id.date()].append(event_id)
                    self._events_at_datetime[event_id.datetime()].append(event_id)
                    trade_type[event_id] = reclassified_trade_type
                    path_to_event[event_id] = os.path.join(dirpath, filename)
                    if event_id in event_ids:
                        raise exception.EventInfo('event id %s is in the file system more than once' % event_id)
                    event_ids.add(event_id)
        self._events_on_date = dict(self._events_on_date)
        self._events_at_datetime = dict(self._events_at_datetime)

        at_datetime = collections.defaultdict(list)
        on_date = collections.defaultdict(list)
        for event_id in event_ids:
            event_datetime = event_id.datetime()
            event_date = event_datetime.date()
            at_datetime[event_datetime].append(event_id)
            on_date[event_date].append(event_id)

        self._ascending_event_datetimes = sorted(at_datetime.keys())
        self._at_datetime = at_datetime
        self._event_ids = event_ids
        self._on_date = on_date
        self._path_to_event = path_to_event
        self._trade_type = trade_type

    def __repr__(self):
        return 'EventInfo(%s, %s)' % (self.issuer, self.cusip)

    def events_at_datetime(self, when_datetime):
        assert isinstance(at_datetime, datetime.datettime)
        return self._events_at_datetime[when_datetime]

    def events_on_date(self, when_date):
        assert isinstance(when_date, datetime.date)
        return self._events_on_date[when_date]

    def path_to_event(self, event_id):
        'return path in file system to the event'
        assert isinstance(event_id, EventId.EventId)
        if event_id in self._event_ids:
            return self._path_to_event[event_id]
        else:
            print event_id, type(event_id)
            raise exception.EventInfoException('event id %s is not in the file system' % event_id)

    def path_to_fitted_dir(self, event_id, target):
        assert isinstance(event_id, EventId.EventId)
        assert isinstance(target, str)
        result = os.path.join(
            path.working(),
            'fit',
            self.issuer,
            self.cusip,
            target,
            str(event_id),
        )
        return result

    def path_to_predicted_dir(self, event_id, target):
        assert isinstance(event_id, EventId.EventId)
        assert isinstance(target, str)
        result = os.path.join(
            path.working(),
            'predict',
            self.issuer,
            self.cusip,
            target,
            str(event_id),
        )
        return result

    def reclassified_trade_type(self, event_id):
        assert isinstance(event_id, EventId.EventId)
        if event_id in self._event_ids:
            return self._trade_type[event_id]
        else:
            raise exception.EventInfoException('event id %s is not in the file system' % event_id)

    def at_datetime(self, when_datetime):
        'list of paths to all events at the datetime'
        assert isinstance(when_datetime, datetime.datetime)
        return self._at_datetime[when_datetime]

    def on_date(self, when_date):
        'list of paths to all events on the date'
        assert isinstance(when_date, datetime.date)
        return self._on_date[when_date]

    def just_prior_datetime(self, event_id):
        'return datetime just prior to the event_id'
        pdb.set_trace()
        assert isinstance(event_id, EventId.EventId)        
        if event_id not in self._event_ids:
            raise exception.EventInfoException('event id %s is not in the file system' % event_id)
        this_datetime = event_id.datetime()
        for i, dt in self._ascending_event_datetimes():
            if dt == this_datetime:
                if i == 0:
                    print 'no event prior to', this_datetime
                    pdb.set_trace()
                else:
                    return self.event_datetimes[i - 1]
        logging.critical('logic error')

    def prior_datetimes(self, starting_datetime):
        'yield each prior event datetime in descending order'
        for i, dt in enumerate(self._ascending_event_datetimes):
            if dt == starting_datetime:
                prefix = self._ascending_event_datetimes[:i]
                for prior_datetime in reversed(prefix):
                    yield prior_datetime
