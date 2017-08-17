'''computations on EventIds for a specific issuer and cusip

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''
import collections
import datetime
import os
import pdb

import exception
import logging
import path


class EventInfo(object):
    def __init__(self, issuer, cusip):
        self.issuer = issuer
        self.cusip = cusip

        # build several tables of event ids
        dir_in = os.path.join(
            path.working(),
            'features_targets',
            issuer,
            cusip,
        )
        dates = set()
        datetimes = set()
        event_ids = set()
        events_on_date = collections.defaultdict(list)  # Dict[datetime.date, List[EvenId]]
        events_at_datetime = collections.defaultdict(list)  # Dict[datetime.datetime, List[EventId]]
        path_to_features = {}  # Dict[event_id, os.path]
        reclassified_trade_type = {}  # Dict[event_id, reclassified_trade_type]
        for filename in os.listdir(dir_in):
            if not filename.endswith('.csv'):
                continue
            event_id_str, reclassified_trade_type_value, suffix = filename.split('.')
            event_id = EventId.EventId.from_str(event_id_str)

            # accumulate values
            dates.add(event_id.date())
            datetimes.add(event_id.datetime())

            if event_id in event_ids:
                raise exception.EventInfo('event id %s is in the file system more than once' % event_id)
            else:
                event_ids.add(event_id)

            events_on_date[event_id.date()].append(event_id)
            events_at_datetime[event_id.datetime()].append(event_id)
            reclassified_trade_type[event_id] = reclassified_trade_type_value
            path_to_features[event_id] = os.path.join(dir_in, filename)

        # make accumulated values available to methods

        sorted_datetimes = sorted(datetimes)
        self._next_datetime = {}  # Dict[datetime, datetime]
        for i, dt in enumerate(sorted_datetimes):
            if (i + 1) < len(sorted_datetimes):
                self._next_datetime[dt] = sorted_datetimes[i + 1]

        # assure sorted low to high by date or datetime
        self._sorted_events_on_date = collections.OrderedDict()
        for key in sorted(events_on_date.keys()):
            self._sorted_events_on_date[key] = sorted(events_on_date[key])

        self._sorted_events_at_datetime = collections.OrderedDict()
        for key in sorted(events_at_datetime.keys()):
            self._sorted_events_at_datetime[key] = sorted(events_at_datetime[key])

        self._event_ids = event_ids
        self._path_to_features = path_to_features
        self._reclassified_trade_type = reclassified_trade_type
        self._sorted_dates = sorted(dates)
        self._sorted_datetimes = sorted(datetimes)
    
    def __repr__(self):
        return 'EventInfo(%s, %s)' % (self.issuer, self.cusip)

    def at_datetime(self, when_datetime):
        'list of paths to all events at the datetime'
        assert isinstance(when_datetime, datetime.datetime)
        return self._at_datetime[when_datetime]

    def events_at_datetime(self, when_datetime):
        'return List[EventId], possibly empty'
        assert isinstance(when_datetime, datetime.datetime)
        return self._sorted_events_at_datetime.get(when_datetime, [])

    def just_prior_datetime(self, event_id):
        'return datetime just prior to the event_id'
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

    def just_prior_distinct_datetime_event(self, query_event_id, reclassified_trade_type):
        'return the oldest prior event that has a distinct datetime and same reclassified trade type'
        assert isinstance(query_event_id, EventId.EventId)
        assert reclassified_trade_type in ('B', 'S')
        for prior_datetime in reversed(self.sorted_prior_datetimes(query_event_id.datetime())):
            prior_event_ids = self.sorted_events_at_datetime(prior_datetime)
            if len(prior_event_ids) == 1:
                prior_event_id = prior_event_ids[0]
                if self.reclassified_trade_type(prior_event_id) == reclassified_trade_type:
                    return prior_event_id
        raise exception.EventInfoException(
            'no prior event with a distinct datetime and trade type %s for event %s' % (
                reclassified_trade_type,
                query_event_id,
                )
        )

    def just_prior_distinct_datetime_event_on(self, reclassified_trade_type, trade_date):
        'return the oldest prior event that has a distinct datetime and same reclassified trade type'
        assert reclassified_trade_type in ('B', 'S')
        assert isinstance(trade_date, datetime.date)
        for event_id in list(reversed(self.sorted_events_on_date(trade_date))):
            if len(self.sorted_events_at_datetime(event_id.datetime())) == 1:
                if self.reclassified_trade_type(event_id) == reclassified_trade_type:
                    return event_id
        raise exception.EventInfoException(
            'no prior event with a distinct datetime and trade type %s on date %s' % (
                reclassified_trade_type,
                trade_date,
                )
        )

    def next_datetime(self, dt):
        'return datetime for the just subsequent event, or None'
        assert isinstance(dt, datetime.datetime)
        return self._next_datetime.get(dt, None)

    def path_to_features(self, event_id):
        'return path in file system to the features for the event_id'
        assert isinstance(event_id, EventId.EventId)
        if event_id in self._event_ids:
            return self._path_to_features[event_id]
        else:
            print event_id, type(event_id)
            raise exception.EventInfoException('event id %s is not in the file system' % event_id)

    def paths_to_features_on_date(self, when_date):
        'list of paths to all events on the date'
        assert isinstance(when_date, datetime.date)
        result = []
        for event_id in self._sorted_events_on_date[when_date]:
            if event_id.date() == when_date:
                result.append(self._path_to_event[event_id])
        pdb.set_trace()
        return result

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
            return self._reclassified_trade_type[event_id]
        else:
            raise exception.EventInfoException('event id %s is not in the file system' % event_id)

    def sorted_events_on_date(self, when_date):
        'return possibly empty list of events on the date in ascending datetime order'
        assert isinstance(when_date, datetime.date)
        return self._sorted_events_on_date.get(when_date, [])

    def sorted_events_at_datetime(self, when_datetime):
        'return list of events at the datetime'
        assert isinstance(when_datetime, datetime.datetime)
        return self._sorted_events_at_datetime[when_datetime]

    def sorted_prior_datetimes(self, starting_datetime):
        'return list of prior datetime in ascending datetime order'
        result = [
            trade_datetime
            for trade_datetime in self._sorted_datetimes  # in ascending order
            if trade_datetime < starting_datetime
        ]
        return result
