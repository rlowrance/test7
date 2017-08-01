'''create predictions and test their accuracy

NOTE: This program looks at all trades for the CUSIP and related OTR CUSIPs from the
beginning of time. Thus trades 10 years ago are used to predict next oasspreads today.
That is way too much history. We should fix this when we have a streaming infrastructure.
Most likely, only the last 1000 or so trades are relevant.

INVOCATION
  python test_train.py {issuer} {cusip} {target} {start_date} {--debug} {--test} {--trace}
where
 issuer the issuer (ex: AAPL)
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 start_date: YYYY-MM-DD is the first date for training a model
 --debug means to call pdb.set_trace() instead of raisinng an exception, on calls to logging.critical()
   and logging.error()
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
  python features_targets.py AAPL 037833AJ9 oasspread 2017-07-18 #

See build.py for input and output files.

IDEA FOR FUTURE:

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''

from __future__ import division

import abc
import argparse
import copy
import csv
import collections
import datetime
import gc
import heapq
import numpy as np
import os
import cPickle as pickle
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import applied_data_science.debug
import applied_data_science.dirutility
import applied_data_science.lower_priority
import applied_data_science.pickle_utilities

from applied_data_science.Bunch import Bunch
from applied_data_science.Date import Date
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.accumulators
import seven.arg_type
import seven.build
import seven.EventId
import seven.feature_makers2
import seven.fit_predict_output
import seven.logging
import seven.read_csv

pp = pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('issuer', type=seven.arg_type.issuer)
    parser.add_argument('cusip', type=seven.arg_type.cusip)
    parser.add_argument('target', type=seven.arg_type.target)
    parser.add_argument('start_date', type=seven.arg_type.date)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')

    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()
    if arg.debug:
        # logging.error() and logging.critial() call pdb.set_trace() instead of raising an exception
        seven.logging.invoke_pdb = True

    random_seed = 123
    random.seed(random_seed)

    paths = seven.build.test_train(arg.issuer, arg.cusip, arg.target, arg.start_date, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    timer = Timer()

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=timer,
    )


def make_elapsed(control):
    def elapsed():
        'return elapsed wallclock seconds'
        return control.timer.elapsed_wallclock_seconds()


def select_relevant_cusips(records, control):
    'return DataFrame containing just the records related to the query cusip'
    # these are the records for the CUSIP and for any on-the-run bond for the CUSIP
    # the on-the-run bonds can vary by trace print

    print 'STUB: select_relevant_cusips: just use the primary cusip'
    query_cusip = control.arg.cusip
    # the team is working on building a file that maps cusip -> effective_date -> otr-cusip
    # once they finish, this function will need to be modified
    if True:
        cusip_records = records.loc[records['cusip'] == query_cusip]
        return cusip_records
    if False:
        # explore otr_cusips for each cusip
        cusips = set(records['cusip'])
        for cusip in cusips:
            cusip_records = records.loc[cusip == records['cusip']]
            otr_cusips = set(cusip_records['cusip1'])
            print 'cusip', cusip, 'otr cusips', otr_cusips
    print 'columns in the trace print file', records.columns
    print 'all cusips in trace print file', set(records['cusip'])
    has_query_cusip = records['cusip'] == query_cusip
    cusip_records = records.loc[has_query_cusip]
    assert len(cusip_records > 0)
    otr_cusips = set(cusip_records['cusip1'])
    print 'all OTR cusips for query cusip %s: %s' % (query_cusip, otr_cusips)
    has_otr = pd.Series(
        data=False,
        index=records.index,
    )
    for otr_cusip in otr_cusips:
        has_otr |= records['cusip1'] == otr_cusip
    selected = has_otr | has_query_cusip
    result = records.loc[selected]
    assert len(result) >= len(cusip_records)
    assert len(result) <= len(records)
    return result


def sort_by_effectivedatetime(df):
    'return new DataFrame in sorted order'
    # Note: the following statement works but generates a SettingWithCopyWarning
    result = df.sort_values('effectivedatetime')
    return result


def analyze_trace_prints(trace_prints, control):
    'print to stdout'
    def print_trade(issuepriceid):
        print issuepriceid, trace_prints.loc[issuepriceid]

    print 'there are %d trace_prints' % len(trace_prints)
    cusips_set = set(trace_prints['cusip'])
    print 'there are %d cusips' % len(cusips_set)
    print 'cusip -> n_prints -> first_dateimte -> last_datetime'
    for cusip in sorted(cusips_set):
        trace_prints_cusip = trace_prints.loc[trace_prints['cusip'] == cusip]
        effectivedatetimes = trace_prints_cusip['effectivedatetime']
        print cusip, len(trace_prints_cusip), min(effectivedatetimes), max(effectivedatetimes)


def read_trace_prints_underlying_file(control):
    'return DataFrame'
    trace_prints = seven.read_csv.input(
        issuer=control.arg.issuer,
        logical_name='trace',
        nrows=1000 if False and control.arg.test else None,
    )
    control.timer.lap('stop read_trace_prints_underlying_file')
    return trace_prints


def transform_trace_prints(trace_prints, control):
    'return (DataFrame of relevant and transformed trades, set of otr_cusips)'
    def select_relevant_dates(df):
        'return DataFrame with effective not at least control.first_relevant_trace_print_date'
        mask = df['effectivedatetime'] >= control.first_relevant_trace_print_date
        result = df.loc[mask]
        control.timer.lap('select_relevant_dates')
        return result

    def make_otr_cusips(df):
        'return Dict[trade_date, otr_cusip]'
        # return dict[effective_date, otr_cusip]
        # input file format to change to have 3 columns
        otr = seven.read_csv.input(
            issuer=control.arg.issuer,
            logical_name='otr',
        )
        otr_cusip = otr.loc[otr['primary_cusip'] == control.arg.cusip]
        result = {}
        for timestamp, record in otr_cusip.iterrows():
            result[timestamp.date()] = record['otr_cusip']
        control.timer.lap('make_otr_cusips')
        return result

    def select_relevant_cusips(df, otr_cusips):
        'return DataFrame with just the primary and OTR cusips'
        # starting on the date column, the otr_cusip is as specified
        # before the first date, the otr_cusip is the primary_cusip
        mask = df['cusip'] == control.arg.cusip
        distinct_otr_cusips = set(otr_cusips.values())
        for otr_cusip in distinct_otr_cusips:
            mask_otr_cusip = df['cusip'] == otr_cusip
            if sum(mask_otr_cusip) == 0:
                # NOTE: the OTR cusips and trace prints come from different files
                # these files may not match
                print 'otr cusip %s is not in the trace print file' % otr_cusip
                for trade_date, otr_cusip1 in otr_cusips.iteritems():
                    if otr_cusip1 == otr_cusip:
                        print 'otr cusip %s was for date %s' % (otr_cusip1, trade_date)
                        mask_trade_date = df['effectivedate'] == trade_date
                        if sum(mask_trade_date) == 0:
                            print ' no trace print occured on that date'
                            # we don't have a file construction problem
                        else:
                            print 'trace_prints for that date'
                            print 'trace_index -> effectivedate -> cusip'
                            for trace_index, row in df.loc[mask_trade_date].iterrows():
                                print trace_index, row['effectivedatetime'], row['cusip']
                            seven.logging.error(
                                'trace print and liq flow on the run files do not match',
                                'the on the run file specified OTR cusip %s' % otr_cusip,
                                'the trace print file does not contain a trade with the cusip',
                            )
            else:
                mask |= mask_otr_cusip
        result = df.loc[mask]
        control.timer.lap('select_relevant_cusips')
        return result

    def select_valid_oasspreads(df, verbose=True):
        'return df where the oasspread is not NaN'
        def print_cusip_count(df):
            for cusip in sorted(set(df['cusip'])):
                print cusip, sum(df['cusip'] == cusip)

        if verbose:
            print 'incoming cusip -> count'
            print_cusip_count(df)

        mask = np.isnan(df['oasspread'])
        print 'ignoring %d trace print that have NaN values' % sum(mask)
        result = df.loc[~mask]
        if verbose:
            print 'outgoing cusip -> count'
            print_cusip_count(result)
        control.timer.lap('select_valid_oasspread')
        return result

    def add_derived_fields(trace_prints):
        'mutate trace_prints'
        trace_prints['effectivedatetime'] = seven.feature_makers.make_effectivedatetime(trace_prints)
        trace_prints['effectiveyear'] = pd.DatetimeIndex(trace_prints['effectivedatetime']).year
        trace_prints['effectivemonth'] = pd.DatetimeIndex(trace_prints['effectivedatetime']).month
        trace_prints['effectiveday'] = pd.DatetimeIndex(trace_prints['effectivedatetime']).day
        trace_prints['issuepriceid'] = trace_prints.index
        control.timer.lap('add_derived_fields')

    def elapsed():
        return control.timer.elapsed_wallclock_seconds()

    start_wallclock = elapsed()
    add_derived_fields(trace_prints)
    if control.arg.analyze_trace:
        analyze_trace_prints(trace_prints, control)
    trace_prints_relevant_dates = select_relevant_dates(trace_prints)
    otr_cusips = make_otr_cusips(trace_prints_relevant_dates)
    distinct_otr_cusips = set(otr_cusips.values())
    print 'there are %d distinct otr cusips for cusip %s' % (len(distinct_otr_cusips), control.arg.cusip)
    trace_prints_for_cusips = select_relevant_cusips(trace_prints_relevant_dates, otr_cusips)
    ok_oasspreads = select_valid_oasspreads(trace_prints_for_cusips)
    sorted_trace_prints = ok_oasspreads.sort_values('effectivedatetime')
    print 'read %d trace_prints' % len(trace_prints)
    print 'of which, %d were on relevant dates' % len(trace_prints_relevant_dates)
    print 'of which, %d were for the specified CUSIP or its on-the-run CUSIPS' % len(trace_prints_for_cusips)
    print 'of which, %d had OK oasspread values' % len(ok_oasspreads)
    print '# cusips in reduced file:', len(otr_cusips)
    print 'transform_trace_prints took %f wallclock seconds' % (elapsed() - start_wallclock)
    return sorted_trace_prints, otr_cusips


def read_and_transform_trace_prints(control):
    'return (DataFrame, otr_cusips)'
    # use a cache to reduce wall clock time for the reading
    def same_invocation_parameters(arg1, arg2):
        return (
            arg1.cusip == arg2.cusip and
            arg1.effective_date == arg2.effective_date and
            arg1.test == arg2.test
        )

    # def modification_time():
    #     return os.path.getmtime(control.path['in_trace'])

    def write_cache(control_, trace_prints_, otr_cusips_):
        'write args to pickle file'
        obj = (control_, trace_prints_, otr_cusips_)
        with open(control.path['optional_out_cache'], 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        control.timer.lap('write_cache')

    def read_cache():
        'return (control, trace_prints, otr_cusips) from pickle file'
        with open(control.path['optional_out_cache'], 'rb') as f:
            control_, trace_prints_, otr_cusips_ = pickle.load(f)
        control.timer.lap('read_cache')
        return control_, trace_prints_, otr_cusips_

    def read_and_transform_underlying_file():
        'return (DataFrame, otr_cusip)'
        df1 = read_trace_prints_underlying_file(control)
        df2, otr_cusips = transform_trace_prints(df1, control)
        return df2, otr_cusips

    def elapsed():
        return control.timer.elapsed_wallclock_seconds()

    start_wallclock = elapsed()
    if control.arg.cache:
        if os.path.isfile(control.path['optional_out_cache']):
            # cache exists
            # if it was created with the same invocation parameters, use it
            # otherwise, re-create it
            cached_control, cached_trace_prints, cached_otr_cusips = read_cache()
            if same_invocation_parameters(cached_control.arg, control.arg):
                # the cache file was built with same parameters as the current invocation
                # so we can use it
                trace_prints = cached_trace_prints
                otr_cusips = cached_otr_cusips
            else:
                trace_prints, otr_cusips = read_and_transform_underlying_file()
                write_cache(control, trace_prints, otr_cusips)
        else:
            # cache does not exist
            # create it and save the current invocation parameters
            trace_prints, otr_cusips = read_and_transform_underlying_file()
            write_cache(control, trace_prints, otr_cusips)
    else:
        trace_prints, otr_cusips = read_and_transform_underlying_file()
    print 'read_and_transform_trace_prints took %f wallclock seconds' % (elapsed() - start_wallclock)
    return trace_prints, otr_cusips


def accumulate(feature_maker, event):
    pdb.set_trace()
    errs = feature_maker.accumulate(event)
    if len(errs) > 0:
        for err in errs:
            print 'accumulate error', feature_maker, err


class Event(object):
    def __init__(self, id, payload):
        assert isinstance(id, seven.EventId.EventId)
        assert isinstance(payload, dict)
        self.id = id
        self.payload = payload

    def __eq__(self, other):
        return self.id == other.id

    def __ge__(self, other):
        return self.id >= other.id

    def __gt__(self, other):
        return self.id > other.id

    def __le__(self, other):
        return self.id <= other.id

    def __lt__(self, other):
        return self.id < other.id

    def __repr__(self):
        return 'Event(%s, %d values)' % (self.id, len(self.payload))


class EventReader(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def next(self):
        'return Event or raise StopIteration()'

    @abc.abstractmethod
    def close(self):
        'close any underlying file'


class OtrCusipEventReader(EventReader):
    def __init__(self, issuer, cusip, test=False):
        self.issuer = issuer
        self.test = test

        path = seven.path.input(issuer, 'otr')
        self.file = open(path)
        self.dict_reader = csv.DictReader(self.file)
        self.prior_datetime = datetime.datetime(datetime.MINYEAR, 1, 1)

    def __iter__(self):
        return self

    def close(self):
        self.file.close()

    def next(self):
        try:
            row = self.dict_reader.next()
        except StopIteration:
            raise StopIteration()
        event_id = seven.EventId.OtrCusipEventId(
            row['date'],
            self.issuer,
        )
        assert event_id.datetime() >= self.prior_datetime
        self.prior_datetime = event_id.datetime()
        return Event(
            id=event_id,
            payload=row,
        )


class TotalDebtEventReader(EventReader):
    def __init__(self, issuer, cusip, test=False):
        self.issuer = issuer
        self.test = test

        path = seven.path.input(issuer, 'total_debt')
        self.file = open(path)
        self.dict_reader = csv.DictReader(self.file)
        self.prior_datetime = datetime.datetime(datetime.MINYEAR, 1, 1)

    def __iter__(self):
        return self

    def close(self):
        self.file.close()

    def next(self):
        try:
            row = self.dict_reader.next()
        except StopIteration:
            raise StopIteration()
        event_id = seven.EventId.TotalDebtEventId(
            row['date'],
            self.issuer,
        )
        assert event_id.datetime() >= self.prior_datetime
        self.prior_datetime = event_id.datetime()
        return Event(
            id=event_id,
            payload=row,
        )


class TraceEventReader(EventReader):
    def __init__(self, issuer, cusip, test=False):
        self.issuer = issuer
        self.test = test

        path = seven.path.input(issuer, 'trace')
        self.file = open(path)
        self.dict_reader = csv.DictReader(self.file)
        self.prior_datetime = datetime.datetime(datetime.MINYEAR, 1, 1,)

    def __iter__(self):
        return self

    def close(self):
        self.file.close()

    def next(self):
        try:
            row = self.dict_reader.next()
        except StopIteration:
            raise StopIteration()
        event_id = seven.EventId.TraceEventId(
            effective_date=row['effectivedate'],
            effective_time=row['effectivetime'],
            issuer=self.issuer,
            issuepriceid=row['issuepriceid'],
        )
        assert event_id.datetime() >= self.prior_datetime
        self.prior_datetime = event_id.datetime()
        return Event(
            id=event_id,
            payload=row,
        )


class EventQueue(object):
    'maintain event timestamp-ordered queue of event'
    def __init__(self, event_reader_classes, issuer, cusip):
        self.event_readers = []
        self.event_queue = []  # : heapq
        for event_reader_class in event_reader_classes:
            event_reader = event_reader_class(issuer, cusip)
            try:
                event = event_reader.next()
            except StopIteration:
                seven.logging.critical('event reader %s returned no events at all' % event_reader)
                sys.exit(1)
            heapq.heappush(self.event_queue, (event, event_reader))
        print 'initial event queue'
        q = copy.copy(self.event_queue)
        while True:
            try:
                event, event_reader = heapq.heappop(q)
                print event
            except IndexError:
                # q is empty
                break

    def next(self):
        'return the oldest event or raise StopIteration()'
        # get the oldest event (and delete it from the queue)
        try:
            oldest_event, event_reader = heapq.heappop(self.event_queue)
        except IndexError:
            raise StopIteration()

        # replace the oldest event with the next event from the same event reader
        try:
            new_event = event_reader.next()
            heapq.heappush(self.event_queue, (new_event, event_reader))
        except StopIteration:
            event_reader.close()

        return oldest_event

    def close(self):
        'close each event reader'
        pdb.set_trace()
        for event_reader in self.event_reader():
            event_reader.close()


def test_event_readers(event_reader_classes, control):
    # test the readers
    # NOTE: this disrupts their state, so we exit at the end
    # test TotalDebtEventReader
    def read_and_print_all(event_reader):
        count = 0
        while (True):
            try:
                event = event_reader.next()
                count += 1
            except StopIteration:
                break
            print event.id
        print 'above are %d EventIds for %s' % (count, type(event_reader))
        pdb.set_trace()

    for event_reader_class in event_reader_classes:
        er = event_reader_class(control.arg.issuer, control.arg.cusip, control.arg.test)
        read_and_print_all(er)


def do_work(control):
    'write predictions from fitted models to file system'
    event_reader_classes = (
        OtrCusipEventReader,
        TotalDebtEventReader,
        TraceEventReader,
    )

    if False:
        test_event_readers(event_reader_classes, control)

    # prime the event streams by read a record from each of them
    event_queue = EventQueue(event_reader_classes, control.arg.issuer, control.arg.cusip)

    # repeatedly process the youngest event
    counter = collections.Counter()
    all_feature_makers = []
    trace_feature_makers = {}          # Dict[cusip, feature_makers2.Trace]
    last_created_features = {}         # Dict[EventId, dict]
    last_created_features_cusips = {}  # Dict[cusip, Dict[EventId, dict]]
    while True:
        try:
            event = event_queue.next()
        except StopIteration:
            break  # all the event readers are empty

        print 'next event', event

        # handle the event
        if isinstance(event.id, seven.EventId.TraceEventId):
            # build features for all cusips because we don't know which will be the OTR cusips
            # until we see OTR events
            cusip = event.payload['cusip']
            if cusip not in trace_feature_makers:
                trace_feature_maker = seven.feature_makers2.Trace(control.arg.issuer, control.arg.cusip)
                all_feature_makers.append(trace_feature_maker)
                trace_feature_makers[cusip] = trace_feature_maker
                last_created_features_cusips[cusip] = {}
            features, errs = trace_feature_makers[cusip].make_features(event.id, event.payload)
            assert (isinstance(features, dict) and errs is None) or (features is None and isinstance(errs, list))
            if errs is None:
                last_created_features_cusips[cusip][event.id] = features
        else:
            errs = []
            print 'not yet handled', event.id
            last_created_features[event.id] = {}  # TODO: replace with actual features
            pdb.set_trace()
        counter['events created'] += 1
        # TODO: create features unless there were errs
        if errs is not None:
            for err in errs:
                seven.logging.warning('feature_maker err eventId %s: %s' % (event.id, err))
            if not isinstance(event.id, seven.EventId.TraceEventId):
                pdb.set_trace()
            continue
        pdb.set_trace()
        counter['features created'] += 1
        # TODO: test (predict) if we have features
        counter['predictions made'] += 1
        # TODO: fit, if certain conditions hold
        counter['fits completed'] += 1
    pdb.set_trace()
    # TODO: clean up
    event_queue.close()
    pp(counter)
    pdb.set_trace()
    return None

    # OLD BELOW ME
    def lap():
        'return elapsed wall clock time:float since previous call to lap()'
        return control.timer.lap('lap', verbose=False)[1]

    def print_info(selected_date):
        mask_date_cusip = (trace_prints['effectivedate'] == selected_date) & (trace_prints['cusip'] == control.arg.cusip)
        n_predictions = sum(mask_date_cusip)
        print 'found %d trace prints on selected date %s with selected cusip %s' % (
            n_predictions,
            selected_date,
            control.arg.cusip,
        )
        print 'found %d distinct effective date times' % len(set(trace_prints['effectivedatetime']))
        print 'cusips in file:', set(trace_prints['cusip'])
        print 'prepared input in %s wall clock seconds' % lap()
        return n_predictions

    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    control.timer.lap('prelude to start do_work')
    trace_prints, otr_cusips = read_and_transform_trace_prints(control)
    all_distinct_cusips = set(trace_prints['cusip'])
    print 'found %d trace prints for the cusip %s and its related OTR cusips' % (len(trace_prints), control.arg.cusip)
    print 'will create features and targets for these cusips:', all_distinct_cusips

    # iterate over each relevant row
    # build and save the features for the cusip
    # the issuer may vary by cusip
    features_accumulator = {}
    for cusip in all_distinct_cusips:
        features_accumulator[cusip] = seven.accumulators.FeaturesAccumulator(
            issuer=control.arg.issuer,
            cusip=cusip,
        )
    targets_accumulator = seven.accumulators.TargetsAccumulator()

    info = collections.Counter()
    warning = collections.Counter()

    selected_date = Date(from_yyyy_mm_dd=control.arg.effective_date).value  # a datetime.date
    print_info(selected_date)

    print 'cusip -> # trace prints'
    for cusip in sorted(set(trace_prints['cusip'])):
        print cusip, sum(trace_prints['cusip'] == cusip)
    print 'invocation cusip', control.arg.cusip

    for trace_index, trace_record in trace_prints.iterrows():
        # Note: the CUSIP for each record is control.arg.cusip or a related OTR cusip
        info['n trace records seen'] += 1
        if info['n trace records seen'] % 100 == 1:
            print 'features_targets.py %s %s %s: processing trace record %d of %d' % (
                control.arg.issuer,
                control.arg.cusip,
                control.arg.effective_date,
                info['n trace records seen'],
                len(trace_prints),
            )

        # stop once we go past the selected date
        trace_record_date = trace_record['effectivedate'].date()  # type is datetime.date
        if trace_record_date > selected_date:
            break  # stop, since the trace_prints are in non-decreasing order by effectivedatetime
        info['n_trace_records processed']
        if trace_record['cusip'] == control.arg.cusip:
            info['n_trace records processed for query cusip'] += 1

        errs = features_accumulator[trace_record['cusip']].accumulate(trace_index, trace_record)
        if errs is not None:  # errs is Union(None, List[str])
            info['had feature error(s)'] += 1
            if trace_record['cusip'] == control.arg.cusip:
                info['had feature error(s) for query cusip'] += 1
            for err in errs:
                warning['feature accumulator:' + err] += 1
            continue

        if trace_record['cusip'] == control.arg.cusip:
            info['target accumulations attempted'] += 1
            errs = targets_accumulator.accumulate(trace_index, trace_record)
            if errs is not None:
                info['had target error(s)'] += 1
                if trace_record['cusip'] == control.arg.cusip:
                    info['had target error(s) for query cusip'] += 1
                for err in errs:
                    warning['target_accumulator: ' + err] += 1
                continue

        info['feature and target records created'] += 1
        info['features and targets created for cusip %s on date %s' % (
            trace_record['cusip'],
            trace_record_date,
        )] += 1

        on_selected_date = trace_record_date == selected_date
        if on_selected_date and trace_record['cusip'] == control.arg.cusip:
            info['features and targets created for query cusip and date'] += 1
            if control.arg.test:
                if info['feature and targets created for query cusip and date'] > 10:
                    print 'DEBUG CODE: discard output'
                    break

        # try to keep memory usage roughly constant
        # that enables running parallel instances on a single system
        gc.collect()

    control.timer.lap('create features from all relevant trace print records')
    print 'summary across all trace print records examined'
    print 'infos'
    for k in sorted(info.keys()):
        print '%71s: %6d' % (k, info[k])

    print 'warnings'
    for k in sorted(warning.keys()):
        print '%71s: %6d' % (k, warning[k])

    print 'cusip -> # features'
    for k, v in features_accumulator.iteritems():
        print k, len(v.accumulated)

    def create_empty_outputs():
        # previously we needed to write an empty file
        # that is no longer necessary
        pass
        # print 'creating empty output file'
        # pd.DataFrame().to_csv(control.path['out_features'])

    if info['features and targets created for query cusip and date'] == 0:
        print 'create no features for the primary custip %s' % control.arg.cusip
        create_empty_outputs()
        sys.exit(0)  # don't exit with an error code, as that would stop scons

    # select features for the primary cusip on the query date (selected_date)
    all_primary_cusip_features = features_accumulator[control.arg.cusip].accumulated
    mask = all_primary_cusip_features['id_effectivedate'] == selected_date
    primary_cusip_features = all_primary_cusip_features.loc[mask]
    print 'primary_cusip_features on the selected date', len(primary_cusip_features)
    if len(primary_cusip_features) == 0:
        print 'no features for the primary cusip'
        create_empty_outputs()
        sys.exit(0)

    # build up the features of the primary cusip and all the OTR cusips
    merged_dataframe = pd.DataFrame()
    merge_info_counter = collections.Counter()

    def merge_info(err):
        print err
        print 'skipping creation of the merged feature set'
        merge_info_counter['info: ' + err] += 1
        merge_info['n merged feature records skipped'] += 1

    for index, primary_cusip_features in primary_cusip_features.iterrows():
        trade_date = primary_cusip_features['id_effectivedate']  # a datetime.date
        if trade_date not in otr_cusips:
            merge_info('no OTR cusip for trade date %s (primary cusip %s index %s)' % (
                    trade_date,
                    control.arg.cusip,
                    index,
            ))
            continue
        otr_cusip = otr_cusips[trade_date]
        otr_cusip_features = features_accumulator[otr_cusip].accumulated  # a DataFrame
        if len(otr_cusip_features) == 0:
            merge_info('no features created yet for OTR cusip %s (primary cusip %s index %s)' % (
                otr_cusip,
                control.arg.cusip,
                index,
            ))
            continue
        # find the earlist prior otr cusip trade
        time_mask = otr_cusip_features['id_effectivedatetime'] < primary_cusip_features['id_effectivedatetime']
        before = otr_cusip_features.loc[time_mask]
        if len(before) == 0:
            merge_info('no otr cusip trace prints before the query trace print %s index %s' % (
                control.arg.cusip,
                index,
            ))
            continue
        else:
            just_before = before.sort_values(by='id_effectivedatetime').iloc[-1]  # a series
            features = {}
            for k, v in primary_cusip_features.iteritems():
                new_primary_feature_name = 'id_p_' + k[3:] if k.startswith('id_') else 'p_' + k
                features[new_primary_feature_name] = v
            for k, v in just_before.iteritems():
                new_otr_feature_name = 'id_otr1_' + k[3:] if k.startswith('id_') else 'otr1_' + k
                features[new_otr_feature_name] = v
            new_row = pd.DataFrame(
                data=features,
                index=pd.Series(
                    data=[index],
                    name='issuepriceid',
                    ))
            merged_dataframe = merged_dataframe.append(new_row)

    # write the merge info
    print 'info about the merging of the primary and OTR cusip features'
    print 'each info resulted in a record being skipped'
    if len(merge_info_counter) == 0:
        print '** nothing skipped in the merge **'
    else:
        for k in sorted(merge_info_counter.keys()):
            print '%71s: %6d' % (k, merge_info_counter[k])

    # write each feature records to a seperate file
    for row_index in xrange(len(merged_dataframe)):
        row_df = merged_dataframe.iloc[[row_index]]
        row_series = row_df.iloc[0]
        date = row_series['id_p_effectivedate']
        time = row_series['id_p_effectivetime']
        issuepriceid = row_series['id_p_issuepriceid']
        filename = '%s-%02d-%02d-%02d-traceprint-%s.%s.csv' % (
            date,
            time.hour,
            time.minute,
            time.second,
            issuepriceid,
            row_series['id_p_reclassified_trade_type'],
            )
        path_out = os.path.join(control.path['dir_out'], filename)
        row_df.to_csv(path_out)
        print 'wrote', path_out

    return None


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    # print control
    print control.arg
    print 'done'
    return


if __name__ == '__main__':
    main(sys.argv)
