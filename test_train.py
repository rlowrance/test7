'''create predictions and test their accuracy

APPROACH: see the file test_traing.org in the same directory as this file.

INVOCATION
  python test_train.py {issuer} {cusip} {target} {hpset} {start_date} {--debug} {--test} {--trace}
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
import math
import pdb
from pprint import pprint
import random
import sys

import applied_data_science.debug
import applied_data_science.dirutility
import applied_data_science.lower_priority
import applied_data_science.pickle_utilities

from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.accumulators
import seven.arg_type
import seven.build
import seven.EventId
import seven.feature_makers2
import seven.fit_predict_output
import seven.HpGrids
import seven.logging
import seven.models2
import seven.read_csv

pp = pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('issuer', type=seven.arg_type.issuer)
    parser.add_argument('cusip', type=seven.arg_type.cusip)
    parser.add_argument('target', type=seven.arg_type.target)
    parser.add_argument('hpset', type=seven.arg_type.hpset)
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

    paths = seven.build.test_train(arg.issuer, arg.cusip, arg.target, arg.hpset, arg.start_date, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    timer = Timer()

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=timer,
    )


class FeatureVector(object):
    def __init__(self,
                 creation_event,
                 primary_cusip_event_features,
                 otr_cusip_event_features,
                 issuer_event_features):
        assert isinstance(creation_event, Event)
        assert isinstance(creation_event.id, seven.EventId.TraceEventId)
        assert isinstance(primary_cusip_event_features, dict)
        assert isinstance(otr_cusip_event_features, dict)
        assert isinstance(issuer_event_features, dict)

        # these fields are part of the API
        self.creation_event = copy.copy(creation_event)
        self.reclassified_trade_type = creation_event.payload['reclassified_trade_type']
        self.payload = self._make_payload(
            primary_cusip_event_features,
            otr_cusip_event_features,
            issuer_event_features,
        )
        # TODO: implement cross-product of the features from the events
        # for example, determine the debt to equity ratio

    def _make_payload(self, primary_cusip_event_features, otr_cusip_event_features, issuer_event_features):
        'return dict with all the features'
        # check that there are no duplicate feature names
        def append(all_features, tag, new_features):
            for k, v in new_features.iteritems():
                if k.startswith('id_'):
                    k_new = 'id_%s_%s' % (tag, k[3:])
                else:
                    k_new = '%s_%s' % (tag, k)
                assert k_new not in all_features
                all_features[k_new] = v

        all_features = {}
        append(all_features, 'p', primary_cusip_event_features)
        append(all_features, 'otr', otr_cusip_event_features)
        append(all_features, 'issuer', issuer_event_features)
        return all_features

    def __repr__(self):
        return 'FeatureVector(creation_event=%s, rct=%s, n features=%d)' % (
            self.creation_event,
            self.reclassified_trade_type,
            len(self.payload),
        )


def maybe_create_feature_vector(control, event, event_feature_makers):
    'return (feature_vector, errs)'
    if event.is_trace_print_with_cusip(control.arg.cusip):
        debug = event.id.datetime() > datetime.datetime(2017, 6, 5, 12, 2, 0)
        debug = False
        if debug:
            pdb.set_trace()
        # make sur that we know all the fields in the primary cusip
        primary_cusip = event.maybe_cusip()
        primary_cusip_event_features = event_feature_makers.cusip_event_features_dict[primary_cusip]
        primary_missing_fields = primary_cusip_event_features.missing_field_names()
        if len(primary_missing_fields) > 0:
            errs = []
            for primary_missing_field in primary_missing_fields:
                errs.append('missing primary field %s' % primary_missing_field)
            return None, errs
        # make sure that we have the trace event features from the OTR cusip
        otr_cusip = primary_cusip_event_features.last_otr_cusip
        otr_cusip_event_features = event_feature_makers.cusip_event_features_dict[otr_cusip]
        if otr_cusip_event_features.last_trace_event_features is None:
            err = 'missing otr last trace event features'
            return None, [err]
        # make sure we have all the features from the issuer
        issuer_missing_fields = event_feature_makers.issuer_event_features.missing_field_names()
        if len(issuer_missing_fields) > 0:
            errs = []
            for issuer_event_field_name in issuer_missing_fields:
                pdb.set_trace()
                err = 'missing issuer field %s' % issuer_event_field_name
                errs.append(err)
            return None, errs
        # have all the fields, so construct the feature vector
        if debug:
            print 'about to create feature vector'
            pdb.set_trace()
        feature_vector = FeatureVector(
            creation_event=event,
            primary_cusip_event_features=primary_cusip_event_features.last_trace_event_features,
            otr_cusip_event_features=otr_cusip_event_features.last_trace_event_features,
            issuer_event_features=event_feature_makers.issuer_event_features.last_total_debt_event_features,
        )
        return feature_vector, None
    else:
        err = 'event source %s, not trace print with query cusip' % event.id.source
        return None, [err]


class TargetVector(object):
    def __init__(self, event, target_name, target_value):
        assert isinstance(event, Event)
        assert isinstance(target_value, float)
        self.creation_event = copy.copy(event)
        self.target_name = target_name
        self.payload = target_value

    def __repr__(self):
        return 'TargetVector(creation_event.id=%s, %s, payload=%f)' % (
            self.creation_event.id,
            self.target_name,
            self.payload,
        )


class ActionSignal(object):
    def __init__(self, path_actions, path_signal):
        self.actions = Actions(path_actions)
        self.signal = Signal(path_signal)

    def actual(self, trade_type, event, target_value):
        assert trade_type in ('B', 'S')
        assert isinstance(event, Event)
        assert isinstance(target_value, float)
        self.signal.actual(trade_type, event, target_value)

    def ensemble_prediction(self, trade_type, event, predicted_value, standard_deviation):
        assert trade_type in ('B', 'S')
        assert isinstance(event, Event)
        assert isinstance(predicted_value, float)
        assert isinstance(standard_deviation, float)
        self.actions.actions(
            event=event,
            types_values=[
                ('ensemble_prediction_oasspread_%s' % trade_type, predicted_value),
                ('experts_standard_deviation_%s' % trade_type, standard_deviation),
            ],
        )
        self.signal.prediction(trade_type, event, predicted_value, standard_deviation)

    def close(self):
        self.actions.close()
        self.signal.close()


class Actions(object):
    'produce the action file'
    def __init__(self, path):
        self.path = path
        self.file = open(path, 'wb')
        self.field_names = ['action_id', 'action_type', 'action_value']
        self.dict_writer = csv.DictWriter(self.file, self.field_names, lineterminator='\n')
        self.dict_writer.writeheader()
        self.last_event_id = ''
        self.last_action_suffix = 1

    def action(self, event, action_type, action_value):
        'append the action, creating a unique id from the event.id'
        return self.actions(event, [(action_type, action_value)])

    def actions(self, event, types_values):
        action_id = self._next_action_id(event)
        for action_type, action_value in types_values:
            self.dict_writer.writerow({
                'action_id': action_id,
                'action_type': action_type,
                'action_value': action_value,
            })

    def close(self):
        self.file.close()

    def _next_action_id(self, event):
        event_id = str(event.id)
        if event_id == self.last_event_id:
            self.last_action_id_suffix += 1
        else:
            self.last_action_id_suffix = 1
        self.last_event_id = event_id
        action_id = 'ml_%s_%d' % (event_id, self.last_action_id_suffix)
        return action_id


class Signal(object):
    'produce the signal as a CSV file'
    def __init__(self, path):
        self.path = path
        self.file = open(path, 'wb')
        self.field_names = [
            'datetime', 'event_source', 'event_source_id',   # these columns are dense
            'prediction_B', 'prediction_S',                  # thse columns are sparse
            'standard_deviation_B', 'standard_deviation_S',  # these columns are sparse
            'actual_B', 'actual_S',                          # these columns are sparse
            ]
        self.dict_writer = csv.DictWriter(self.file, self.field_names, lineterminator='\n')
        self.dict_writer.writeheader()

    def _event(self, event):
        'return dict for the event columns'
        print '_event', event
        event_source_id = (
            '%s (%s)' % (event.id.source_id, event.payload['cusip']) if event.id.source.startswith('trace_') else
            event.id.source_id
        )

        return {
            'datetime': event.id.datetime(),
            'event_source': event.id.source,
            'event_source_id': event_source_id,
        }

    def actual(self, trade_type, event, target_value):
        d = self._event(event)
        d.update({
            'actual_%s' % trade_type: target_value,
        })
        self.dict_writer.writerow(d)

    def predictions(self, trade_type, event, predicted_value, standard_deviation):
        d = self._event(event)
        d.update({
            'prediction_%s' % trade_type: predicted_value,
            'standard_deviation_%s' % trade_type: standard_deviation,
        })
        self.dict_writer.writerow(d)

    def close(self):
        self.file.close()


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
        if self.id.source.startswith('trace_'):
            return 'Event(%s, %d values, cusip %s, rct %s)' % (
                self.id,
                len(self.payload),
                self.payload['cusip'],
                self.payload['reclassified_trade_type'],
            )
        else:
            return 'Event(%s, %d values)' % (
                self.id,
                len(self.payload),
            )

    def is_trace_print(self):
        'return True or False'
        return self.id.source.startswith('trace_')

    def maybe_cusip(self):
        'return cusip (if this event is a trace print) else None'
        if self.is_trace_print():
            return self.payload['cusip']
        else:
            return None

    def is_trace_print_with_cusip(self, cusip):
        'return True or False'
        if self.is_trace_print():
            return self.maybe_cusip() == cusip
        else:
            return False


class EventReader(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def close(self):
        'close any underlying file'

    @abc.abstractmethod
    def next(self):
        'return Event or raise StopIteration()'


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

        # path = seven.path.input(issuer, 'trace')
        path = seven.path.sort_trace_file(issuer)
        self.file = open(path)
        self.dict_reader = csv.DictReader(self.file)
        self.prior_datetime = datetime.datetime(datetime.MINYEAR, 1, 1,)
        self.records_read = 0

    def __iter__(self):
        return self

    def close(self):
        self.file.close()

    def next(self):
        try:
            row = self.dict_reader.next()
            self.records_read += 1
            if row['trade_type'] == 'D':
                if row['reclassified_trade_type'] == 'D':
                    print row['reclassified_trade_type']
                    pdb.set_trace()
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
            self.event_readers.append(event_reader)
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
        for event_reader in self.event_readers:
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


def oops(reason, msg, event):
    assert isinstance(reason, str)
    assert isinstance(msg, str)
    assert isinstance(event, Event)

    seven.logging.info('event %s: %s: %s' % (event.id, reason, msg))


def event_not_usable(errs, event):
    for err in errs:
        oops('event not usable', err, event)


def no_accuracy(msg, event):
    oops('unable to determine prediction accuracy', msg, event)


def no_feature_vector(msg, event):
    oops('feature vector not created', msg, event)


def no_prediction(msg, event):
    oops('no prediction has been created', msg, event)


def no_training(msg, event):
    oops('no training was done', msg, event)


def make_max_n_trades_back(hpset):
    'return the maximum number of historic trades a model uses'
    grid = seven.HpGrids.construct_HpGridN(hpset)
    max_n_trades_back = 0
    for model_spec in grid.iter_model_specs():
        if model_spec.n_trades_back is not None:  # the naive model does not have n_trades_back
            max_n_trades_back = max(max_n_trades_back, model_spec.n_trades_back)
    return max_n_trades_back


def make_training_data(target_name, feature_vectors):
    'return (List[training_feature_vector], List[target_vector]'
    # the feature vectors all have the same reclassified trade type
    assert target_name == 'oasspread'
    reclassified_trade_type = None

    # the targets have the oasspread of the next trade
    target_vectors = []
    for i in range(0, len(feature_vectors) - 1):
        # check that all the feature vectors have the same reclassified trade type
        if reclassified_trade_type is None:
            reclassified_trade_type = feature_vectors[i].reclassified_trade_type()
        assert feature_vectors[i].reclassified_trade_type() == reclassified_trade_type
        target_value = float(feature_vectors[i + 1].creation_event.payload[target_name])
        target_vector = TargetVector(
            feature_vectors[i + 1].creation_event,
            target_name,
            target_value,
            )
        target_vectors.append(target_vector)
    result = (feature_vectors[:-1], target_vectors)
    if len(result[0]) != len(result[1]):
        seven.logging.critical('internal error: make_training_data')
        sys.exit(1)
    return feature_vectors[:-1], target_vectors


def make_trained_expert_models(control, training_feature_vectors, training_target_vectors, verbose=False):
    'return Dict[model_spec, fitted_model]'
    assert control.arg.target == 'oasspread'

    grid = seven.HpGrids.construct_HpGridN(control.arg.hpset)
    result = {}  # Dict[model_spec, trained model]
    for model_spec in grid.iter_model_specs():
        model_constructor = (
            seven.models2.ModelNaive if model_spec.name == 'n' else
            seven.models2.ModelElasticNet if model_spec.name == 'en' else
            seven.models2.ModelRandomForests if model_spec.name == 'rf' else
            None
        )
        model = model_constructor(model_spec, control.random_seed)
        try:
            if verbose:
                print 'fitting', model.model_spec
            model.fit(training_feature_vectors, training_target_vectors)
        except seven.models2.ExceptionFit as e:
            seven.logging.warning('could not fit %s: %s' % (model_spec, e))
            continue
        result[model_spec] = model  # the fitted model
    return result


class EnsembleHyperparameters(object):
    def __init__(self,
                 max_n_expert_predictions_saved=100,
                 weight_temperature=1.0,
                 weight_units='days',
                 ):
        self.max_n_expert_predictions_saved = max_n_expert_predictions_saved
        self.weight_temperature = weight_temperature
        self.weight_units = weight_units  # pairs with weight_temperature

    def __repr__(self):
        return 'EnsembleHyperparameters(%f, %f)' % (
            self.max_n_experts_saved,
            self.weight_temperatures,
        )


class EventFeatures(object):
    __metaclass__ = abc.ABCMeta

    def missing_field_names(self):
        'return list of names of missing fields (these have value None)'
        result = []
        for k, v in self.__dict__.iteritems():
            if v is None:
                result.append(k)
        return result

    def _make_repr(self, class_name):
        'return str'
        args = None
        for k, v in self.__dict__.iteritems():
            arg = '%s=%r' % (k, v)
            if args is None:
                args = arg
            else:
                args += ', %s' % arg
        return '%s(%s)' % (class_name, args)

    def _make_str(self, class_name):
        missing_field_names = self.missing_field_names()
        if len(missing_field_names) == 0:
            return '%s(complete)' % class_name
        else:
            missing_fields = [missing_field for missing_field in missing_field_names]
            return '%s(missing %s)' % (class_name, missing_fields)


class CusipEventFeatures(EventFeatures):
    def __init__(self, last_trace_event_features=None, last_otr_cusip=None):
        self.last_trace_event_features = last_trace_event_features  # dict[feature_name, feature_value]
        self.last_otr_cusip = last_otr_cusip                        # str

    def __repr__(self):
        return self._make_repr('CusipEventFeatures')

    def __str__(self):
        return self._make_str('CusipEventFeatures')


class IssuerEventFeatures(EventFeatures):
    def __init__(self, last_total_debt_event_features=None):
        self.last_total_debt_event_features = last_total_debt_event_features  # dict[feature_name, feature_value]

    def __repr__(self):
        return self._make_repr('IssuerEventFeatures')

    def __str__(self):
        return self._make_str('IssuerEventFeatures')


class EventFeatureMakers(object):
    'create features from events, tracking historic context'
    def __init__(self, control, counter):
        self.control = control
        self.counter = counter

        # keep track of the last features created from events
        # the last features will be used to create the next feature vector
        self.cusip_event_features_dict = {}  # Dict[cusip, CusipEventFeatures]
        self.issuer_event_features = IssuerEventFeatures()

        # feature makers (that hold state around their event streams)
        self.trace_event_feature_makers = {}  # Dict[cusip, feature_maker]
        self.total_debt_event_feature_maker = seven.feature_makers2.TotalDebt(control.arg.issuer, control.arg.cusip)

    def maybe_extract_features(self, event):
        'return None (if features were extracted without error) or errs'
        # MAYBE: move this code into Event.maybe_extract_features()
        # if so, how will we determine that we have all feature types
        #   maybe Event.n_feature_sets_created
        if isinstance(event.id, seven.EventId.TraceEventId):
            # build features for all cusips because we don't know which will be the OTR cusips
            # until we see OTR events
            self.counter['trace events examined'] += 1
            cusip = event.payload['cusip']
            if cusip not in self.trace_event_feature_makers:
                # the feature makers accumulate state
                # we need one for each cusip
                self.trace_event_feature_makers[cusip] = seven.feature_makers2.Trace(
                    self.control.arg.issuer,
                    cusip,
                )
            if cusip not in self.cusip_event_features_dict:
                self.cusip_event_features_dict[cusip] = CusipEventFeatures()
            # if cusip.endswith('BN9'):
            #     print 'found OTR cusip of interest', cusip
            #     pdb.set_trace()
            features, errs = self.trace_event_feature_makers[cusip].make_features(event.id, event.payload)
            if errs is None:
                self.cusip_event_features_dict[cusip].last_trace_event_features = features
                self.counter['trace event feature sets created for cusip %s' % cusip] += 1
                return None
            else:
                self.counter['trace event errs found cusip %s' % cusip] += len(errs)
                return errs

        elif isinstance(event.id, seven.EventId.OtrCusipEventId):
            # change the on-the-run cusip, if the event is for the primary cusip
            self.counter['otr cusip events examined'] += 1
            otr_cusip = event.payload['otr_cusip']
            primary_cusip = event.payload['primary_cusip']
            # if primary_cusip.endswith('AJ9') and otr_cusip.endswith('BN9'):
            #     print 'found otr cusip assignment', primary_cusip, otr_cusip
            #     pdb.set_trace()
            if otr_cusip not in self.cusip_event_features_dict:
                self.cusip_event_features_dict[otr_cusip] = CusipEventFeatures()
            if primary_cusip not in self.cusip_event_features_dict:
                self.cusip_event_features_dict[primary_cusip] = CusipEventFeatures()
            if primary_cusip == self.control.arg.cusip:
                features = {'id_otr_cusip': otr_cusip}
                self.cusip_event_features_dict[primary_cusip].last_otr_cusip = otr_cusip
                self.counter['otr cusip feature sets created'] += 1
                return None
            else:
                err = 'OTR cusip %s for non-query cusip %s' % (otr_cusip, primary_cusip)
                self.counter['otr cusip events skipped (not for primary cusip)'] += 1
                return [err]

        elif isinstance(event.id, seven.EventId.TotalDebtEventId):
            # Total Debt is a feature of the issuer
            self.counter['total debt events examined'] += 1
            features, errs = self.total_debt_event_feature_maker.make_features(event.id, event.payload)
            if errs is None:
                self.issuer_event_features.last_total_debt_event_features = features
                self.counter['total debut feature sets created'] += 1
                return None
            else:
                self.counter['total debt errs found'] += len(errs)
                return errs

        else:
            print 'not yet handling', event.id
            pdb.set_trace()


def make_expert_accuracies(control, ensemble_hyperparameters, event, expert_predictions):
    'return Dict[model_spec, float] of normalized weights for the experts'
    pdb.set_trace()
    # detemine unnormalized weights
    actual = float(event.payload[control.arg.target])
    # action_signal.actual(reclassified_trade_type, event, actual)
    temperature_expert = ensemble_hyperparameters.weight_temperature
    unnormalized_weights = collections.defaultdict(float)  # Dict[model_spec, float]
    sum_unnormalized_weights = 0.0
    for expert in expert_predictions:
        expert_event = expert.event
        these_expert_predictions = expert.expert_predictions  # Dict[model_spec, float]
        timedelta = event.id.datetime() - expert_event.id.datetime()
        assert ensemble_hyperparameters.weight_units == 'days'
        elapsed_seconds = timedelta.total_seconds()
        elapsed_days = elapsed_seconds / (24.0 * 60.0 * 60.0)
        weight_expert = math.exp(-temperature_expert * elapsed_days)
        print 'accuracy expert event', expert_event, weight_expert
        for model_spec, prediction in these_expert_predictions.iteritems():
            # print model_spec, prediction
            abs_error = abs(actual - prediction)
            unnormalized_weights[model_spec] += weight_expert * abs_error
            sum_unnormalized_weights += weight_expert * abs_error
            print elapsed_days, model_spec, prediction, abs_error, weight_expert, unnormalized_weights[model_spec]
        pdb.set_trace()
    pdb.set_trace()

    # normalize the weights
    normalized_weights = {}  # Dict[model_spec, float]
    for model_spec, unnormalized_weight in unnormalized_weights.iteritems():
        normalized_weights[model_spec] = unnormalized_weight / sum_unnormalized_weights
    if True:
        print 'model_spec, where normalized weights > 2 * mean normalized weight'
        mean_normalized_weight = 1.0 / len(normalized_weights)
        sum_normalized_weights = 0.0
        for model_spec, normalized_weight in normalized_weights.iteritems():
            sum_normalized_weights += normalized_weight
            if normalized_weight > 2.0 * mean_normalized_weight:
                print '%30s %8.6f' % (str(model_spec), normalized_weight)
        print 'sum of normalized weights', sum_normalized_weights
    pdb.set_trace()

    seven.logging.info('new accuracies from event id %s' % event.id)
    return normalized_weights


def do_work(control):
    'write predictions from fitted models to file system'
    ensemble_hyperparameters = EnsembleHyperparameters()  # for now, take defaults
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

    # we separaptely build a model for B and S trades
    # The B models are trained only with B feature vectors
    # The S models are trainined only with S feature vectors

    expected_reclassified_trade_types = ('B', 'S')

    # build empty data structures that vary according to the reclassified trade type
    EnsemblePrediction = collections.namedtuple('EnsemblePrediction', 'event ensemble_predictions')
    ensemble_predictions = {}  # Dict[reclassified_trade_type, deque[EnsemblePrediction]]

    ExpertAccuracies = collections.namedtuple('ExpertAccuracites', 'event dictionary')
    expert_accuracies = {}  # Dict[reclassified_trade_type, ExpertAccuracies]

    ExpertPrediction = collections.namedtuple('ExpertPrediction', 'event expert_predictions')
    expert_predictions = {}  # Dict[reclassified_trade_type, deque[ExpertPrediction]]

    feature_vectors = {}  # Dict[reclassified_trade_type, deque]

    TrainedExpert = collections.namedtuple('TrainedExpert', 'feature_vector experts')
    trained_experts = {}  # Dict[reclassified_trade_type, List[TrainedExpert]

    max_n_trades_back = make_max_n_trades_back(control.arg.hpset)
    for reclassified_trade_type in expected_reclassified_trade_types:
        ensemble_predictions[reclassified_trade_type] = collections.deque([], None)  # None ==> unbounded
        expert_accuracies[reclassified_trade_type] = {}
        expert_predictions[reclassified_trade_type] = collections.deque(
            [],
            ensemble_hyperparameters.max_n_expert_predictions_saved,
            )
        feature_vectors[reclassified_trade_type] = collections.deque([], max_n_trades_back)
        trained_experts[reclassified_trade_type] = []

    action_signal = ActionSignal(
        path_actions=control.path['out_actions'],
        path_signal=control.path['out_signal']
    )
    year, month, day = control.arg.start_date.split('-')
    start_date = datetime.datetime(int(year), int(month), int(day), 0, 0, 0)
    event_feature_makers = EventFeatureMakers(control, counter)
    ignored = datetime.datetime(2017, 4, 1, 0, 0, 0)
    print 'pretending that events before %s never happened' % ignored
    while True:
        try:
            event = event_queue.next()
            counter['events read']
        except StopIteration:
            break  # all the event readers are empty

        if event.id.datetime() < ignored:
            continue

        counter['events processed'] += 1
        print 'processing event # %d: %s' % (counter['events processed'], event)

        # print summary of global state
        def print_state_list(name, value):
            print '%30s B %6d S %6d' % (
                name,
                len(value['B']),
                len(value['S']),
            )

        if event.id.datetime() >= start_date:
            print_state_list('ensemble_predictions', ensemble_predictions)
            print_state_list('expert_accuracies', expert_accuracies)
            print_state_list('feature_vectors', feature_vectors)
            print_state_list('trained_experts', trained_experts)
            print 'processed %d events in %0.2f wallclock minutes' % (
                counter['events processed'] - 1,
                control.timer.elapsed_wallclock_seconds() / 60.0,
            )

        # attempt to extract features from the event
        errs = event_feature_makers.maybe_extract_features(event)
        if errs is not None:
            event_not_usable(errs, event)
            continue
        else:
            counter['event features created'] += 1

        # only extract event features until we reach the start date
        if event.id.datetime() < start_date:
            msg = 'skipped more than event feature extraction because before start date %s' % control.arg.start_date
            if counter[msg] == 0:
                print msg
            counter[msg] += 1
            continue

        # determine accuracy of the experts, if we have any trained expert models
        if (event.is_trace_print_with_cusip(control.arg.cusip) and
           len(expert_predictions[reclassified_trade_type]) > 0):
            pdb.set_trace()
            expert_accuracies = ExpertAccuracies(
                event=event,
                dictionary=make_expert_accuracies(
                    control,
                    ensemble_hyperparameters,
                    event,
                    expert_predictions[reclassified_trade_type],
                )
            )

        # create feature vector, if we have created features for the required events
        feature_vector, errs = maybe_create_feature_vector(
            control,
            event,
            event_feature_makers,
        )
        if errs is not None:
            for err in errs:
                no_feature_vector(err, event)
            continue
        feature_vectors[feature_vector.reclassified_trade_type].append(feature_vector)
        counter['feature vectors created'] += 1

        print 'skipping training for now'
        continue
        pdb.set_trace()

        # test (=predict) the feature vector, if we have a model
        reclassified_trade_type = feature_vector.reclassified_trade_type()
        if len(trained_experts[reclassified_trade_type]) > 0:
            # find most recently trained expert for this trade type
            last_trained_expert = trained_experts[reclassified_trade_type][-1]  # the last expert was trained most recently
            event_expert_predictions = {}  # Dict[model_spec, float]
            for trained_model_spec, trained_model in last_trained_expert.experts.iteritems():
                predictions = trained_model.predict([feature_vector])
                assert len(predictions) == 1
                event_expert_predictions[trained_model_spec] = predictions[0]
            expert_predictions[reclassified_trade_type].append(ExpertPrediction(
                event=event,
                expert_predictions=event_expert_predictions,
            ))
            seven.logging.info('created predictions for features vector %s' % feature_vector)
            # Tcreate ensemble predictions, if we have accuracies for the experts
            if len(expert_accuracies[reclassified_trade_type]) > 0:
                # create the ensemble prediction
                # it is the accuracy-weighted sum of the experts' predictions
                ensemble_prediction = 0.0
                print 'TODO: fix the accuracies, they seem inverted'
                for model_spec, model_weight in expert_accuracies[reclassified_trade_type].dictionary.iteritems():
                    print 'ensemble prediction expert %30s weight %f prediction %f' % (
                        model_spec,
                        model_weight,
                        event_expert_predictions[model_spec],
                    )
                    ensemble_prediction += model_weight * event_expert_predictions[model_spec]
                print 'ensemble prediction final value', ensemble_prediction
                # determine variance around the ensemble prediction
                variance = 0.0
                for model_spec, model_weight in expert_accuracies[reclassified_trade_type].dictionary.iteritems():
                    delta = ensemble_prediction - event_expert_predictions[model_spec]
                    weighted_delta = model_weight * delta
                    variance += weighted_delta * weighted_delta
                standard_deviation = math.sqrt(variance)
                action_signal.ensemble_prediction(
                    trade_type=reclassified_trade_type,
                    event=event,
                    predicted_value=ensemble_prediction,
                    standard_deviation=standard_deviation,
                )
                ensemble_predictions.append(EnsemblePrediction(event, ensemble_prediction))
                seven.logging.info('ensemble prediction trade_type %s value %f stddev %f' % (
                    reclassified_trade_type,
                    ensemble_prediction,
                    standard_deviation,
                ))

        # train new models using the new feature vector
        # for now, train whenever we can
        # later, we may decide to train occassionally
        # the training could run assynchronously and then feed a trained-model event into the system
        if not event.id.source.startswith('trace_'):
            no_training('event from source %s, not trace prints' % event.id.source, event)
            continue
        if not event.payload['cusip'] == control.arg.cusip:
            no_training('event is trace print for cusip %s, not query %s' % (
                event.payload['cusip'],
                control.arg.cusip,
                ),
                event,
            )
            continue

        # event was for a trace print for the query cusip
        # the reclassified trade type is in every trace print event record
        reclassified_trade_type == feature_vector.reclassified_trade_type()

        # save the feature vectors separately for each reclassified trade type
        assert reclassified_trade_type in expected_reclassified_trade_types
        feature_vectors[reclassified_trade_type].append(feature_vector)

        # make the training data
        relevant_training_features, relevant_training_targets = make_training_data(
            control.arg.target,
            list(feature_vectors[reclassified_trade_type]),
        )
        assert len(relevant_training_features) == len(relevant_training_targets)

        if len(relevant_training_features) == 0:
            no_training(
                'zero training vectors for %s' % feature_vector,
                event,
            )
            continue

        # train the experts
        seven.logging.info('training the experts, having seen event id %s' % event.id)
        control.timer.lap('start make the trained experts')
        new_experts = make_trained_expert_models(
            control,
            relevant_training_features,
            relevant_training_targets,
        )
        cpu, wallclock = control.timer.lap('train the experts')
        seven.logging.info('fit %s experts in %.0f wallclock seconds' % (len(new_experts), wallclock))
        trained_experts[reclassified_trade_type].append(
            TrainedExpert(
                feature_vector=feature_vector,
                experts=new_experts,
            )
        )
        counter['expert trainings completed'] += 1

        # TODO: write the importances
        # {dir_out}/importances-{event_id}-{model_name}-{B|S}.csv

        counter['event loops completed'] += 1
        seven.logging.info('finished %d complete event loops' % counter['event loops completed'])

        reclassified_trade_type = None   # try to prevent coding errors in the next loop iteration

        if control.arg.test and counter['event loops completed'] > 500:
            print 'breaking out of event loop: TESTING'
            break
        gc.collect()  # try to get memory usage roughly constant (to enable running of multiple instances)

    action_signal.close()
    event_queue.close()
    print 'counters'
    for k in sorted(counter.keys()):
        print '%-70s: %6d' % (k, counter[k])
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
