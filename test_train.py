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
import signal
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
        return 'TargetVector(creation_event=%s, target_name=%s, payload=%f)' % (
            self.creation_event,
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

    def is_trace_print_with_cusip(self, cusip):
        'return True or False'
        if self.is_trace_print():
            return self.maybe_cusip() == cusip
        else:
            return False

    def maybe_cusip(self):
        'return cusip (if this event is a trace print) else None'
        if self.is_trace_print():
            return self.payload['cusip']
        else:
            return None

    def maybe_reclassified_trade_type(self):
        'return reclassified trade type (if event source is TraceEvent), else None'
        if isinstance(self.id, seven.EventId.TraceEventId):
            return self.payload['reclassified_trade_type']
        else:
            return None


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


def no_expert_prediction(msg, event):
    oops('no expert prediction created', msg, event)


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


def maybe_make_accuracies(event, expert_predictions):
    'return (accuracies, errs)'
    assert isinstance(event, Event)
    assert isinstance(expert_predictions, TypedDequeDict)
    pdb.set_trace()
    if len(expert_predictions) == 0:
        err = 'no expert predictions'
        return None, [err]
    pdb.set_trace()
    print 'write more'
    pass


def maybe_make_expert_predictions(control, feature_vector, trained_expert_models):
    'return (expert_predictions:Dict[model_spec, float], errs)'
    # make predictions using the last trained expert model
    if len(trained_expert_models) == 0:
        err = 'no trained expert models'
        return None, [err]
    last_expert_models = trained_expert_models[-1].experts
    result = {}
    print 'predicting most recently constructed feature vector using most recently trained experts'
    for model_spec, trained_model in last_expert_models.iteritems():
        predictions = trained_model.predict([feature_vector.payload])
        assert len(predictions) == 1
        prediction = predictions[0]
        result[model_spec] = prediction
    return result, None


def maybe_train_expert_models(control,
                              ensemble_hyperparameters,
                              event,
                              feature_vectors,
                              last_expert_training_time,
                              verbose=False):
    'return (fitted_models, errs)'
    assert isinstance(ensemble_hyperparameters, EnsembleHyperparameters)
    assert isinstance(event, Event)
    assert isinstance(feature_vectors, collections.deque)
    assert isinstance(last_expert_training_time, datetime.datetime)

    if event.is_trace_print_with_cusip(control.arg.cusip):
        delay = ensemble_hyperparameters.min_minutes_between_training
        if event.id.datetime() < last_expert_training_time + delay:
            err = 'not time to train'
            return None, [err]
    else:
        err = 'not a trace print event for query cusip'
        return None, [err]

    if len(feature_vectors) < 2:
        err = 'need at least 2 feature vectors, had %d' % len(feature_vectors)
        return None, [err]

    training_features = []
    training_targets = []
    for i in xrange(0, len(feature_vectors) - 1):
        training_features.append(feature_vectors[i].payload)
        # the target is the next oasspread
        next = feature_vectors[i + 1]
        training_targets.append(next.payload['p_trace_prior_0_%s' % control.arg.target])
    assert len(training_features) == len(training_targets)

    print 'fitting experts to %d training samples' % len(training_features)
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
                print 'fitting expert', model.model_spec
            model.fit(training_features, training_targets)
        except seven.models2.ExceptionFit as e:
            seven.logging.warning('could not fit %s: %s' % (model_spec, e))
            continue
        result[model_spec] = model  # the fitted model
    return result, None


class EnsembleHyperparameters(object):
    def __init__(self):
        self.expected_reclassified_trade_types = ('B', 'S')
        self.max_n_ensemble_predictions = 100
        self.max_n_expert_accuracies = 100
        self.max_n_expert_predictions = 100
        self.max_n_feature_vectors = 100
        self.max_n_trained_experts = 100
        self.min_minutes_between_training = datetime.timedelta(0, 60.0 * 60.0)  # 1 hour
        self.weight_temperature = 1.0
        self.weight_units = 'days'  # pairs with weight_temperature

    def __repr__(self):
        return 'EnsembleHyperparameters(...)'


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


class TypedDequeDict(object):
    def __init__(self, item_type, initial_items=[], maxlen=None, allowed_dict_keys=('B', 'S')):
        self._item_type = item_type
        self._initial_items = copy.copy(initial_items)
        self._maxlen = maxlen
        self._allowed_dict_keys = copy.copy(allowed_dict_keys)

        self._dict = {}  # Dict[trade_type, deque]
        for key in allowed_dict_keys:
            self._dict[key] = collections.deque(self._initial_items, self._maxlen)

    def __repr__(self):
        lengths = ''
        for expected_trade_type in self._allowed_dict_keys:
            next = 'len %s=%d' % (expected_trade_type, len(self._dict[expected_trade_type]))
            if len(lengths) == 0:
                lengths = next
            else:
                lengths += ', ' + next
        return 'TradeTypdedDeque(%s)' % lengths

    def __getitem__(self, dict_key):
        'return the deque with the trade type'
        assert dict_key in self._allowed_dict_keys
        return self._dict[dict_key]

    def append(self, key, item):
        assert key in self._allowed_dict_keys
        assert isinstance(item, self._item_type)
        self._dict[key].append(item)
        return self

    def len(self, trade_type):
        assert trade_type in self._allowed_dict_keys
        return len(self._dict[trade_type])


EnsemblePrediction = collections.namedtuple('EnsemblePrediction', 'event ensemble_predictions')
ExpertAccuracies = collections.namedtuple('ExpertAccuracites', 'event dictionary')
ExpertPredictions = collections.namedtuple('ExpertPrediction', 'event expert_predictions')
TrainedExpert = collections.namedtuple('TrainedExpert', 'feature_vector experts')


class ControlCHandler(object):
    # NOTE: this does not work on Windows
    def __init__(self):
        def signal_handler(signal, handler):
            self.user_pressed_control_c = True
            signal.signal(signal.SIGINT, self._previous_hanlder)

        self._previous_handler = signal.signal(signal.SIGINT, signal_handler)
        self.user_pressed_control_c = False


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

    # max_n_trades_back = make_max_n_trades_back(control.arg.hpset)

    # these variable hold the major elements of the state

    ensemble_predictions = TypedDequeDict(
        item_type=EnsemblePrediction,
        maxlen=ensemble_hyperparameters.max_n_ensemble_predictions,
    )
    expert_accuracies = TypedDequeDict(
        item_type=ExpertAccuracies,
        maxlen=ensemble_hyperparameters.max_n_expert_accuracies,
    )
    expert_predictions = TypedDequeDict(
        item_type=ExpertPredictions,
        maxlen=ensemble_hyperparameters.max_n_expert_predictions,
    )
    feature_vectors = TypedDequeDict(
        item_type=FeatureVector,
        maxlen=ensemble_hyperparameters.max_n_feature_vectors,
    )
    trained_expert_models = TypedDequeDict(
        item_type=TrainedExpert,
        maxlen=ensemble_hyperparameters.max_n_trained_experts,
    )

    # define other variables needed in the event loop
    action_signal = ActionSignal(
        path_actions=control.path['out_actions'],
        path_signal=control.path['out_signal']
    )
    year, month, day = control.arg.start_date.split('-')
    start_date = datetime.datetime(int(year), int(month), int(day), 0, 0, 0)
    event_feature_makers = EventFeatureMakers(control, counter)
    last_expert_training_time = datetime.datetime(1, 1, 1, 0, 0, 0)  # a long time ago
    ignored = datetime.datetime(2017, 4, 1, 0, 0, 0)
    # control_c_handler = ControlCHandler()
    print 'pretending that events before %s never happened' % ignored
    while True:
        # if control_c_handler.user_pressed_control_c:
        #     print 'breaking out of event loop because of CTRL+C'
        #     break
        try:
            event = event_queue.next()
            counter['events read']
        except StopIteration:
            break  # all the event readers are empty

        if event.id.datetime() < ignored:
            counter['events ignored'] += 1
            continue

        counter['events processed'] += 1
        print 'processing event # %d: %s' % (counter['events processed'], event)

        # print summary of global state
        if event.id.datetime() >= start_date:
            format = '%30s: %s'
            print format % ('ensemble_predictions', ensemble_predictions)
            print format % ('expert_accuracies', expert_accuracies)
            print format % ('expert_predictions', expert_predictions)
            print format % ('feature_vectors', feature_vectors)
            print format % ('trained_expert_models', trained_expert_models)

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

        # do no other work until we reach the start date
        if event.id.datetime() < start_date:
            msg = 'skipped more than event feature extraction because before start date %s' % control.arg.start_date
            counter[msg] += 1
            continue

        # determine accuracy of the experts, if we have any trained expert models
        if event.is_trace_print_with_cusip(control.arg.cusip) and False:
            pdb.set_trace()
            accuracies, errs = maybe_make_accuracies(
                event,
                expert_predictions[event.maybe_reclassified_trade_type()],
            )
            if errs is None:
                for err in errs:
                    no_accuracy(err, event)
            else:
                expert_accuracies.append(event.maybe_reclassified_trade_type, accuracies)
                # TODO: figure out how to make an ensemble prediction
                # Hint: use the expert accuracies
                pass
        else:
            no_accuracy('event is not from a trace print', event)

        # create feature vector and train the experts, if we have created features for the required events
        if event.is_trace_print_with_cusip(control.arg.cusip):
            # create feature vectors and train the experts only for trace prints for the query cusip
            # we know the reclassified trade type value exists, because the event is a trace print
            feature_vector, errs = maybe_create_feature_vector(
                control,
                event,
                event_feature_makers,
            )
            if errs is not None:
                for err in errs:
                    no_feature_vector(err, event)
                continue
            # since its a trace print, we know that it has a reclassified trade type
            feature_vectors.append(event.maybe_reclassified_trade_type(), feature_vector)
            counter['feature vectors created'] += 1

            # test the last-trained expert, if any are trained
            event_expert_predictions, errs = maybe_make_expert_predictions(
                control,
                feature_vector,  # use the feature vector just constructede
                trained_expert_models[event.maybe_reclassified_trade_type()],  # use just S or B trade types
            )
            if errs is not None:
                for err in errs:
                    no_expert_prediction(err, event)
            else:
                expert_predictions.append(
                    event.maybe_reclassified_trade_type(),
                    ExpertPredictions(
                        event=event,
                        expert_predictions=event_expert_predictions,
                    ),
                )
                counter['predicted with experts'] += 1

            # possible train new experts, if we have
            # - a trace print event for the query cusip
            # - sufficient feature vectors of the same reclassified trade type as the event
            event_trained_expert_models, errs = maybe_train_expert_models(
                control,
                ensemble_hyperparameters,
                event,
                feature_vectors[event.maybe_reclassified_trade_type()],
                last_expert_training_time,
            )
            if errs is not None:
                for err in errs:
                    no_training(err, event)
                continue
            trained_expert_models.append(
                event.maybe_reclassified_trade_type(),
                TrainedExpert(
                    feature_vectors,
                    event_trained_expert_models,
                ),
            )
            last_expert_training_time = event.id.datetime()
            counter['experts trained'] += 1
        else:
            no_feature_vector('event not a trace print for query cusip', event)

        gc.collect()

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
