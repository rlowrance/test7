'''create predictions and test their accuracy

APPROACH: see the file test_traing.org in the same directory as this file.

INVOCATION
  python test_train.py {issuer} {cusip} {target} {hpset} {start_events} {start_predictions} {--debug} {--test} {--trace}
where
 issuer the issuer (ex: AAPL)
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 start_events: YYYY-MM-DD is the first date that we examine an event.
   It should be on the start of a calendar quarter, because
     the events include the fundamental for the issuer
     fundamentals tend to be published once a quarter
start_predictions: YYYY-MM-DD is the first date on which we attempt to create expert and ensemble predictions
 --debug means to call pdb.set_trace() instead of raisinng an exception, on calls to logging.critical()
   and logging.error()
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
  python features_targets.py AAPL 037833AJ9 oasspread grid5 2017-07-01 2017-07-01 --debug # run until end of events
  python features_targets.py AAPL 037833AJ9 oasspread grid5 2017-04-01 2017-06-01 --debug --test # a few ensemble predictions

See build.py for input and output files.

IDEA FOR FUTURE:

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''

from __future__ import division

# import abc
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
import seven.event_readers
import seven.input_event
import seven.feature_makers2
import seven.fit_predict_output
import seven.HpGrids
import seven.input_event
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
    parser.add_argument('start_events', type=seven.arg_type.date)
    parser.add_argument('start_predictions', type=seven.arg_type.date)
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

    paths = seven.build.test_train(
        arg.issuer,
        arg.cusip,
        arg.target,
        arg.hpset,
        arg.start_events,
        arg.start_predictions,
        test=arg.test,
    )
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    timer = Timer()

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=timer,
    )


EnsemblePrediction = collections.namedtuple(
    'EnsemblePrediction',
    'event predicted_value standard_deviation',
    )
ExpertAccuracies = collections.namedtuple(
    'ExpertAccuracies',
    'event dictionary',
    )
ExpertPredictions = collections.namedtuple(
    'ExpertPrediction',
    'event expert_predictions query_vector training_features training_targets',
    )
TrainedExpert = collections.namedtuple(
    'TrainedExpert',
    'experts training_features, training_targets',
    )


class Actions(object):
    'produce the action file'
    def __init__(self, path):
        self._path = path
        self._field_names = ['action_id', 'action_type', 'action_value']
        self._last_event_id = ''
        self._last_action_suffix = 1
        self._n_rows_written = 0
        self.open()

    def action(self, event, action_type, action_value):
        'append the action, creating a unique id from the event.id'
        return self.actions(event, [(action_type, action_value)])

    def actions(self, event, types_values):
        action_id = self._next_action_id(event)
        for action_type, action_value in types_values:
            self._dict_writer.writerow({
                'action_id': action_id,
                'action_type': action_type,
                'action_value': action_value,
            })
        self.close()
        self.open()

    def close(self):
        self._file.close()

    def open(self):
        self._file = open(self._path, 'ab')
        self._dict_writer = csv.DictWriter(self._file, self._field_names, lineterminator='\n')
        if self._n_rows_written == 0:
            self._dict_writer.writeheader()

    def _next_action_id(self, event):
        print 'TODO: include lag in actual wall clock time from the event'
        # TODO: include wall clock time in the Event
        event_id = str(event.id)
        if event_id == self._last_event_id:
            self._last_action_id_suffix += 1
        else:
            self._last_action_id_suffix = 1
        self._last_event_id = event_id
        action_id = 'ml_%s_%d' % (event_id, self._last_action_id_suffix)
        return action_id


class ActionImportancesSignalOLD(object):
    def __init__(self, path_actions, path_importances, path_signal):
        self._actions = Actions(path_actions)
        self._importances = Importances(path_importances)
        self._signal = Signal(path_signal)

    def actual(self, trade_type, event, target_value):
        assert trade_type in ('B', 'S')
        assert isinstance(event, seven.input_event.Event)
        assert isinstance(target_value, float)
        self._signal.actual(trade_type, event, target_value)

    def ensemble_prediction(self, trade_type, event, predicted_value, standard_deviation):
        assert trade_type in ('B', 'S')
        assert isinstance(event, seven.input_event.Event)
        assert isinstance(predicted_value, float)
        assert isinstance(standard_deviation, float)
        self._actions.actions(
            event=event,
            types_values=[
                ('ensemble_prediction_oasspread_%s' % trade_type, predicted_value),
                ('experts_standard_deviation_%s' % trade_type, standard_deviation),
            ],
        )
        self._signal.prediction(trade_type, event, predicted_value, standard_deviation)

    def importances(self, trade_type, event, importances):
        self._importances.importances(trade_type, event, importances)

    def close(self):
        self._actions.close()
        self._importances.close()
        self._signal.close()


class ControlCHandler(object):
    # NOTE: this does not work on Windows
    def __init__(self):
        def signal_handler(signal, handler):
            self.user_pressed_control_c = True
            signal.signal(signal.SIGINT, self._previous_handler)

        self._previous_handler = signal.signal(signal.SIGINT, signal_handler)
        self.user_pressed_control_c = False


class EnsembleHyperparameters(object):
    def __init__(self):
        self.expected_reclassified_trade_types = ('B', 'S')
        self.max_n_ensemble_predictions = 100
        self.max_n_expert_accuracies = 100
        self.max_n_expert_predictions = 100
        self.max_n_feature_vectors = 100
        self.max_n_trained_experts = 100
        self.min_minutes_between_training = datetime.timedelta(0, 60.0 * 60.0)  # 1 hour
        self.min_minutes_between_training = datetime.timedelta(0, 1.0)  # 1 second
        self.weight_temperature = 1.0
        self.weight_units = 'days'  # pairs with weight_temperature

    def __repr__(self):
        return 'EnsembleHyperparameters(...)'


class EventQueue(object):
    'maintain event timestamp-ordered queue of event'
    def __init__(self, event_reader_classes, control, exceptions):
        self._event_readers = []
        self._event_queue = []  # : heapq
        self._exceptions = exceptions
        for event_reader_class in event_reader_classes:
            event_reader = event_reader_class(control)
            self._event_readers.append(event_reader)
            try:
                while True:
                    event, err = event_reader.next()
                    if err is None:
                        break
                    else:
                        self._exceptions.no_event(err, event)
            except StopIteration:
                seven.logging.warning('event reader %s returned no events at all' % event_reader)
            heapq.heappush(self._event_queue, (event, event_reader))
        print 'initial event queue'
        q = copy.copy(self._event_queue)  # make a copy, becuase the heappop method mutates its argument
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
            oldest_event, event_reader = heapq.heappop(self._event_queue)
        except IndexError:
            raise StopIteration()

        # replace the oldest event with the next event from the same event reader
        try:
            while True:
                new_event, err = event_reader.next()
                if err is None:
                    break
                else:
                    self._exceptions.no_event(err, new_event)
            heapq.heappush(self._event_queue, (new_event, event_reader))
        except StopIteration:
            event_reader.close()

        return oldest_event

    def close(self):
        'close each event reader'
        for event_reader in self._event_readers:
            event_reader.close()


class FeatureVector(object):
    def __init__(self,
                 creation_event,
                 primary_cusip_event_features,
                 otr_cusip_event_features):
        # these fields are part of the API
        self.creation_event = copy.copy(creation_event)
        primary_event = primary_cusip_event_features.value['id_trace_event']
        self.reclassified_trade_type = primary_event.payload['reclassified_trade_type']
        self.payload = self._make_payload(
            primary_cusip_event_features,
            otr_cusip_event_features,
        )
        # TODO: implement cross-product of the features from the events
        # for example, determine the debt to equity ratio

    def _make_payload(self, primary_cusip_event_features, otr_cusip_event_features):
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
        append(all_features, 'p', primary_cusip_event_features.value)
        append(all_features, 'otr', otr_cusip_event_features.value)
        return all_features

    def __eq__(self, other):
        'return True iff each payload non-id field is the same'
        for k, v in self.payload.iteritems():
            if not k.startswith('id_'):
                if other.payload[k] != v:
                    return False
        return True

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'FeatureVector(creation_event=%s, rct=%s, n features=%d)' % (
            self.creation_event,
            self.reclassified_trade_type,
            len(self.payload),
        )

    def __str__(self):
        'return fields in the Feature Vector'
        # As present, there only only 6
        values = ''
        for key in sorted(self.payload.keys()):
            if not key.startswith('id_'):
                if len(values) > 0:
                    values += ', '
                values += '%s=%0.2f' % (key, self.payload[key])
        return 'FeatureVector(%s)' % values


class Importances(object):
    def __init__(self, path):
        self._path = path
        self._field_names = [
            'event_datetime', 'event_id',
            'model_name', 'feature_name', 'accuracy_weighted_feature_importance',
        ]
        self._n_rows_written = 0
        self.open()

    def close(self):
        self._file.close()

    def importances(self, trade_type, event, importances):
        'write next rows of CSV file'
        for model_name, importances_d in importances.iteritems():
            for feature_name, feature_importance in importances_d.iteritems():
                d = {
                    'event_datetime': event.id.datetime(),
                    'event_id': str(event.id),
                    'model_name': model_name,
                    'feature_name': feature_name,
                    'accuracy_weighted_feature_importance': feature_importance,
                }
                self._dict_writer.writerow(d)
                self._n_rows_written += 1
        self.close()
        self.open()

    def open(self):
        self._file = open(self._path, 'ab')
        self._dict_writer = csv.DictWriter(self._file, self._field_names, lineterminator='\n')
        if self._n_rows_written == 0:
            self._dict_writer.writeheader()


class Irregularities(object):
    'keep track of exceptions: event not usbale, enseble prediction not made, ...'
    def __init__(self):
        self.counter = collections.Counter()

    def event_not_usable(self, errs, event):
        for err in errs:
            self._oops('event not usable', err, event)

    def no_accuracy(self, msg, event):
        self._oops('unable to determine expert accuracy', msg, event)

    def no_actual(self, msg, event):
        self._oops('unable to determine actual target value', msg, event)

    def no_event(self, msg, event):
        self._oops('event reader not able to create an event', msg, event)

    def no_expert_predictions(self, msg, event):
        self._oops('unable to create expert predictions', msg, event)

    def no_expert_training(self, msg, event):
        self._oops('unable to train the experts', msg, event)

    def no_feature_vector(self, msg, event):
        self._oops('unable to create feature vector', msg, event)

    def no_ensemble_prediction(self, msg, event):
        self._oops('unable to create ensemble prediction', msg, event)

    def no_importances(self, msg, event):
        self._oops('unable to creating importances', msg, event)

    def no_training(self, msg, event):
        self._oops('unable to train the experts', msg, event)

    def _oops(self, what_happened, message, event):
        seven.logging.info('%s: %s: %s' % (what_happened, message, event))
        self.counter[message] += 1


class Signal(object):
    'produce the signal as a CSV file'
    def __init__(self, path):
        self._path = path
        self._field_names = [
            'datetime', 'event_source', 'event_source_id',   # these columns are dense
            'event_cusip',
            'prediction_B', 'prediction_S',                  # thse columns are sparse
            'standard_deviation_B', 'standard_deviation_S',  # these columns are sparse
            'actual_B', 'actual_S',                          # these columns are sparse
            ]
        self._n_rows_written = 0
        self.open()

    def _event(self, event):
        'return dict for the event columns'

        result = {
            'datetime': event.datetime(),
            'event_source': event.source,
            'event_source_id': event.source_identifier,
        }
        if event.maybe_cusip() is not None:
            result['event_cusip'] = event.cusip()
        return result

    def actual(self, trade_type, event, target_value):
        d = self._event(event)
        d.update({
            'actual_%s' % trade_type: target_value,
        })
        self._dict_writer.writerow(d)

    def prediction(self, trade_type, event, predicted_value, standard_deviation):
        d = self._event(event)
        d.update({
            'prediction_%s' % trade_type: predicted_value,
            'standard_deviation_%s' % trade_type: standard_deviation,
        })
        self._dict_writer.writerow(d)

    def close(self):
        self._file.close()

    def open(self):
        self._file = open(self._path, 'wb')
        self._dict_writer = csv.DictWriter(self._file, self._field_names, lineterminator='\n')
        if self._n_rows_written == 0:
            self._dict_writer.writeheader()


class SimulatedTime(object):
    def __init__(self):
        self.datetime = datetime.datetime(1, 1, 1, 0, 0, 0)

        self._last_event_wallclock = None

    def __repr__(self):
        return 'SimulatedTime(%s)' % self.datetime

    def handle_event(self):
        elapsed_wallclock = datetime.datetime.now() - self._last_event_wallclock
        self.datetime = self.datetime + elapsed_wallclock

    def see_event(self, event):
        if event.date() != self.datetime.date():
            self.datetime = event.datetime()
        else:
            if event.datetime() > self.datetime:
                self.datetime = event.datetime()  # assume no latency in obtain the event
        self._last_event_wallclock = datetime.datetime.now()


class TargetVector(object):
    def __init__(self, event, target_name, target_value):
        assert isinstance(event, seven.input_event.Event)
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


class Trace(object):
    'write complete log (a trace log)'
    def __init__(self, path):
        self._path = path
        self._field_names = ['simulated_datetime', 'time_description', 'what_happened', 'info']
        self._n_rows_written = 0

        self._open()

    def close(self):
        self._file.close()

    def experts_trained(self, dt, expert_models, training_features, training_targets):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclcok to train experts',
            'what_happened': 'new experts were trained',
            'info': 'trained %d experts on %d feature vectors' % (
                len(expert_models),
                len(training_features),
            )
        }
        self._dict_writer.writerow(d)

    def new_feature_vector_created(self, dt, feature_vector):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to create feature vector',
            'what_happened': 'new feature vector created from relevant event features',
            'info': str(feature_vector),
        }
        self._dict_writer.writerow(d)

    def liq_flow_on_the_run_event_created(self, dt, event):
        d = {
            'simulated_datetime': dt,
            'time_description': 'max(simulated time, time from event record)',
            'what_happened': 'event created from liq flow on the run stream record',
            'info': str(event),
        }
        self._dict_writer.writerow(d)

    def liq_flow_on_the_run_event_features_created(self, dt, event):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to extract event features',
            'what_happened': 'liq_flow_on_the_run event features were extracted and saved',
            'info': str(event),
        }
        self._dict_writer.writerow(d)

    def trace_event_created(self, dt, event):
        d = {
            'simulated_datetime': dt,
            'time_description': 'max(simulated time, time from event record)',
            'what_happened': 'event created from trace print stream record',
            'info': str(event),
        }
        self._dict_writer.writerow(d)

    def trace_event_features_created(self, dt, event):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to extract event features',
            'what_happened': 'trace event features were extracted and saved',
            'info': str(event),
        }
        self._dict_writer.writerow(d)

    def _open(self):
        self._file = open(self._path, 'wb')
        self._dict_writer = csv.DictWriter(self._file, self._field_names, lineterminator='\n')
        if self._n_rows_written == 0:
            self._dict_writer.writeheader()


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
        return 'TypedDequeDict(%s)' % lengths

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


def make_max_n_trades_back(hpset):
    'return the maximum number of historic trades a model uses'
    grid = seven.HpGrids.construct_HpGridN(hpset)
    max_n_trades_back = 0
    for model_spec in grid.iter_model_specs():
        if model_spec.n_trades_back is not None:  # the naive model does not have n_trades_back
            max_n_trades_back = max(max_n_trades_back, model_spec.n_trades_back)
    return max_n_trades_back


def maybe_create_feature_vectorOLD(control, event, event_feature_makers):
    'return (feature_vector, errs)'
    if event.is_trace_print_with_cusip(control.arg.cusip):
        debug = event.id.datetime() > datetime.datetime(2017, 6, 5, 12, 2, 0)
        debug = False
        if debug:
            pdb.set_trace()
        # make sure that we know all the fields in the primary cusip
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


def maybe_make_accuracies(event, expert_predictions, ensemble_hyperparameters, control, verbose=True):
    'return (accuracies, errs)'
    assert isinstance(event, seven.input_event.Event)
    assert isinstance(expert_predictions, collections.deque)
    assert isinstance(ensemble_hyperparameters, EnsembleHyperparameters)
    if len(expert_predictions) == 0:
        err = 'no expert predictions'
        return None, [err]
    try:
        actual = float(event.payload[control.arg.target])
    except:
        err = 'unable to convert event %s value %s to float' % (
            control.arg.target,
            event.payload[control.arg.target],
        )
        return None, [err]
    unnormalized_accuracies = collections.defaultdict(float)  # Dict[model_spec, float]
    sum_weights = 0.0
    for expert_prediction in expert_predictions:
        assert isinstance(expert_prediction, ExpertPredictions)
        # the accuracy is a weighted average
        # determine the weight, a funtion of the age of the expert prediction and the accuracy of a model
        assert ensemble_hyperparameters.weight_units == 'days'
        age_td = event.id.datetime() - expert_prediction.event.id.datetime()
        age_seconds = age_td.total_seconds()
        age_days = age_seconds / (24.0 * 60.0 * 60.0)
        assert age_days >= 0
        for model_spec, prediction in expert_prediction.expert_predictions.iteritems():
            abs_error = abs(prediction - actual)
            weight = math.exp(-ensemble_hyperparameters.weight_temperature * abs_error * age_days)
            if verbose:
                print 'unnormalized %30s %f %f %f' % (model_spec, age_days, abs_error, weight)
            unnormalized_accuracies[model_spec] += weight
            sum_weights += weight
    normalized_accuracies = {}  # Dict[model_spec, float]
    sum_normalized_accuracies = 0.0
    for model_spec, unnormalized_accuracy in unnormalized_accuracies.iteritems():
        normalized_accuracy = unnormalized_accuracy / sum_weights
        if verbose:
            print 'accuracies %d expert prediction sets %30s unnormalized %6.2f normalized %8.6f' % (
                len(expert_predictions),
                model_spec,
                unnormalized_accuracies[model_spec],
                normalized_accuracy,
            )
        normalized_accuracies[model_spec] = normalized_accuracy
        sum_normalized_accuracies += normalized_accuracy
    if verbose:
        print sum_normalized_accuracies
    result = ExpertAccuracies(
        event=event,
        dictionary=normalized_accuracies
    )
    return result, None


def maybe_make_ensemble_prediction(control, ensemble_hyperparameters,
                                   feature_vector, expert_accuracies, trained_expert_models,
                                   verbose=False):
    'return (ensemble_prediction, errs)'
    assert isinstance(ensemble_hyperparameters, EnsembleHyperparameters)
    assert isinstance(feature_vector, FeatureVector)
    assert isinstance(expert_accuracies, collections.deque)
    assert isinstance(trained_expert_models, collections.deque)

    if len(expert_accuracies) == 0:
        err = 'no expert accuracies'
        return None, [err]
    if len(trained_expert_models) == 0:
        err = 'no trained expert models'
        return None, [err]
    relevant_expert_accuracies = expert_accuracies[-1]  # the last determination of accuracy
    relevant_trained_expert_models = trained_expert_models[-1]  # the last ones trained
    ensemble_prediction = 0.0
    expert_predictions = {}
    for model_spec, trained_expert_model in relevant_trained_expert_models.experts.iteritems():
        weight = relevant_expert_accuracies.dictionary[model_spec]
        expert_prediction = trained_expert_model.predict([feature_vector.payload])[0]
        ensemble_prediction += weight * expert_prediction
        expert_predictions[model_spec] = expert_prediction
        if verbose:
            print 'ensemble prediction model_spec %30s expert weight %8.6f expert prediction %f' % (
                model_spec,
                weight,
                expert_prediction,
            )
    if verbose:
        print 'ensemble prediction', ensemble_prediction
    # determine weighted variance of the experts' predictions
    variance = 0.0
    for model_spec, expert_prediction in expert_predictions.iteritems():
        weight = relevant_expert_accuracies.dictionary[model_spec]
        delta = expert_prediction - ensemble_prediction
        weighted_delta = weight * delta
        variance += weighted_delta * weighted_delta
    standard_deviation = math.sqrt(variance)
    if verbose:
        print 'weighted standard deviation of expert predictions', standard_deviation
    return ((ensemble_prediction, standard_deviation), None)


def maybe_make_expert_predictions(control, feature_vector, trained_expert_models):
    'return (expert_predictions:Dict[model_spec, float], errs)'
    # make predictions using the last trained expert model
    if len(trained_expert_models) == 0:
        err = 'no trained expert models'
        return None, [err]
    last_trained_expert_models = trained_expert_models[-1]
    experts = last_trained_expert_models.experts
    expert_predictions = {}
    print 'predicting most recently constructed feature vector using most recently trained experts'
    for model_spec, trained_model in experts.iteritems():
        predictions = trained_model.predict([feature_vector.payload])
        assert len(predictions) == 1
        prediction = predictions[0]
        expert_predictions[model_spec] = prediction
    result = (
        expert_predictions,
        last_trained_expert_models.training_features,
        last_trained_expert_models.training_targets,
        )
    return result, None


def maybe_make_feature_vector(control, current_otr_cusip, event, event_feature_values):
    ' return (FeatureVectors, errs)'
    errs = []
    if control.arg.cusip not in event_feature_values.cusip:
        errs.append('no event features for primary cusip %s' % control.arg.cusip)
    if current_otr_cusip not in event_feature_values.cusip:
        errs.append('no event features for otr cusip %s' % current_otr_cusip)
    if len(errs) == 0:
        feature_vector = FeatureVector(
            creation_event=event,
            primary_cusip_event_features=event_feature_values.cusip[control.arg.cusip],
            otr_cusip_event_features=event_feature_values.cusip[current_otr_cusip],
        )
        return feature_vector, None
    else:
        return None, errs


def maybe_make_weighted_importances(accuracies, trained_expert_models, verbose=False):
    'return (weighed_importances, errs) for last trained_expert_model'
    assert isinstance(trained_expert_models, collections.deque)
    if len(trained_expert_models) == 0:
        err = 'no trained expert models'
        return None, [err]
    # importances: Dict[model_spec_name, Dict[feature_name, feature_importance: float]]
    importances = collections.defaultdict(lambda: collections.defaultdict(float))
    for model_spec, trained_model in trained_expert_models[-1].experts.iteritems():
        for feature_name, feature_importance in trained_model.importances.iteritems():
            if verbose:
                print model_spec, accuracies.dictionary[model_spec], feature_name, feature_importance
            importances[model_spec.name][feature_name] += accuracies.dictionary[model_spec] * feature_importance
    return importances, None


def maybe_train_expert_models(control,
                              ensemble_hyperparameters,
                              event,
                              feature_vectors,
                              last_expert_training_time,
                              verbose=False):
    'return ((trained_models, training_features, training_targets), errs)'
    assert isinstance(ensemble_hyperparameters, EnsembleHyperparameters)
    assert isinstance(event, seven.input_event.Event)
    assert isinstance(feature_vectors, collections.deque)
    assert isinstance(last_expert_training_time, datetime.datetime)

    if event.is_trace_print_with_cusip(control.arg.cusip):
        delay = ensemble_hyperparameters.min_minutes_between_training
        if event.datetime() < last_expert_training_time + delay:
            err = 'not time to train'
            return None, [err]
    else:
        err = 'not a trace print event for query cusip'
        return None, [err]

    if len(feature_vectors) < 2:
        err = 'need at least 2 feature vectors, had %d' % len(feature_vectors)
        return None, [err]

    def dt(index):
        return feature_vectors[index].payload['id_p_trace_event'].datetime()

    training_features = []
    training_targets = []
    err = None
    for i in xrange(0, len(feature_vectors) - 1):
        training_features.append(feature_vectors[i].payload)
        features_datetime = dt(i)
        next_index = i + 1
        while (next_index < len(feature_vectors)) and dt(next_index) <= features_datetime:
            next_index += 1
        if next_index == len(feature_vectors):
            err = 'no future primary cusip trade to find target value'
            break
        target_feature_vector = feature_vectors[next_index]
        training_targets.append(target_feature_vector.payload['p_trace_%s' % control.arg.target])
    if len(training_targets) == 0:
        return None, err
    if err is not None:
        seven.logging.info('truncated possible training set: %s' % err)
    assert len(training_features) == len(training_targets)

    print 'fitting experts to %d training samples' % len(training_features)
    grid = seven.HpGrids.construct_HpGridN(control.arg.hpset)
    trained_models = {}  # Dict[model_spec, trained model]
    for model_spec in grid.iter_model_specs():
        if model_spec.transform_y is not None:
            # this code does no transform say oasspread in the features
            # that needs to be done if the y values are transformed
            seven.logging.critical('I did not transform the y values in the features')
            sys.exit(1)
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
        trained_models[model_spec] = model  # the fitted model
    return (trained_models, training_features, training_targets), None


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
    irregularity = Irregularities()

    ensemble_hyperparameters = EnsembleHyperparameters()  # for now, take defaults
    event_reader_classes = (
        seven.event_readers.AmtOutstandingHistory,
        seven.event_readers.CurrentCoupon,
        seven.event_readers.EtfWeightOfCusipPctAgg,
        seven.event_readers.EtfWeightOfCusipPctLqd,
        seven.event_readers.EtfWeightOfIssuerPctAgg,
        seven.event_readers.EtfWeightOfIssuerPctLqd,
        seven.event_readers.EtfWeightOfSectorPctAgg,
        seven.event_readers.EtfWeightOfSectorPctLqd,
        seven.event_readers.FunExpectedInterestCoverage,
        seven.event_readers.FunGrossLeverage,
        seven.event_readers.FunLtmEbitda,
        seven.event_readers.FunMktCap,
        seven.event_readers.FunMktGrossLeverage,
        seven.event_readers.FunReportedInterestCoverage,
        seven.event_readers.FunTotalAssets,
        seven.event_readers.FunTotalDebt,
        seven.event_readers.HistEquityPrices,
        seven.event_readers.LiqFlowOnTheRun,
        seven.event_readers.SecMaster,
        seven.event_readers.Trace,
    )

    if False:
        test_event_readers(event_reader_classes, control)

    # prime the event streams by read a record from each of them
    event_queue = EventQueue(event_reader_classes, control, irregularity)

    # repeatedly process the youngest event
    counter = collections.Counter()

    # we seperately build a model for B and S trades
    # The B models are trained only with B feature vectors
    # The S models are trained only with S feature vectors

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

    def to_datetime_date(s):
        year, month, day = s.split('-')
        return datetime.date(int(year), int(month), int(day))

    # event_feature_makers = EventFeatureMakers(control, counter)
    EventFeatureMakers = collections.namedtuple('EventFeatureMakers', 'cusip not_cusip')
    event_feature_makers = EventFeatureMakers(
        cusip={},     # Dict[cusip, feature_makers2.FeatureMaker instance]
        not_cusip={}  # Dict[input_event.Event.source:str, feature_makers2.FeatureMaker instance]
    )
    EventFeatureValues = collections.namedtuple('EventFeatureValues', 'cusip not_cusip')
    event_feature_values = EventFeatureValues(
        cusip={},      # Dict[cusip, input_event.EventFeatures]
        not_cusip={},  # Dict[input_event.Event.source:str, input_event.EventFeatures]
    )
    last_expert_training_time = datetime.datetime(1, 1, 1, 0, 0, 0)  # a long time ago
    start_events = to_datetime_date(control.arg.start_events)
    start_predictions = to_datetime_date(control.arg.start_predictions)
    ignored = datetime.datetime(2017, 7, 1, 0, 0, 0)  # NOTE: must be at the start of a calendar quarter
    current_otr_cusip = ''
    feature_vector = None
    output_actions = Actions(control.path['out_actions'])
    output_importances = Importances(control.path['out_importances'])
    output_signal = Signal(control.path['out_signal'])
    output_trace = Trace(control.path['out_trace'])
    seven.logging.verbose_info = False
    seven.logging.verbose_warning = False
    simulated_time = SimulatedTime()
    print 'pretending that events before %s never happened' % ignored
    while True:
        time_event_first_seen = datetime.datetime.now()

        def elapsed_wallclock():
            return datetime.datetime.now() - time_event_first_seen

        def simulated_time_from(dt):
            return dt + (datetime.datetime.now() - time_event_first_seen)

        try:
            event = event_queue.next()
            counter['events read'] += 1
        except StopIteration:
            break  # all the event readers are empty

        if event.date() < start_events:
            counter['events ignored'] += 1
            continue

        simulated_time.see_event(event)
        seven.logging.verbose_info = True
        seven.logging.verbose_warning = True
        counter['events processed'] += 1
        print 'processing event # %d: %s' % (counter['events processed'], event)

        # print summary of global state
        if event.date() >= start_predictions:
            format_template = '%30s: %s'
            print format_template % ('ensemble_predictions', ensemble_predictions)
            print format_template % ('expert_accuracies', expert_accuracies)
            print format_template % ('expert_predictions', expert_predictions)
            print format_template % ('feature_vectors', feature_vectors)
            print format_template % ('trained_expert_models', trained_expert_models)

            print 'processed %d events in %0.2f wallclock minutes' % (
                counter['events processed'] - 1,
                control.timer.elapsed_wallclock_seconds() / 60.0,
            )

        # attempt to extract event-features from the event
        # keep track of the current OTR cusip
        if True:  # convert to event features for selected event types
            # set current_otr_cusip for use below
            output_trace.trace_event_created(simulated_time.datetime, event)
            if event.source == 'trace':
                cusip = event.cusip()
                print 'about to create event features for cusip', cusip
                if cusip not in event_feature_makers.cusip:
                    event_feature_makers.cusip[cusip] = event.event_feature_maker_class(control.arg, event)
                event_features, errs = event_feature_makers.cusip[cusip].make_features(event)
                if errs is not None:
                    irregularity.event_not_usable(errs, event)
                    counter['cusip %s events that were not usable' % cusip] += 1
                    continue
                event_feature_values.cusip[cusip] = event_features
                counter['event-feature sets created cusip %s' % cusip] += 1
                simulated_time.handle_event()
                output_trace.trace_event_features_created(simulated_time.datetime, event)
            else:
                source = event.source
                if source == 'liq_flow_on_the_run':
                    if source not in event_feature_makers.not_cusip:
                        event_feature_makers.not_cusip[source] = event.event_feature_maker_class(control.arg, event)
                    event_features, errs = event_feature_makers.not_cusip[source].make_features(event)
                    if errs is not None:
                        irregularity.event_not_usable(errs, event)
                        counter['source %s events that were not usable' % source] += 1
                        continue
                    event_feature_values.not_cusip[source] = event_features
                    if source == 'liq_flow_on_the_run':
                        current_otr_cusip = event_features['liq_flow_on_the_run_otr_cusip']
                    counter['event-features sets created source %s' % source] += 1
                    simulated_time.handle_event()
                    output_trace.liq_flow_on_the_run_event_features_created(simulated_time.datetime, event)
                else:
                    print 'for now, ignoring events from %s' % source
                    continue

        # do no other work until we reach the start date
        if event.date() < start_predictions:
            msg = 'skipped more than event feature extraction because before start date %s' % control.arg.start_date
            counter[msg] += 1
            continue

        counter['events on or after start predictions date'] += 1
        if False and counter['events on or after start predictions date'] > 100:
            print 'for now, breaking out of event loop'
            break

        if True:  # maybe create feature vector
            # The feature vector incorporates information from every event
            # Some of those events (like otr events) do not change the feature vector
            # Create a new feature vector only if its payload would be changed by the accumulated event stream

            # outputs usable below:
            #   feature_vector, possible from a prior event
            #   have_different_feature_vector, the current event created a feature vector with new values
            have_different_feature_vector = False
            new_feature_vector, errs = maybe_make_feature_vector(
                control=control,
                current_otr_cusip=current_otr_cusip,
                event=event,
                event_feature_values=event_feature_values,
            )
            if errs is not None:
                for err in errs:
                    irregularity.no_feature_vector(err, event)
            else:
                have_different_feature_vector = (feature_vector is None) or (new_feature_vector != feature_vector)
                if have_different_feature_vector:
                    feature_vector = new_feature_vector
                    feature_vectors.append(feature_vector.reclassified_trade_type, feature_vector)
                    simulated_time.handle_event()
                    output_trace.new_feature_vector_created(
                        simulated_time.datetime,
                        feature_vector,
                    )
                    counter['feature vectors created'] += 1

        if True:  # maybe determine accuracy of experts
            if event.is_trace_print_with_cusip(control.arg.cusip):   # maybe make accuracies
                # signal the actual target value
                try:
                    actual = float(event.payload[control.arg.target])
                except Exception as e:
                    irregularity.no_actual(str(e), event)
                output_signal.actual(
                    event.payload['reclassified_trade_type'],
                    event,
                    actual,
                )
                # try to determine the accuracies of all of the experts
                # based on the trace print event actual target value
                accuracies, errs = maybe_make_accuracies(
                    event,
                    expert_predictions[event.maybe_reclassified_trade_type()],
                    ensemble_hyperparameters,
                    control,
                )
                if errs is not None:
                    for err in errs:
                        irregularity.no_accuracy(err, event)
                else:
                    pdb.set_trace()
                    expert_accuracies.append(event.maybe_reclassified_trade_type(), accuracies)
                    weighted_importances, errs = maybe_make_weighted_importances(
                        accuracies,
                        trained_expert_models[event.maybe_reclassified_trade_type()],
                    )
                    if errs is not None:
                        for err in errs:
                            irregularity.no_importances(err, event)
                    else:
                        output_importances.importances(
                            event.maybe_reclassified_trade_type(),
                            event,
                            weighted_importances,
                        )
            else:
                irregularity.no_accuracy('event is not from a trace print', event)

        if True:  # maybe make an ensemble prediction (using possibly updated expert accuracies)
            if have_different_feature_vector:
                # attempt to make an ensemble prediction
                ensemble_prediction_standard_deviation, errs = maybe_make_ensemble_prediction(
                    control,
                    ensemble_hyperparameters,
                    feature_vector,
                    expert_accuracies[event.maybe_reclassified_trade_type()],
                    trained_expert_models[event.maybe_reclassified_trade_type()],
                )
                if errs is not None:
                    for err in errs:
                        irregularity.no_ensemble_prediction(err, event)
                else:
                    pdb.set_trace()
                    ensemble_prediction, standard_deviation = ensemble_prediction_standard_deviation
                    ensemble_predictions.append(
                        event.maybe_reclassified_trade_type(),
                        EnsemblePrediction(
                            event=event,
                            predicted_value=ensemble_prediction,
                            standard_deviation=standard_deviation,
                        ),
                    )
                    output_actions.action(
                        event,
                        'ensemble_prediction_oasspread_%s' % event.reclassified_trade_type(),
                        ensemble_prediction,
                    )
                    output_actions.action(
                        event,
                        'experts_standard_deviation_%s' % event.reclassified_trade_type(),
                        standard_deviation,
                    )
                    output_signal.prediction(
                        event.reclassified_trade_type(),
                        event,
                        ensemble_prediction,
                        standard_deviation,
                    )
            else:
                irregularity.no_ensemble_prediction('no different feature vector', event)
        if True:  # maybe test (predict with) the experts
            if have_different_feature_vector:
                # attempt to make predictions will the last-trained set of experts
                result, errs = maybe_make_expert_predictions(
                    control,
                    feature_vector,  # use the feature vector just constructede
                    trained_expert_models[event.maybe_reclassified_trade_type()],  # use just S or B trade types
                )
                if errs is not None:
                    for err in errs:
                        irregularity.no_expert_predictions(err, event)
                else:
                    event_expert_predictions, training_features, training_targets = result
                    expert_predictions.append(
                        event.maybe_reclassified_trade_type(),
                        ExpertPredictions(
                            event=event,
                            expert_predictions=event_expert_predictions,
                            query_vector=feature_vector,
                            training_features=training_features,
                            training_targets=training_targets,
                        ),
                    )
                    counter['experts tested'] += 1
            else:
                irregularity.no_expert_predictions('no different feature vector', event)

        if True:  # maybe train the experts
            if have_different_feature_vector:
                result, errs = maybe_train_expert_models(
                    control,
                    ensemble_hyperparameters,
                    event,
                    feature_vectors[event.maybe_reclassified_trade_type()],
                    last_expert_training_time,
                )
                if errs is not None:
                    for err in errs:
                        irregularity.no_training(err, event)
                    continue
                event_trained_expert_models, training_features, training_targets = result
                trained_expert_models.append(
                    event.maybe_reclassified_trade_type(),
                    TrainedExpert(
                        experts=event_trained_expert_models,
                        training_features=training_features,
                        training_targets=training_targets,
                    ),
                )
                last_expert_training_time = event.datetime()  # experts after a certain elapsed time
                simulated_time.handle_event()
                output_trace.experts_trained(
                    simulated_time.datetime,
                    event_trained_expert_models,
                    training_features,
                    training_targets,
                )
                counter['experts trained'] += 1
            else:
                irregularity.no_expert_training('no different feature vector', event)

        if counter['experts trained'] >= 4:
            print 'for now, stopping'

        if control.arg.test and len(ensemble_predictions['B']) > 2 and len(ensemble_predictions['S']) > 2:
            print 'breaking out of event loop because of --test'
            break

        gc.collect()

    output_actions.close()
    output_importances.close()
    output_signal.close()
    output_trace.close()
    event_queue.close()
    print 'counters'
    for k in sorted(counter.keys()):
        print '%-70s: %6d' % (k, counter[k])
    print
    print '**************************************'
    print 'exceptional conditions'
    for k in sorted(irregularity.counter.keys()):
        print '%-40s: %6d' % (k, irregularity.counter[k])
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
