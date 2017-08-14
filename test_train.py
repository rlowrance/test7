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

APPROACH:
 Repeat until no more input:
    1. Read the next event.
        - For now, the events are in files, one file per source for the event.
        - Each record in each event file has a date or datetime stamp. Each record has the data for an event.
        - Read the records in datetime order.
        - For now, assume that a record with a date stamp and not a datetime stamp was generated at 00:00:00,
            so that it comes first on the date.
        - For now, arbitrarily select from records with the same datetime. This is a problem, as it
            assumes an ordering. In the future, treat all these events as having occured at the same time.
    2. blah
    2. blah
    2. blah
    2. blah
    2. blah
    2. blah
    2. blah
    2. blah

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
import os
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
    'EnsemblePrediction', [
        'action_identifier',
        'creation_event',
        'elapsed_wallclock_seconds',  # time to make expert predictions and ensemble prediction
        'ensemble_prediction',
        'explanation',
        'feature_vector_for_actual',
        'feature_vector_for_query',
        'importances',  # Dict[model_spec.name, Dict[feature_name, weighted_importance]]
        'list_of_trained_experts',
        'simulated_datetime',
        'standard_deviation',
    ],
    )
ExpertAccuracies = collections.namedtuple(
    'ExpertAccuracies', [
        'actual',
        'creation_event',
        'expert_predictions_list',
        'explanation',
        'normalized_accuracies',
        'unnormalized_accuracies',
    ],
    )
ExpertPredictions = collections.namedtuple(
    'ExpertPrediction', [
        'creation_event',
        'feature_vector',
        'expert_predictions',
        'trained_experts',
    ],
    )
TrainedExperts = collections.namedtuple(
    'TrainedExpert', [
        'creation_event',
        'elapsed_wallclock_seconds',  # time for training all of the experts
        'trained_models',             # Dict[model_spec, trained_model]
        'training_features',
        'training_targets',
    ],
    )


class Actions(object):
    'actions are outputs of the machine learning to upstream processes'
    # not the diagnostics, at least for now
    # the actions need to be in a file that an upstream process can tail
    def __init__(self, path):
        self._path = path
        self._field_names = ['action_id', 'action_type', 'action_value']
        self._last_event_id = ''
        self._last_action_suffix = 1
        self._n_rows_written = 0
        self._open()

    def close(self):
        self._file.close()

    def ensemble_prediction(self, ensemble_prediction, trade_type):
        self._write({
            'action_id': ensemble_prediction.action_identifier,
            'action_type': 'ensemble prediction %s' % trade_type,
            'action_value': ensemble_prediction.ensemble_prediction,
        })
        self._write({
            'action_id': ensemble_prediction.action_identifier,
            'action_type': 'ensemble prediction standard deviation %s' % trade_type,
            'action_value': ensemble_prediction.standard_deviation,
        })
        # force out any in-memory buffers
        self.close()
        self._open()

    def _write(self, d):
        'write row'
        self._dict_writer.writerow(d)
        self._n_rows_written += 1

    def _open(self):
        self._file = open(self._path, 'ab')
        self._dict_writer = csv.DictWriter(self._file, self._field_names, lineterminator='\n')
        if self._n_rows_written == 0:
            self._dict_writer.writeheader()


class ActionIdentifiers(object):
    'create unique action identifiers'
    def __init__(self):
        self._suffix = None
        self._last_datetime = datetime.datetime(1, 1, 1)

    def next_identifier(self, current_datetime):
        'return str ml-{datetime}-DDD, where the DDD makes the datetime unique'
        if current_datetime == self._last_datetime:
            self._suffix += 1
        else:
            self._suffix = 1
        current_datetime_str = str(current_datetime).replace(' ', 'T').replace(':', '-')  # like Q lang
        action_identifier = 'ml_%s_%03d' % (
            current_datetime_str,
            self._suffix,
        )
        self._last_datetime = current_datetime
        return action_identifier


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
        self.min_timedelta_between_training = datetime.timedelta(0, 60.0 * 60.0)  # 1 hour
        self.min_timedelta_between_training = datetime.timedelta(0, 1.0)  # 1 second
        self.min_timedelta_between_training = datetime.timedelta(0, 0.0)  # always retrain
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
        return 'FeatureVector(%s, %s)' % (self.reclassified_trade_type, values)


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

    def events_at_same_time(self, prior_event, current_event):
        msg = 'prior event source: %sd source_id %s' % (
            prior_event.source,
            prior_event.source_identifier,
        )
        self._oops('event occured at same datetime', msg, current_event)

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

    def skipped_event(self, msg, event):
        self._oops('event skipped', msg, event)

    def _oops(self, what_happened, message, event):
        seven.logging.info('%s: %s: %s' % (what_happened, message, event))
        self.counter[message] += 1


class Signals(object):
    'signals are the combined actual oasspreads and predictions'
    # these are for presentation purposes and to help diagnose accuracy
    # the signal needs to be in an easy-to-parse file
    def __init__(self, path):
        self._path = path
        self._field_names = [
            'simulated_datetime',
            'event_source', 'event_source_identifier', 'event_cusip',
            'action_identifier',
            'prediction_B', 'prediction_S',
            'standard_deviation_B', 'standard_deviation_S',
            'actual_B', 'actual_S',
            ]
        self._n_rows_written = 0
        self._open()

    def ensemble_prediction(self, ensemble_prediction, trade_type):
        row = {
            'simulated_datetime': ensemble_prediction.simulated_datetime,
            'action_identifier': ensemble_prediction.action_identifier,
            'prediction_%s' % trade_type: ensemble_prediction.ensemble_prediction,
            'standard_deviation_%s' % trade_type: ensemble_prediction.standard_deviation,
        }
        self._write(row)

    def close(self):
        self._file.close()

    def trace_event(self, event):
        row = {
            'simulated_datetime': str(event.datetime()),
            'event_source': event.source,
            'event_source_identifier': event.source_identifier,
            'event_cusip': event.cusip(),
        }
        if event.reclassified_trade_type == 'B':
            row['actual_B'] = event.payload['oasspread']
        else:
            row['actual_S'] = event.payload['oasspread']
        self._write(row)

    def _open(self):
        self._file = open(self._path, 'wb')
        self._dict_writer = csv.DictWriter(self._file, self._field_names, lineterminator='\n')
        if self._n_rows_written == 0:
            self._dict_writer.writeheader()

    def _write(self, d):
        'write row'
        self._dict_writer.writerow(d)
        self._n_rows_written += 1


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

    def ensemble_prediction(self, dt, ensemble_prediction):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to create an ensemble prediction',
            'what_happened': 'the most recent accuracies were used to blend the predictions of the most recently-trained experts',
            'info': 'predicted %s with standard deviation of %s' % (
                ensemble_prediction.ensemble_prediction,
                ensemble_prediction.standard_deviation,
            ),
        }
        self._dict_writer.writerow(d)

    def event_features_created_for_source_not_trace(self, dt, event):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to determine create event features',
            'what_happened': 'data in an input event was converted to a uniform internal form',
            'info': str(event),
        }
        self._dict_writer.writerow(d)

    def event_features_created_for_source_trace(self, dt, event):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to determine create event features',
            'what_happened': 'data in an input event was converted to a uniform internal form',
            'info': str(event),
        }
        self._dict_writer.writerow(d)

    def expert_accuracies(self, dt, expert_accuracies):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to determine accuracies',
            'what_happened': 'the accuracy of all experts was determined using the last query trace print',
            'info': 'determined accuracy of %d sets of %d trained experts' % (
                len(expert_accuracies.expert_predictions),
                len(expert_accuracies.normalized_accuracies),
            )
        }
        self._dict_writer.writerow(d)

    def expert_predictions(self, dt, expert_predictions):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to predict with experts',
            'what_happened': 'the last-trained experts made predictions with the most recent feature vector',
            'info': 'made %d predictions using feature vector created at or just after %s' % (
                len(expert_predictions.expert_predictions),
                expert_predictions.feature_vector.creation_event.datetime(),
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

    def test_ensemble(self, dt, tested_ensemble, trade_type):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to test ensemble',
            'what_happened': 'all experts made predictions, then their predictions were aggregated',
            'info': 'created action %s by using %d sets of %s experts' % (
                tested_ensemble.action_identifier,
                len(tested_ensemble.list_of_trained_experts),
                trade_type,
            )
        }
        self._dict_writer.writerow(d)

    def trained_experts(self, dt, trained_experts, trade_type):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to train experts',
            'what_happened': 'new experts were trained',
            'info': 'trained %d %s experts on %d feature vectors in %f seconds' % (
                len(trained_experts.trained_models),
                trade_type,
                len(trained_experts.training_features),
                trained_experts.elapsed_wallclock_seconds,
            )
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


def ensemble_prediction_write_explanation(control, ensemble_prediction):
    'create new file explaining the ensemble prediction'
    pdb.set_trace()
    filename = 'ensemble-prediction-%s.txt' % str(ensemble_prediction.action_identifier)
    path = os.path.join(control.path['dir_out'], filename)
    with open(path, 'w') as f:
        f.write('ensemble prediction\n')
        info_format = ' %30s = %s\n'
        f.write(info_format % ('ensemble_prediction', ensemble_prediction.ensemble_prediction))
        f.write(info_format % ('standard_deviation', ensemble_prediction.standard_deviation))
        f.write(info_format % ('trade_type', ensemble_prediction.trade_type))
        f.write(info_format % ('simulated_datetime', ensemble_prediction.simulated_datetime))
        f.write(info_format % ('feature_vector', ensemble_prediction.feature_vector))
        f.write(info_format % ('action_identifier', ensemble_prediction.action_identifier))
        f.write(info_format % ('creation_event', ensemble_prediction.creation_event))
        f.write('explanation\n')
        for line in ensemble_prediction.explanation:
            f.write(' %s\n' % line)
        pdb.set_trace()
        f.write('weighted feature importances en models\n')
        for feature_name, feature_importance in ensemble_prediction.weighted_importances_en.iteritems():
            f.write(' %70s: %9.6f')
        pdb.set_trace()
        f.write('weighted feature importances rf models\n')
        for feature_name, feature_importance in ensemble_prediction.weighted_importances_rf.iteritems():
            f.write(' %70s: %9.6f')
        pdb.set_trace()
        f.write('details on accuracies (weights) of experts')
        for line in ensemble_prediction.expert_accuracies.explanation:
            f.write(' %s' % line)


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


def maybe_make_accuracies(event, expert_predictions_list, ensemble_hyperparameters, control, verbose=True):
    'return (ExpertAccuracies, errs)'
    assert isinstance(event, seven.input_event.Event)
    assert isinstance(expert_predictions_list, collections.deque)
    assert isinstance(ensemble_hyperparameters, EnsembleHyperparameters)
    if len(expert_predictions_list) == 0:
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
    # determine unnormalized weights for each model_spec
    # remember portions of computations so that the explanation is easy to create
    abs_errors = collections.defaultdict(dict)  # Dict[expert_prediction_creation_date, Dict]
    unnormalized_weights = collections.defaultdict(float)  # Dict[model_spec, float]
    # Dict[expert_prediction_creation_date_time, Dict[model_spec, unnormalized_weighted_abs_error]]
    unnormalized_weights_detail = collections.defaultdict(dict)
    expert_predictions_age_days = {}
    sum_weights = 0.0
    pdb.set_trace()
    for expert_predictions in expert_predictions_list:
        assert isinstance(expert_predictions, ExpertPredictions)
        # the accuracy is a weighted average
        # determine the weight, a function of the age of the expert prediction and the accuracy of a model
        expert_predictions_creation_datetime = expert_predictions.creation_event.datetime()
        assert ensemble_hyperparameters.weight_units == 'days'
        age_timedelta = event.datetime() - expert_predictions_creation_datetime
        age_seconds = age_timedelta.total_seconds()
        age_days = age_seconds / (24.0 * 60.0 * 60.0)
        assert age_days >= 0
        expert_predictions_age_days[expert_predictions_creation_datetime] = age_days
        for model_spec, prediction in expert_predictions.expert_predictions.iteritems():
            abs_error = abs(prediction - actual)
            abs_errors[expert_predictions_creation_datetime][model_spec] = abs_error
            weight = math.exp(-ensemble_hyperparameters.weight_temperature * abs_error * age_days)
            unnormalized_weights_detail[expert_predictions_creation_datetime][model_spec] = weight
            if verbose:
                print 'unnormalized %30s %f %f %f' % (model_spec, age_days, abs_error, weight)
            unnormalized_weights[model_spec] += weight
            sum_weights += weight
    pdb.set_trace()
    # normalize the weights for each model_spec
    normalized_weights = {}  # Dict[model_spec, float]
    sum_normalized_weights = 0.0
    for model_spec, unnormalized_weight in unnormalized_weights.iteritems():
        normalized_weight = unnormalized_weight / sum_weights
        if verbose:
            print 'accuracies %d expert prediction sets %30s unnormalized %6.2f normalized %8.6f' % (
                len(expert_predictions),
                model_spec,
                unnormalized_weights[model_spec],
                normalized_weight,
            )
        normalized_weights[model_spec] = normalized_weight
        sum_normalized_weights += normalized_weight
    pdb.set_trace()
    if verbose:
        print sum_normalized_weights
    # produce the explanation (which are lines of text that a human can read)
    lines = []
    for model_spec in unnormalized_weights.keys():
        for expert_predictions in expert_predictions_list:
            expert_predictions_creation_datetime = expert_predictions.creation_event.datetime()
            line_template = (
                'predictions from experts at %s:' +
                ' age=%0.6 days' +
                ' abs_error=%0.2f' +
                ' unnormalized_weight (across dates)=%0.6f' +
                ' normalized_weight (across dates)=%0.6f')
            line = line_template % (
                expert_predictions_creation_datetime,
                expert_predictions_age_days[model_spec],
                abs_errors[expert_predictions_creation_datetime][model_spec],
                unnormalized_weights[model_spec],
                normalized_weights[model_spec],
            )
            lines.append(line)
        pdb.set_trace()
    pdb.set_trace()

    result = ExpertAccuracies(
        actual=actual,
        creation_event=event,
        expert_predictions_list=expert_predictions_list,
        explanation=lines,
        normalized_weights=normalized_weights,
        unnormalized_weightss=unnormalized_weights,
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
    lines = []
    weighted_importances_en = collections.defaultdict(float)  # Dict[model_spec, importances])
    weighted_importances_rf = collections.defaultdict(float)  # Dict[model_spec, importances])
    for model_spec, trained_expert_model in relevant_trained_expert_models.trained_models.iteritems():
        weight = relevant_expert_accuracies.normalized_accuracies[model_spec]
        for feature_name, importance in trained_expert_model.importances.iteritems():
            if model_spec.name == 'en':
                weighted_importances_en[feature_name] += weight * importance
            elif model_spec.name == 'rf':
                weighted_importances_rf[feature_name] += weight * importance
        expert_prediction = trained_expert_model.predict([feature_vector.payload])[0]
        ensemble_prediction += weight * expert_prediction
        expert_predictions[model_spec] = expert_prediction
        line = 'model_spec %30s expert weight %8.6f expert prediction %f' % (
            model_spec,
            weight,
            expert_prediction,
        )
        if verbose:
            print line
        lines.append(line)
    if verbose:
        print 'ensemble prediction', ensemble_prediction
    # determine weighted variance of the experts' predictions
    variance = 0.0
    for model_spec, expert_prediction in expert_predictions.iteritems():
        weight = relevant_expert_accuracies.normalized_accuracies[model_spec]
        delta = expert_prediction - ensemble_prediction
        weighted_delta = weight * delta
        variance += weighted_delta * weighted_delta
    standard_deviation = math.sqrt(variance)
    if verbose:
        print 'weighted standard deviation of expert predictions', standard_deviation
    result = EnsemblePrediction(
        action_identifier=None,
        creation_event=None,
        ensemble_prediction=ensemble_prediction,
        expert_accuracies=relevant_expert_accuracies,    # the last-created expert accuracy
        explanation=lines,
        feature_vector=feature_vector,
        simulated_datetime=None,
        standard_deviation=standard_deviation,
        trade_type=None,
        trained_experts=relevant_trained_expert_models,  # the last-created trained experts
        weighted_importances_en=dict(weighted_importances_en),
        weighted_importances_rf=dict(weighted_importances_rf),
    )
    return result, None


def maybe_make_expert_predictions(control, feature_vector, trained_expert_models):
    'return (expert_predictions:Dict[model_spec, float], errs)'
    # make predictions using the last trained expert model
    if len(trained_expert_models) == 0:
        err = 'no trained expert models'
        return None, [err]
    last_trained_expert_models = trained_expert_models[-1]
    expert_predictions = {}
    print 'predicting most recently constructed feature vector using most recently trained experts'
    for model_spec, trained_model in last_trained_expert_models.trained_models.iteritems():
        predictions = trained_model.predict([feature_vector.payload])
        assert len(predictions) == 1
        prediction = predictions[0]
        expert_predictions[model_spec] = prediction
    result = {
        'expert_predictions': expert_predictions,
        'trained_experts': last_trained_expert_models,
    }
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


def maybe_test_ensemble(control,
                        ensemble_hyperparameters,
                        creation_event,
                        feature_vectors,
                        list_of_trained_experts,
                        verbose=True):
    'return (EnsemblePrediction, errs)'
    def make_actual(feature_vector):
        'return (actual, err)'
        try:
            s = feature_vector.payload['p_trace_%s' % control.arg.target]
            actual = float(s)
        except ValueError:
            err = 'string "%s" was not interpretable as a float' % s
            return None, err
        return actual, None

    assert isinstance(creation_event, seven.input_event.Event)
    assert isinstance(feature_vectors, collections.deque)
    assert isinstance(list_of_trained_experts, collections.deque)

    if len(feature_vectors) < 2:
        err = 'need at least 2 feature vectors, have %d' % len(feature_vectors)
        return None, [err]

    if len(list_of_trained_experts) == 0:
        err = 'had 0 trained sets of experts, need at least 1'
        return None, [err]

    actual, err = make_actual(feature_vectors[-1])
    if err is not None:
        return None, [err]

    query_feature_vector = feature_vectors[-2]
    query_features = query_feature_vector.payload

    # predict with the experts, tracking errors and building report lines
    lines = []

    def line(s):
        if verbose:
            print s
        lines.append(s)

    testing_datetime = creation_event.datetime()

    # determine unnormalized weights for each expert's prediction using the query features
    line('using %d sets of trained experts to predict query at %s with actual target %s' % (
        len(list_of_trained_experts),
        testing_datetime,
        actual,
    ))

    # Dict[trained_expert_index, value]
    ages_days = {}

    # Dict[trained_expert_index, Dict[model_spec, value]]
    absolute_errors = collections.defaultdict(dict)
    expert_predictions = collections.defaultdict(dict)
    unnormalized_weights = collections.defaultdict(dict)

    sum_all_unnormalized_weights = 0.0
    for trained_experts_index, trained_experts in enumerate(list_of_trained_experts):
        training_datetime = trained_experts.creation_event.datetime()
        age_timedelta = testing_datetime - training_datetime
        age_days = age_timedelta.total_seconds() / (60.0 * 60.0 * 24.0)
        print training_datetime, testing_datetime, age_days
        ages_days[trained_experts_index] = age_days
        assert ensemble_hyperparameters.weight_units == 'days'
        line(' determining accuracy of trained expert set # %3d trained at %s, %f days ago' % (
            trained_experts_index + 1,
            training_datetime,
            age_days,
        ))
        for model_spec, trained_expert in trained_experts.trained_models.iteritems():
            predictions = trained_expert.predict([query_features])
            assert len(predictions) == 1
            prediction = predictions[0]
            expert_predictions[trained_experts_index][model_spec] = prediction

            error = prediction - actual
            absolute_error = abs(error)
            absolute_errors[trained_experts_index][model_spec] = absolute_error

            unnormalized_weight = math.exp(-ensemble_hyperparameters.weight_temperature * age_days * absolute_error)
            unnormalized_weights[trained_experts_index][model_spec] = unnormalized_weight

            sum_all_unnormalized_weights += unnormalized_weight

    # determine normalized the weights and the ensemble prediction and the weighted importances
    line('contribution of sets of experts to the ensemble prediction; actual = %6.2f' % actual)
    weighted_importances = collections.defaultdict(lambda: collections.defaultdict(float))
    ensemble_prediction = 0.0
    normalized_weights = collections.defaultdict(dict)
    for trained_experts_index, trained_experts in enumerate(list_of_trained_experts):
        line(
            'expert set %3d' % (trained_experts_index + 1) +
            ' trained on %d samples' % (len(trained_experts.training_features)) +
            ' at %s' % trained_experts.creation_event.datetime() +
            ''
        )
        for model_spec in sorted(trained_experts.trained_models.keys()):
            trained_expert = trained_experts.trained_models[model_spec]
            normalized_weight = unnormalized_weights[trained_experts_index][model_spec] / sum_all_unnormalized_weights
            normalized_weights[trained_experts_index][model_spec] = normalized_weight

            expert_prediction = expert_predictions[trained_experts_index][model_spec]
            contribution = normalized_weight * expert_prediction
            ensemble_prediction += contribution

            line(
                ' %-30s' % model_spec +
                ' predicted %6.2f' % expert_predictions[trained_experts_index][model_spec] +
                ' abs error %6.2f' % absolute_errors[trained_experts_index][model_spec] +
                ' weights unnormalized %8.6f' % unnormalized_weights[trained_experts_index][model_spec] +
                ' normalized %8.6f' % normalized_weight +
                ' contr %10.6f' % contribution +
                ''
            )

            # weighted importances
            model_name = model_spec.name
            for feature_name, feature_importance in trained_expert.importances.iteritems():
                weighted_importances[model_name][feature_name] += normalized_weight * feature_importance

    # create explanation lines for importances
    line('weighted feature importances (suppressing zero importances)')
    for model_name in sorted(weighted_importances.keys()):
        model_importances = weighted_importances[model_name]
        line(' for models %s' % model_name)
        print model_name, model_importances
        for feature_name in sorted(model_importances.keys()):
            weighted_feature_importance = model_importances[feature_name]
            if weighted_feature_importance != 0.0:
                line(
                    '  feature name %70s' % feature_name +
                    ' weighted importance %10.6f' % weighted_feature_importance +
                    ''
                )

    # determine weighted standard deviation of the experts
    line('standard deviation of predictions of experts relative to the ensemble prediction')
    weighted_variance = 0.0
    for trained_experts_index, trained_experts in enumerate(list_of_trained_experts):
        line(
            ' expert set %3d' % (trained_experts_index + 1) +
            ''
        )
        for model_spec in trained_experts.trained_models.keys():
            weight = normalized_weights[trained_experts_index][model_spec]
            expert_prediction = expert_predictions[trained_experts_index][model_spec]
            delta = expert_prediction - ensemble_prediction
            weighted_delta = weight * delta
            weighted_variance += weighted_delta * weighted_delta
            line(
                '  normalized weight %8.6f' % weight +
                ' expert prediction %6.2f' % expert_prediction +
                ' ensemble prediction %6.2f' % ensemble_prediction +
                ' weighted delta %10.6f' % weighted_delta +
                ''
            )
    standard_deviation = math.sqrt(weighted_variance)

    result = EnsemblePrediction(
        action_identifier=None,          # caller will supply
        creation_event=creation_event,
        elapsed_wallclock_seconds=None,  # caller will supply
        ensemble_prediction=ensemble_prediction,
        explanation=lines,
        importances=weighted_importances,
        feature_vector_for_actual=feature_vectors[-1],
        feature_vector_for_query=query_feature_vector,
        list_of_trained_experts=list_of_trained_experts,
        standard_deviation=standard_deviation,
        simulated_datetime=None,        # caller will supply
    )
    return result, None


def maybe_train_experts(control,
                        ensemble_hyperparameters,
                        creation_event,
                        feature_vectors,
                        last_expert_training_time,
                        simulated_datetime,
                        verbose=False):
    'return ((trained_models, training_features, training_targets), errs)'
    assert isinstance(creation_event, seven.input_event.Event)
    assert isinstance(feature_vectors, collections.deque)
    assert isinstance(last_expert_training_time, datetime.datetime)

    if (simulated_datetime - last_expert_training_time) < ensemble_hyperparameters.min_timedelta_between_training:
        err = '%s had not passed since last training' % ensemble_hyperparameters.min_timedelta_between_training
        return None, [err]

    if len(feature_vectors) < 2:
        err = 'need at least 2 feature vectors, had %d' % len(feature_vectors)
        return None, [err]

    def dt(index):
        'return datetime of the trace event for the primary cusip'
        return feature_vectors[index].payload['id_p_trace_event'].datetime()

    training_features = []
    training_targets = []
    err = None
    print 'creating training data'
    for i in xrange(0, len(feature_vectors) - 1):
        training_features.append(feature_vectors[i].payload)
        features_datetime = dt(i)
        next_index = i + 1
        while (next_index < len(feature_vectors)) and dt(next_index) <= features_datetime:
            next_index += 1
        if next_index == len(feature_vectors):
            err = 'no future trade in same primary cusip to find target value'
            return None, [err]
            break
        target_feature_vector = feature_vectors[next_index]
        training_targets.append(target_feature_vector.payload['p_trace_%s' % control.arg.target])
    if len(training_targets) == 0:
        err = 'no training targets were found for the feature vectors'
        return None, [err]

    assert len(training_features) == len(training_targets)

    print 'training experts on %d training samples' % len(training_features)
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
    result = TrainedExperts(
        creation_event=creation_event,
        elapsed_wallclock_seconds=None,  # caller sets this
        trained_models=trained_models,
        training_features=training_features,
        training_targets=training_targets,
    )
    return result, None


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

    state_all_tested_ensembles = TypedDequeDict(
        item_type=EnsemblePrediction,
        maxlen=ensemble_hyperparameters.max_n_ensemble_predictions,
    )
    # state_expert_accuracies = TypedDequeDict(
    #     item_type=ExpertAccuracies,
    #     maxlen=ensemble_hyperparameters.max_n_expert_accuracies,
    # )
    # state_expert_predictions = TypedDequeDict(
    #     item_type=ExpertPredictions,
    #     maxlen=ensemble_hyperparameters.max_n_expert_predictions,
    # )
    state_all_feature_vectors = TypedDequeDict(
        item_type=FeatureVector,
        maxlen=ensemble_hyperparameters.max_n_feature_vectors,
    )
    state_all_trained_sets_of_experts = TypedDequeDict(
        item_type=TrainedExperts,
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
    output_signals = Signals(control.path['out_signal'])
    output_trace = Trace(control.path['out_trace'])
    seven.logging.verbose_info = False
    seven.logging.verbose_warning = False
    simulated_time = SimulatedTime()
    action_identifiers = ActionIdentifiers()
    events_at = collections.Counter()
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

        events_at[event.datetime()] += 1

        if event.date() < start_events:
            counter['events ignored'] += 1
            continue

        # track events that occured at the exact same moment
        # In the future, such events need to be treated as a cluster and process together

        simulated_time.see_event(event)
        seven.logging.verbose_info = True
        seven.logging.verbose_warning = True
        counter['events processed'] += 1
        print 'processing event # %d: %s' % (counter['events processed'], event)

        # print summary of global state
        if event.date() >= start_predictions:
            format_template = '%30s: %s'
            print format_template % ('ensemble_predictions', state_all_tested_ensembles)
            # print format_template % ('expert_accuracies', state_expert_accuracies)
            # print format_template % ('expert_predictions', state_expert_predictions)
            print format_template % ('all_feature_vectors', state_all_feature_vectors)
            print format_template % ('trained_expert_models', state_all_trained_sets_of_experts)

            print 'processed %d events in %0.2f wallclock minutes' % (
                counter['events processed'] - 1,
                control.timer.elapsed_wallclock_seconds() / 60.0,
            )

        # attempt to extract event-features from the event
        # keep track of the current OTR cusip
        if True:  # convert to event features for selected event types
            # set current_otr_cusip for use below
            if event.source == 'trace':
                cusip = event.cusip()
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
                output_trace.event_features_created_for_source_trace(simulated_time.datetime, event)
                if cusip == control.arg.cusip:
                    output_signals.trace_event(event)
            else:
                # handle all non-trace input events
                source = event.source
                if source not in event_feature_makers.not_cusip:
                    print source
                    print event
                    print event.event_feature_maker_class
                    event_feature_makers.not_cusip[source] = event.event_feature_maker_class(control.arg, event)
                event_features, errs = event_feature_makers.not_cusip[source].make_features(event)
                if errs is not None:
                    irregularity.event_not_usable(errs, event)
                    counter['source %s events that were skipped' % source] += 1
                    continue
                event_feature_values.not_cusip[source] = event_features
                counter['event-features sets created source %s' % source] += 1
                simulated_time.handle_event()
                output_trace.event_features_created_for_source_not_trace(simulated_time.datetime, event)
                if source == 'liq_flow_on_the_run':
                    if event_features['id_liq_flow_on_the_run_primary_cusip'] == control.arg.cusip:
                        if event_features['id_liq_flow_on_the_run_otr_cusip'] == current_otr_cusip:
                            pdb.set_trace()
                            irregularity.skipped_event('otr event with no change in otr cusip', event)
                            continue
                        else:
                            current_otr_cusip = event_features['id_liq_flow_on_the_run_otr_cusip']
                    else:
                        irregularity.skipped_event('otr event for non-query cusip', event)
                        continue
                else:
                    irregularity.skipped_event('not a liq flow on the run event', event)
                    continue

        # do no other work until we reach the start date
        if event.date() < start_predictions:
            msg = 'skipped more than event feature extraction because before start date %s' % control.arg.start_date
            counter[msg] += 1
            continue

        counter['events on or after start-predictions date'] += 1
        if True:  # maybe create feature vector
            # The feature vector incorporates information from every event that has made it this far
            # So any event could cause a new feature vector to be created
            # NOTE: upstream event sources that send identical events with simply different time stamps
            # will cause this code to generate multiple events with identical feature sets
            # MAYBE: find this problems in the event-feature creation code, which could check for duplicates.
            # NOTE: that was done for the lqd_flow_on_the_run already.
            # NOTE: Once features apart from the CUSIP-level features are used in the feature vecctor,
            # this code will need to be modified to detemine when those other event features have changes.
            # For that, the simulated datetime of the last change to the non-cusip event features may work.

            if control.arg.cusip not in event_feature_values.cusip:
                irregularity.no_feature_vector(
                    'no event features for primary cusip %s' % control.arg.cusip,
                    event,
                )
                continue
            if current_otr_cusip not in event_feature_values.cusip:
                irregularity.no_feature_vector(
                    'no event feaures for OTR cusip %s' % current_otr_cusip,
                    event,
                )
                continue
            if event.source == 'trace':
                event_cusip = event.cusip()
                if event_cusip == control.arg.cusip or event_cusip == current_otr_cusip:
                    pass
                else:
                    irregularity.no_feature_vector(
                        'trace event for cusip %s, not primary not OTR' % event_cusip,
                        event,
                    )
                    continue
            feature_vector = FeatureVector(
                creation_event=event,
                otr_cusip_event_features=event_feature_values.cusip[current_otr_cusip],
                primary_cusip_event_features=event_feature_values.cusip[control.arg.cusip],
            )
            state_all_feature_vectors.append(
                feature_vector.reclassified_trade_type,
                feature_vector,
            )
            simulated_time.handle_event()
            output_trace.new_feature_vector_created(
                simulated_time.datetime,
                feature_vector,
            )
            counter['feature vectors created'] += 1

        if False:  # maybe determine accuracy of all sets of experts on this trace print event
            pass
            # if event.is_trace_print_with_cusip(control.arg.cusip):   # maybe make accuracies
            #     # try to determine the accuracies of all of the experts
            #     # based on the trace print event actual target value
            #     result, errs = maybe_make_accuracies(
            #         event,
            #         state_expert_predictions[event.reclassified_trade_type()],
            #         ensemble_hyperparameters,
            #         control,
            #     )
            #     if errs is not None:
            #         for err in errs:
            #             irregularity.no_accuracy(err, event)
            #     else:
            #         assert isinstance(result, ExpertAccuracies)
            #         state_expert_accuracies.append(
            #             event.reclassified_trade_type(),
            #             result,
            #         )
            #         simulated_time.handle_event()
            #         output_trace.expert_accuracies(
            #             simulated_time.datetime,
            #             result,
            #         )
            #         counter['accuracies determined'] += 1
            # else:
            #     irregularity.no_accuracy('event is not from a trace print for the query cusip', event)

        if False:  # maybe test the ensemble model
            print 'maybe test the ensemble model'
            for trade_type in ('B', 'S'):
                start_wallclock = datetime.datetime.now()
                result, errs = maybe_test_ensemble(
                    control=control,
                    creation_event=event,
                    ensemble_hyperparameters=ensemble_hyperparameters,
                    feature_vectors=state_all_feature_vectors[trade_type],
                    list_of_trained_experts=state_all_trained_sets_of_experts[trade_type],
                )
                if errs is not None:
                    for err in errs:
                        irregularity.no_ensemble_prediction(err, event)
                else:
                    simulated_time.handle_event()
                    action_identifier = action_identifiers.next_identifier(simulated_time.datetime)
                    tested_ensemble = result._replace(
                        action_identifier=action_identifier,
                        elapsed_wallclock_seconds=(datetime.datetime.now() - start_wallclock).total_seconds(),
                        simulated_datetime=simulated_time.datetime,
                    )
                    state_all_tested_ensembles.append(
                        trade_type,
                        tested_ensemble,
                    )
                    output_actions.ensemble_prediction(tested_ensemble, trade_type)
                    output_signals.ensemble_prediction(tested_ensemble, trade_type)
                    output_trace.test_ensemble(
                        simulated_time.datetime,
                        tested_ensemble,
                        trade_type,
                    )
                    # write the explanation (which is lines of plain text)
                    explanation_filename = ('ensemble_prediction-%s.txt' % action_identifier)
                    explanation_path = os.path.join(control.path['dir_out'], explanation_filename)
                    with open(explanation_path, 'w') as f:
                        for line in tested_ensemble.explanation:
                            f.write(line)
                            f.write('\n')
                    counter['tested %s ensemble' % trade_type] += 1

        if False:  # maybe test (predict with) the most recently-trained experts
            pass
            # if have_new_feature_vector:
            #     # attempt to make predictions will the last-trained set of experts
            #     result, errs = maybe_make_expert_predictions(
            #         control,
            #         feature_vector,  # use the feature vector just constructede
            #         state_trained_experts[event.maybe_reclassified_trade_type()],  # use just S or B trade types
            #     )
            #     if errs is not None:
            #         for err in errs:
            #             irregularity.no_expert_predictions(err, event)
            #     else:
            #         expert_predictions = ExpertPredictions(
            #             creation_event=event,
            #             feature_vector=feature_vector,
            #             expert_predictions=result['expert_predictions'],
            #             trained_experts=['trained_experts'],  # last trained experts
            #         )
            #         state_expert_predictions.append(
            #             event.maybe_reclassified_trade_type(),
            #             expert_predictions,
            #         )
            #         simulated_time.handle_event()
            #         output_trace.expert_predictions(
            #             simulated_time.datetime,
            #             expert_predictions,
            #         )
            #         counter['experts tested'] += 1
            # else:
            #     irregularity.no_expert_predictions('no different feature vector', event)

        if False:  # maybe train the experts
            for trade_type in ('B', 'S'):
                start_wallclock = datetime.datetime.now()
                result, errs = maybe_train_experts(
                    control=control,
                    ensemble_hyperparameters=ensemble_hyperparameters,
                    creation_event=event,
                    feature_vectors=state_all_feature_vectors[trade_type],
                    last_expert_training_time=last_expert_training_time,
                    simulated_datetime=simulated_time.datetime,
                )
                if errs is not None:
                    for err in errs:
                        irregularity.no_training(err, event)
                else:
                    trained_experts = result._replace(
                        elapsed_wallclock_seconds=(datetime.datetime.now() - start_wallclock).total_seconds()
                    )
                    state_all_trained_sets_of_experts.append(
                        trade_type,
                        trained_experts,
                    )
                    simulated_time.handle_event()
                    output_trace.trained_experts(
                        simulated_time.datetime,
                        trained_experts,
                        trade_type,
                    )
                    last_expert_training_time = simulated_time.datetime
                    counter['sets of %s experts trained' % trade_type] += 1

        if False and counter['sets of B experts trained'] >= 1:
            print 'for now, stopping base on number of accuracies determined'
            break

        if counter['tested B ensemble'] >= 1:
            print 'for now, stopping based on number of ensembles tested'
            break

        if control.arg.test:
            stop = False
            for trade_type in ('B', 'S'):
                stop = len(state_all_tested_ensembles[trade_type] > 1)
            if stop:
                print 'break out of event loop because of --test'
                print 'NOTE: output is in a TEST directory'
                break

        gc.collect()

    output_actions.close()
    output_importances.close()
    output_signals.close()
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
    print
    print '*******************************************'
    for trade_type in ('B', 'S'):
        print 'ended with %d %s feature vectors' % (
            len(state_all_feature_vectors[trade_type]),
            trade_type,
        )
        print 'ended with %d %s trained sets of experts' % (
            len(state_all_trained_sets_of_experts[trade_type]),
            trade_type,
        )
        print 'ended with %d %s tested ensemble models' % (
            len(state_all_tested_ensembles[trade_type]),
            trade_type,
        )
    print
    print '*********************'
    print 'events at same time with non-zero hour:minute:second'
    for k in sorted(events_at.keys()):
        v = events_at[k]
        if k.hour == 0 and k.minute == 0 and k.second == 0:
            pass
        else:
            if v > 1:
                print '%30s: %d' % (k, v)
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
