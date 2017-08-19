'''create predictions and test their accuracy

APPROACH: see the file test_traing.org in the same directory as this file.

INVOCATION
  python test_train.py {issuer} {cusip} {target} {hpset}
    {start_events} {start_predictions} {stop_predictions}
    [--config {config}] [--debug] [--test] [--trace]
where
 issuer the issuer (ex: AAPL)
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 start_events: YYYY-MM-DD is the first date that we examine an event.
   It should be on the start of a calendar quarter, because
     the events include the fundamental for the issuer
     fundamentals tend to be published once a quarter
  It should be about a quarter before the start_prediction date, so that feature vectors
     can be accumulated before the predictions start
 start_predictions: YYYY-MM-DD is the first date on which we attempt to test and train
 stop_predictions: YYYY-MM-DD is the last date on which we attempt to test and train
 --config {config} means to read file {config} and use its contents to update the internal
   control variable. This facility allows the developer to further control the application.
   One use is to change the location of certain input files, which might be useful when
   debugging production problems. See the function make_control() for how this invocation
   parameter is used in this program.
 --debug means to call pdb.set_trace() instead of raisinng an exception, on calls to logging.critical()
   and logging.error()
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
  python test_train.py AAPL 037833AJ9 oasspread grid5 2017-04-01 2017-07-01 2016-07-05 --debug # run until end of events
  python test_train.py AAPL 037833AJ9 oasspread grid5 2015-10-01 2017-06-01 2016-06-01 --debug --test # holiday, nothing to predict
  python test_train.py AAPL 037833AJ9 oasspread grid5 2015-10-01 2017-06-01 2016-06-04 --debug --test # through Monday

See build.py for input and output files.

APPROACH:
 1. Construct two models
    train_and_test_B = model for oasspread B predictions := oasspread_predictions('B')
    train_and_test_S = model for oasspread S predictions := oasspread_predictions('S')
 2. Repeat until no more events. Each event has a source file, date time, and fields from a record
    in its souce file.
    a. Read next event. It is the one with the lowest datetime from all the source files.
    b. If its datetime is before arg {start_events}, ignore it and go to step a.
    c. If its datetime is after arg {stop_predictions}, break out of this loop and perform
       reporting on what happened.
    b. Otherwise the event datetime is in [{start_predictions], {stop_prediction}}]. Do:
       (1) Assemble any attributes from the event. Some of the attributes may depend on seeing
           several events from the same source. For example, an attribute from the trace print
           source may be the change in the oasspread from the last trace print.
       (2) Mutate the fields in the feature vector corresponding to the event source.
       (3) If all the fields in the feature vector are known, fully construct the feature vector.
           The construction of features in the feature vector is iterative:
           (a) Start with the attributes from the events. Those become "level 1 features."
           (b) Build "level 2 features" from the level 1 features. For example, if one event
               source is the 10Q debt and another is 10Q equity, a level 2 feature
               could be the debt-to-equity ratio.
           (c) Continue to build "level K features" iterating K. For now, K = 1.
       (4) If the feature vector has any changed features from the last iteration, test and train
           using it.
           (a) If the last primary cusip event was a reclassified B trade, use test_and_train_B.
               If the last primary cusip event was a reclassified S trade, use test_and_train_S.
               Otherwise, there is an error, as for now, trace print events (which are used to
               identify the primary cusip) or ignored if they were not reclassified as B or S.
           (b) Test by predicting its target value. Use the just previously-created feature vector
               as the query. Skip this step if there are no trained experts. If there are trained
               experts, predict with the N most recent sets of trained experts and blend their
               predictions to produce the ensemble prediction and standard deviation for that
               prediction.
           (c) Train another set of experts using the existing training data augmented by the
               new feature vector. Skip this step if there is not enough training data.

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
import errno
import gc
import heapq
import json
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

# from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.accumulators
import seven.arg_type
import seven.build
import seven.Event
import seven.EventAttributes
import seven.event_readers
import seven.HpGrids
import seven.logging
import seven.make_event_attributes
import seven.models2
import seven.read_csv

pp = pprint


class Control(object):
    def __init__(self, arg, path, random_seed, timer):
        self.arg = arg
        self.path = path
        self.random_seed = random_seed
        self.timer = timer

    def __repr__(self):
        return 'Control(arg=%s, n path=%d, random_seed=%f, timer=%s)' % (
            self.arg,
            len(self.path),
            self.random_seed,
            self.timer,
        )

    def new_with_path(self, new_logical_name, new_location):
        result = copy.copy(self)
        result.path[new_logical_name] = new_location
        return result


def make_control(argv):
    'return a Control'
    parser = argparse.ArgumentParser()
    parser.add_argument('issuer', type=seven.arg_type.issuer)
    parser.add_argument('cusip', type=seven.arg_type.cusip)
    parser.add_argument('target', type=seven.arg_type.target)
    parser.add_argument('hpset', type=seven.arg_type.hpset)
    parser.add_argument('start_events', type=seven.arg_type.date_quarter_start)
    parser.add_argument('start_predictions', type=seven.arg_type.date)
    parser.add_argument('stop_predictions', type=seven.arg_type.date)
    parser.add_argument('--config', action='store')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')

    arg = parser.parse_args(argv[1:])

    seconds_in_90_days = 90.0 * 24.0 * 60.0 * 60.0
    assert (arg.start_predictions - arg.start_events).total_seconds() >= seconds_in_90_days
    assert arg.start_predictions <= arg.stop_predictions

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
        arg.stop_predictions,
        test=arg.test,
    )
    applied_data_science.dirutility.assure_exists(paths['dir_out'])
    applied_data_science.dirutility.assure_exists(os.path.join(paths['dir_out'], 'ensemble-predictions'))

    timer = Timer()

    control1 = Control(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=timer,
    )
    control = (
        control1 if arg.config is None else
        config(control1, arg.config)
    )
    return control


def config(control, config_path):
    'return updated control object'
    with open(config_path) as f:
        j = json.load(f)
    result = copy.copy(control)
    for k, v in j.iteritems():
        if k == 'path':
            for logical_name, location in v.iteritems():
                result = result.new_with_path(logical_name, location)
        else:
            seven.logging.critical('config: unrecognized key: %s' % k)
    return result


class ExpertAccuracy(object):
    def __init__(self,
                 actual,                # float
                 absolute_errors,       # Dict[model_spec, float]
                 ensemble_prediction,   # float, for one set of trained experts
                 explanation,           # List[str]
                 importances,           # Dict[model_spec, Dict[feature_name, weighted_importance]]
                 normalized_weights,    # Dict[model_spec, noramlized_weight: float]
                 predictions,           # Dict[model_spec, float]
                 standard_deviation,    # float, for one set of trained experts
                 ):
        assert isinstance(actual, float)
        assert isinstance(absolute_errors, dict)
        assert isinstance(ensemble_prediction, float)
        assert isinstance(explanation, list)
        assert isinstance(importances, dict)
        assert isinstance(normalized_weights, dict)
        assert isinstance(predictions, dict)
        assert isinstance(standard_deviation, float)
        self.actual = actual
        self.absolute_errors = absolute_errors
        self.ensemble_prediction = ensemble_prediction
        self.explanation = explanation
        self.importances = importances
        self.normalized_weights = normalized_weights
        self.predictions = predictions
        self.standard_deviation = standard_deviation

    def __repr__(self):
        return 'ExpertAccuracies(actual=%0.2f, ensemble_prediction=%0.2f, standard_deviation=%0.2f)' % (
            self.actual,
            self.ensemble_prediction,
            self.standard_deviation,
        )


class EnsemblePrediction(object):
    def __init__(self,
                 action_identifier,
                 actual,
                 creation_event,
                 elapsed_wallclock_seconds,  # time to make expert predictions and ensemble prediction
                 ensemble_prediction,        # across the sets of trained experts
                 expert_accuracies,          # List[ExpertAccuracy], oldest to youngest
                 explanation,
                 feature_vector_for_actual,
                 feature_vector_for_query,
                 importances,                # Dict[model_spec.name, Dict[feature_name, weighted_importance]]
                 list_of_trained_experts,    # List[TrainedExpert], ordered from oldest to yougest
                 simulated_datetime,
                 standard_deviation,
                 ):
        self.action_identifier = action_identifier
        self.actual = actual
        self.creation_event = creation_event
        self.elapsed_wallclock_seconds = elapsed_wallclock_seconds
        self.ensemble_prediction = ensemble_prediction
        self.expert_accuracies = expert_accuracies
        self.explanation = explanation
        self.feature_vector_for_actual = feature_vector_for_actual
        self.feature_vector_for_query = feature_vector_for_query
        self.importances = importances
        self.list_of_trained_experts = list_of_trained_experts
        self.simulated_datetime = simulated_datetime
        self.standard_deviation = standard_deviation

    def __repr__(self):
        return 'EnsemblePrediction(simulated_datetime=%s, ensemble_prediction=%0.6f, standard_deviation=%0.6f' % (
            self.simulated_datetime,
            self.ensemble_prediction,
            self.standard_deviation,
            )


class TrainedExperts(object):
    def __init__(self,
                 creation_event,
                 elapsed_wallclock_seconds,
                 simulated_datetime,
                 trained_models,
                 training_features,
                 training_targets,
                 ):
        self.creation_event = creation_event
        self.elapsed_wallclock_seconds = elapsed_wallclock_seconds
        self.simulated_datetime = simulated_datetime
        self.trained_models = trained_models
        self.training_features = training_features
        self.training_targets = training_targets

    def __hash__(self):
        return hash(self.creation_event)

    def __repr__(self):
        return 'TrainedExperts(trained at %s)' % self.simulated_datetime


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
        self.max_n_feature_vectors = 250  # max n_trades_back for grid5 is 200
        self.max_n_trained_sets_of_experts = 3
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
                 creation_datetime,
                 creation_event,
                 payload,
                 ):
        assert isinstance(creation_datetime, datetime.datetime)
        # these fields are part of the API
        self.creation_datetime = creation_datetime
        self.creation_event = copy.copy(creation_event)
        self.payload = payload
        # TODO: implement cross-product of the features from the events
        # for example, determine the debt to equity ratio

    def reclassified_trade_type(self):
        return self.creation_event.reclassified_trade_type()

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
        return 'FeatureVector(creation_event=%s, n features=%d, created=%s)' % (
            self.creation_event,
            len(self.payload),
            self.creation_datetime,
        )


class FeatureVectorMaker(object):
    def __init__(self, invocation_args):
        self._invocation_args = invocation_args

        self._event_attributes_cusip_otr = None
        self._event_attributes_cusip_primary = None

    def all_features(self):
        'return dict with all the features'
        # check that there are no duplicate feature names
        def append(all_features, tag, attributes):
            for k, v in attributes.value.iteritems():
                if k.startswith('id_'):
                    k_new = 'id_%s_%s' % (tag, k[3:])
                else:
                    k_new = '%s_%s' % (tag, k)
                assert k_new not in all_features
                all_features[k_new] = v

        all_features = {}
        append(all_features, 'otr', self._event_attributes_cusip_otr)
        append(all_features, 'p', self._event_attributes_cusip_primary)
        return all_features

    def have_all_event_attributes(self):
        'return True or False'
        return (
            self._event_attributes_cusip_otr is not None and
            self._event_attributes_cusip_primary is not None)

    def missing_attributes(self):
        'return str of missing attributes (possibly empty string)'
        return (
            ' cusip_otr' if self._event_attributes_cusip_otr is None else '' +
            ' cusip_primary' if self.event_attributes_cusip_primary is None else '' +
            ''
        )

    def update_cusip_otr(self, event, event_attributes):
        assert isinstance(event_attributes, seven.EventAttributes.EventAttributes)
        self._event_attributes_cusip_otr = event_attributes

    def update_cusip_primary(self, event, event_attributes):
        assert isinstance(event_attributes, seven.EventAttributes.EventAttributes)
        self._event_attributes_cusip_primary = event_attributes


class Importances(object):
    # deprecated: The importances are being incorporated into the signals
    # TODO: remove me once signals prints importances
    def __init__(self, path):
        self._path = path
        self._field_names = [
            'event_datetime', 'event_id',
            'model_name', 'feature_name', 'accuracy_weighted_feature_importance',
        ]
        self._n_rows_written = 0

        os.remove(path)
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

    def no_attributes(self, msg, event):
        self._oops('unable to make attributes from event', msg, event)

    def no_attribute_maker(self, msg, event):
        self._oops('no attribute maker', msg, event)

    def no_actual(self, msg, event):
        self._oops('unable to determine actual target value', msg, event)

    def no_event(self, msg, event):
        self._oops('event reader not able to create an event', msg, event)

    def no_expert_predictions(self, msg, event):
        self._oops('unable to create expert predictions', msg, event)

    def no_expert_training(self, msg, event):
        self._oops('unable to train the experts', msg, event)

    def no_feature_vector(self, msg, event):
        self._oops('feature vector not yet created', msg, event)

    def no_ensemble_prediction(self, msg, event):
        self._oops('unable to create ensemble prediction', msg, event)

    def no_importances(self, msg, event):
        self._oops('unable to creating importances', msg, event)

    def no_test(self, msg, event):
        self._oops('did not test', msg, event)

    def no_testing(self, msg, event):
        self._oops('did not test new feature vector', msg, event)

    def no_train(self, msg, event):
        self._oops('did not train', msg, event)

    def skipped_event(self, msg, event):
        self._oops('event skipped', msg, event)

    def _oops(self, what_happened, message, event):
        seven.logging.info('%s: %s: %s' % (what_happened, message, event))
        self.counter[message] += 1


class Output(object):
    'output to a CSV file using a dictionary writer'
    # close and reopen file after every record, so that the file is not truncated
    # if the program exits abnormally
    def __init__(self, path, field_names):
        self._path = path
        self._field_names = field_names

        self._dict_writer = None
        self._file = None
        self._n_rows_written = 0

        self._silently_remove(path)
        self._open()

    def close(self):
        self._file.close()

    def _open(self):
        self._file = open(self._path, 'ab')  # the b is needed for Windows
        self._dict_writer = csv.DictWriter(
            self._file,
            self._field_names,
            lineterminator='\n',
        )
        if self._n_rows_written == 0:
            self._dict_writer.writeheader()

    def _silently_remove(self, path):
        'remove file, if it exists'
        # ref: https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
        try:
            os.remove(path)
        except OSError as e:
            if e.errno != errno.ENOENT:  # no such file or directory
                raise                    # re-raise the exception
            pass  # ignore file does not exist problem (and others)

    def _writerow(self, row):
        'write row'
        self._dict_writer.writerow(row)
        self._n_rows_written += 1

        self.close()
        self._open()


class OutputActions(Output):
    'actions are outputs of the machine learning to upstream processes'
    # not the diagnostics, at least for now
    # the actions need to be in a file that an upstream process can tail
    def __init__(self, path):
        super(OutputActions, self).__init__(
            path=path,
            field_names=[
                'action_id',
                'action_type',
                'action_value',
            ],
        )

    def ensemble_prediction(self, ep, trade_type):
        assert isinstance(ep, EnsemblePrediction)
        self._writerow({
            'action_id': ep.action_identifier,
            'action_type': 'ensemble prediction %s' % trade_type,
            'action_value': ep.ensemble_prediction,
        })
        self._writerow({
            'action_id': ep.action_identifier,
            'action_type': 'ensemble prediction standard deviation %s' % trade_type,
            'action_value': ep.standard_deviation,
        })


class OutputExperts(Output):
    def __init__(self, path):
        super(OutputExperts, self).__init__(
            path=path,
            field_names=[
                'event_datetime',
                'creation_event',
                'cusip',
                'last_trained_expert_model_spec',
                'expert_normalized_weight',
                'target_actual',
                'target_predicted',
                'absolute_error',
            ]
        )

    def expert_accuracy(self, ensemble_prediction):
        assert isinstance(ensemble_prediction, EnsemblePrediction)
        expert_accuracies = ensemble_prediction.expert_accuracies[-1]  # report on the most recently created
        normalized_weights = expert_accuracies.normalized_weights  # Dict[model_spec, normalized_weight]
        sorted_normalized_weights = self._sorted_by_decreasing_value(normalized_weights)
        actual = expert_accuracies.actual
        for d in sorted_normalized_weights:
            model_spec = d['key']
            normalized_weight = d['value']
            # for model_spec, normalized_weight in expert_accuracies.normalized_weights.iteritems():
            predicted = expert_accuracies.predictions[model_spec]
            row = {
                'event_datetime': ensemble_prediction.creation_event.datetime(),
                'creation_event': '%s' % ensemble_prediction.creation_event,
                'cusip': ensemble_prediction.creation_event.cusip(),
                'last_trained_expert_model_spec': model_spec,
                'expert_normalized_weight': normalized_weight,
                'target_actual': actual,
                'target_predicted': predicted,
                'absolute_error': abs(actual - predicted),
            }
            self._writerow(row)

    def _sorted_by_decreasing_value(self, d):
        'return d:Dict sorted by decreasing value'
        # ref: https://stackoverflow.com/questions/20577840/python-dictionary-sorting-in-descending-order-based-on-values
        result = []
        for key, value in sorted(d.iteritems(), key=lambda (k, v): (v, k), reverse=True):
            result.append({
                'value': value,
                'key': key,
            })
        return result


class OutputImportances(Output):
    def __init__(self, path, n_most_important_features):
        self._n_most_important_features = n_most_important_features
        super(OutputImportances, self).__init__(
            path=path,
            field_names=[
                'simulated_datetime',
                'importances_model_family',
                'importances_feature_name',
                'importances_weight'
            ],
        )

    def ensemble_prediction(self, ensemble_prediction, reclassified_trade_type):
        most_important = self._most_important_features(ensemble_prediction)
        for model_name, feature_importance_list in most_important.iteritems():
            for feature, importance in feature_importance_list:
                row = {
                    'simulated_datetime': ensemble_prediction.simulated_datetime,
                    'importances_model_family': model_name,
                    'importances_feature_name': feature,
                    'importances_weight': importance,
                }
                self._writerow(row)

    def _most_important_features(self, ensemble_prediction):
        assert isinstance(ensemble_prediction, EnsemblePrediction)
        assert self._n_most_important_features >= 0
        result = {}
        for model_name in ('en', 'rf'):
            names_importances = dict(ensemble_prediction.importances[model_name])
            sorted_names_importances = sorted(
                names_importances.items(),       # list of pairs (k, v)
                key=lambda item: abs(item[1]),   # en importances can be negative
                reverse=True,
            )
            ordered_pairs = []
            for i, name_importance in enumerate(sorted_names_importances):
                if i == self._n_most_important_features:
                    break
                ordered_pairs.append(name_importance)
            result[model_name] = ordered_pairs
        return result


class OutputSignals(Output):
    'signals are the combined actual oasspreads and predictions'
    # these are for presentation purposes and to help diagnose accuracy
    # the signal needs to be in an easy-to-parse file
    def __init__(self, path):
        self._last_simulated_datetime = datetime.datetime(1, 1, 1)
        super(OutputSignals, self).__init__(
            path=path,
            field_names=[
                'simulated_datetime',
                'event_source', 'event_source_identifier', 'event_cusip',
                'action_identifier',
                'prediction_B', 'prediction_S',
                'standard_deviation_B', 'standard_deviation_S',
                'actual_B', 'actual_S',
            ],
        )

    def ensemble_prediction(self, ep, trade_type):
        # signal the prediction and its standard deviation
        assert isinstance(ep, EnsemblePrediction)
        self._check_datetime(ep.simulated_datetime)
        row = {
            'simulated_datetime': ep.simulated_datetime,
            'action_identifier': ep.action_identifier,
            'prediction_%s' % trade_type: ep.ensemble_prediction,
            'standard_deviation_%s' % trade_type: ep.standard_deviation,
        }
        self._writerow(row)
        # the most important features are written to a separate output file, not here
        # signal the most important features
        # most_important = most_importance_features(ep, self._n_most_important_features)
        # for model_name, feature_importance_list in most_important.iteritems():
        #     for feature, importance in feature_importance_list:
        #         row = {
        #             'simulated_datetime': ep.simulated_datetime,
        #             'importances_model_family': model_name,
        #             'importances_feature_name': feature,
        #             'importances_weight': importance,
        #         }
        #         self._write_row(row)

    def event_trace_print(self, event):
        self._check_datetime(event.datetime())
        row = {
            'simulated_datetime': str(event.datetime()),
            'event_source': event.source,
            'event_source_identifier': event.source_identifier,
            'event_cusip': event.cusip(),
            'actual_%s' % event.reclassified_trade_type(): event.oasspread(),
        }
        self._writerow(row)

    def _check_datetime(self, new_simulated_datetime):
        if new_simulated_datetime < self._last_simulated_datetime:
            # print new_simulated_datetime
            # print self._last_simulated_datetime
            seven.logging.info('simulated datetimes out of order')
        self._last_simulated_datetime = new_simulated_datetime


class OutputTrace(Output):
    'write complete log (a trace log)'
    def __init__(self, path):
        super(OutputTrace, self).__init__(
            path=path,
            field_names=[
                'simulated_datetime',
                'time_description',
                'what_happened',
                'info',
            ],
        )

    def ensemble_prediction(self, ensemble_prediction, count, reclassified_trade_type):
        assert isinstance(ensemble_prediction, EnsemblePrediction)
        d = {
            'simulated_datetime': ensemble_prediction.simulated_datetime,
            'time_description': 'simulated time + wallclock to create an ensemble prediction',
            'what_happened': 'the most recent accuracies were used to blend the predictions of recently-trained experts',
            'info': 'predicted #%d in %0.2f seconds next %s to be %f with standard deviation of %s using %d sets of experts; actual was %f' % (
                count,
                ensemble_prediction.elapsed_wallclock_seconds,
                reclassified_trade_type,
                ensemble_prediction.ensemble_prediction,
                ensemble_prediction.standard_deviation,
                len(ensemble_prediction.list_of_trained_experts),
                ensemble_prediction.actual,
            ),
        }
        self._writerow(d)

    def event_loop(self, n_events_handled, time_delta, total_wallclock_seconds):
        d = {
            'simulated_datetime': None,
            'time_description': None,
            'what_happened': 'the event loop started to handle event %d' % (n_events_handled + 1),
            'info': 'wallclock seconds since start: %0.2f, of which for test %0.2f for train %0.2f' % (
                time_delta.total_seconds(),
                total_wallclock_seconds['test'],
                total_wallclock_seconds['train'],
            ),
        }
        self._writerow(d)

    def new_feature_vector_created(self, feature_vector):
        d = {
            'simulated_datetime': feature_vector.creation_datetime,
            'time_description': 'simulated time + wallclock to create feature vector',
            'what_happened': 'new feature vector created from relevant event features',
            'info': str(feature_vector),
        }
        self._writerow(d)

    def no_test(self, dt, err, event):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to maybe test and train',
            'what_happened': 'was not able to test',
            'info': 'reason: %s, event=%s' % (
                err,
                event,
            ),
        }
        self._writerow(d)

    def no_train(self, dt, err, event):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to maybe test and train',
            'what_happened': 'was not able to train',
            'info': 'reason: %s, event=%s' % (
                err,
                event,
            ),
        }
        self._writerow(d)

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
        self._writerow(d)

    def trained_experts(self, trained_experts, trade_type):
        d = {
            'simulated_datetime': trained_experts.simulated_datetime,
            'time_description': 'simulated time + wallclock to train experts',
            'what_happened': 'new experts were trained',
            'info': 'trained %d %s experts on %d feature vectors in %f seconds' % (
                len(trained_experts.trained_models),
                trade_type,
                len(trained_experts.training_features),
                trained_experts.elapsed_wallclock_seconds,
            )
        }
        self._writerow(d)

    def update_features_cusip_primary(self, dt, event, event_attributes):
        return self._update_features(
            dt,
            event,
            event_attributes,
            'updated event features to include latest primary cusip attributes',
        )

    def update_features_cusip_otr(self, dt, event, event_attributes):
        return self._update_features(
            dt,
            event,
            event_attributes,
            'updated event features to include latest OTR cusip attributes',
        )

    def update_liq_flow_on_the_run(self, dt, event, event_attributes):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to determine new OTR cusip',
            'what_happened': 'new OTR cusip was identified',
            'info': 'OTR cusip changed to %s' % event_attributes['id_liq_flow_on_the_run_otr_cusip'],
        }
        self._writerow(d)

    def _update_features(self, dt, event, event_attributes, what_happened):
        def n_non_id_values():
            result = 0
            for k in event_attributes.value.keys():
                if not k.startswith('id_'):
                    result += 1
            return result

        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to update features in feature vector',
            'what_happened': what_happened,
            'info': 'updated %d features in the feature vector from event %s' % (
                n_non_id_values(),
                event
            )
        }
        self._writerow(d)


class SimulatedClock(object):
    def __init__(self):
        self.datetime = datetime.datetime(1, 1, 1, 0, 0, 0)

        self._last_event_wallclock = None

    def __repr__(self):
        return 'SimulatedTime(%s)' % self.datetime

    def handle_event(self):
        'increment the simulated datetime by the actual wall clock time since the last event was seen'
        elapsed_wallclock = datetime.datetime.now() - self._last_event_wallclock
        self.datetime = self.datetime + elapsed_wallclock

    def see_event(self, event):
        'remember when last event was seen; update simulated clock to be at least past the time in the event'
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


class TestTrain(object):
    def __init__(self,
                 action_identifiers,
                 control,
                 ensemble_hyperparameters,
                 select_target,
                 simulated_clock,
                 ):
        self._action_identifiers = action_identifiers
        self._control = control
        self._ensemble_hyperparameters = ensemble_hyperparameters
        self._select_target = select_target
        self._simulated_clock = simulated_clock

        self._all_feature_vectors = collections.deque(
            [],
            self._ensemble_hyperparameters.max_n_feature_vectors,
        )
        self._last_expert_training_time = datetime.datetime(1, 1, 1)
        self._list_of_trained_experts = collections.deque(
            [],
            self._ensemble_hyperparameters.max_n_trained_sets_of_experts,
        )

    def __repr__(self):
        return 'TestTrain(%d feature vectors, %d sets of trained experts)' % (
            len(self._all_feature_vectors),
            len(self._list_of_trained_experts),
        )

    def accumulate_feature_vector(self, feature_vector):
        assert isinstance(feature_vector, FeatureVector)
        self._all_feature_vectors.append(feature_vector)

    def maybe_test_and_train(self,
                             current_event,
                             verbose=False,
                             ):
        'return (Union[EnsemblePrediction, None], Union[TrainedExperts, None], Union[List[err], None])'
        # if possible, test (and return an EnsemblePrediction that describes the accuracy of the test)
        # if possible, train the experts. They are used to do tests on subsequent calls

        ensemble_prediction, errs_test = self._maybe_test(current_event)
        trained_experts, errs_train = self._maybe_train(current_event)
        if trained_experts is not None:
            self._list_of_trained_experts.append(trained_experts)  # so that _maybe_test can find them

        return ensemble_prediction, errs_test, trained_experts, errs_train

    def report(self):
        print 'all feature vectors'
        for feature_vector in self._all_feature_vectors:
            print ' %s' % feature_vector
        print 'all trained experts'
        for trained_experts in self._list_of_trained_experts:
            print ' %s' % trained_experts

    def _make_actual(self, feature_vector):
        'return (actual, err)'
        try:
            raw_value = self._select_target(feature_vector)
            actual = float(raw_value)
        except ValueError:
            err = 'actual value not interpretable as a float: %s' % raw_value
            return None, err
        return actual, None

    def _maybe_test(self, creation_event, verbose=False):
        'return (EnsemblePredicition, errs: List[str])'
        def test_one_set_of_experts(actual, trained_experts, query_features, verbose=True):
            'return an ExpertAccuracy'
            # determine unnormalized weights of experts (based on prediction accuracy)
            expert_absolute_errors = {}
            expert_predictions = {}
            expert_unnormalized_weights = {}
            sum_unnormalized_weights = 0.0
            for model_spec, trained_expert in trained_experts.trained_models.iteritems():
                predictions = trained_expert.predict(query_features)
                assert len(predictions) == 1
                prediction = predictions[0]

                absolute_error = abs(actual - prediction)
                expert_predictions[model_spec] = prediction
                expert_absolute_errors[model_spec] = absolute_error

                unnormalized_weight = math.exp(-self._ensemble_hyperparameters.weight_temperature * absolute_error)
                sum_unnormalized_weights += unnormalized_weight
                expert_unnormalized_weights[model_spec] = unnormalized_weight

            # determine normalized weights of experts
            expert_normalized_weights = {}
            for model_spec, unnormalized_weight in expert_unnormalized_weights.iteritems():
                expert_normalized_weights[model_spec] = unnormalized_weight / sum_unnormalized_weights

            # make the ensemble prediction from these experts
            ensemble_prediction = 0.0
            for model_spec, expert_prediction in expert_predictions.iteritems():
                ensemble_prediction += expert_prediction * expert_normalized_weights[model_spec]

            # determine weighted standard deviation of the experts' predictions
            variance = 0.0
            for model_spec, expert_prediction in expert_predictions.iteritems():
                delta = ensemble_prediction - expert_prediction
                variance += delta * delta * expert_normalized_weights[model_spec]
            standard_deviation = math.sqrt(variance)

            # determine weighted importances of features
            weighted_importances = collections.defaultdict(lambda: collections.defaultdict(float))
            for model_spec, trained_expert in trained_experts.trained_models.iteritems():
                weight = expert_normalized_weights[model_spec]
                for feature_name, feature_importance in trained_expert.importances.iteritems():
                    weighted_importance = weight * feature_importance
                    weighted_importances[model_spec][feature_name] += weighted_importance

            # create the explanation lines for the ensemble predicton and standard deviation
            lines = []
            lines.append('ensemble prediction and standard deviation')
            for model_spec in expert_normalized_weights.keys():
                lines.append(
                    'expert %30s prediction %10.6f actual %10.6f absolute error %10.6f normalized weight %10.6f' % (
                        model_spec,
                        expert_predictions[model_spec],
                        actual,
                        expert_absolute_errors[model_spec],
                        expert_normalized_weights[model_spec],
                    ))
            lines.append('ensemble_prediction=%f, standard_deviation=%f' % (
                ensemble_prediction,
                standard_deviation,
            ))
            # create the explanation lines for the weighted feature importances
            lines.append(' ')
            lines.append('start of weighted feature importance when not zero')
            for model_spec, feature_importance in weighted_importances.iteritems():
                for feature_name, weighted_importance in feature_importance.iteritems():
                    if weighted_importance > 0.0:
                        weight = expert_normalized_weights[model_spec]
                        lines.append('model_spec %30s weight %10.6f feature %60s weighted importance %10.6f' % (
                            model_spec,
                            weight,
                            feature_name,
                            weighted_importance,
                        ))
            lines.append('end of weighted feature importance when not zero')

            if verbose:
                print 'explanation of trained experts %s' % trained_experts
                for line in lines:
                    print line

            expert_accuracy = ExpertAccuracy(
                actual=actual,
                absolute_errors=expert_absolute_errors,
                ensemble_prediction=ensemble_prediction,
                explanation=lines,
                importances=weighted_importances,
                normalized_weights=expert_normalized_weights,
                predictions=expert_predictions,
                standard_deviation=standard_deviation,
            )

            return expert_accuracy

        start_wallclock = datetime.datetime.now()

        errs = []
        if len(self._all_feature_vectors) < 2:
            err = 'need at least 2 feature vectors to test, had %d' % len(self._all_feature_vectors)
            errs.append(err)

        if len(self._list_of_trained_experts) == 0:
            err = 'had 0 trained collections of experts, need at least 1'
            errs.append(err)

        if len(errs) > 0:
            return None, errs

        actual_feature_vector = self._all_feature_vectors[-1]
        query_feature_vector = self._all_feature_vectors[-2]
        query_features = [query_feature_vector.payload]  # must be a List[Dict]

        actual, err = self._make_actual(actual_feature_vector)
        if err is not None:
            return None, [err]

        print 'testing %d sets of experts ...' % len(self._list_of_trained_experts)

        # test each of the sets of experts
        # detemine their unnormalized weights
        sum_unnormalized_weight = 0.0
        experts_accuracies_list = []
        trained_experts_accuracies = {}
        unnormalized_weights = {}
        testing_datetime = actual_feature_vector.creation_event.datetime()
        for trained_experts in self._list_of_trained_experts:
            # unnormalized weight
            training_datetime = trained_experts.creation_event.datetime()
            age_timedelta = testing_datetime - training_datetime
            assert self._ensemble_hyperparameters.weight_units == 'days'
            age_days = age_timedelta.total_seconds() / (60.0 * 60.0 * 24.0)
            unnormalized_weight = math.exp(
                -self._ensemble_hyperparameters.weight_temperature * age_days
            )
            unnormalized_weights[trained_experts] = unnormalized_weight
            sum_unnormalized_weight += unnormalized_weight

            # accuracy of these experts
            expert_accuracy = test_one_set_of_experts(
                actual=actual,
                trained_experts=trained_experts,
                query_features=query_features,
            )
            trained_experts_accuracies[trained_experts] = expert_accuracy
            experts_accuracies_list.append(expert_accuracy)

        # just above, we determined the accuracies of each of the sets of experts
        # As part of that, we computed their ensemble prediction and standard deviation
        # Now we blend these sets of experts together to form the final ensemble prediction and standard deviation
        ensemble_prediction = 0.0
        ensemble_standard_deviation = 0.0
        normalized_weights = {}
        for trained_experts in self._list_of_trained_experts:
            normalized_weight = unnormalized_weights[trained_experts] / sum_unnormalized_weight
            print trained_experts, normalized_weight
            normalized_weights[trained_experts] = normalized_weight

            trained_experts_accuracy = trained_experts_accuracies[trained_experts]
            ensemble_prediction += normalized_weight * trained_experts_accuracy.ensemble_prediction
            ensemble_standard_deviation += normalized_weight * trained_experts_accuracy.standard_deviation

        # aggregate the importances (using the normalized_weights)
        aggregate_importances = collections.defaultdict(lambda: collections.defaultdict(float))
        for trained_expert, normalized_weight in normalized_weights.iteritems():
            expert_accuracy = trained_experts_accuracies[trained_expert]
            for model_spec, feature_importance in expert_accuracy.importances.iteritems():
                for feature_name, feature_importance in feature_importance.iteritems():
                    aggregate_importances[model_spec.name][feature_name] += normalized_weight * feature_importance

        # build the overall explanation (List[str])
        lines = []
        lines.append(
            'tested with %d most-recnelty trained sets of experts' % len(self._list_of_trained_experts),
        )
        for trained_experts in self._list_of_trained_experts:
            lines.append(' experts %50s had overall weight %10.6f' % (
                trained_experts,
                normalized_weights[trained_expert],
            )
            )
        for trained_expert in self._list_of_trained_experts:
            lines.append('*** details on accuracy of %s' % trained_expert)
            for line in trained_experts_accuracies[trained_expert].explanation:
                lines.append(line)
        for expert_accuracy in experts_accuracies_list:
            lines.append('**** detail')
            for line in expert_accuracy.explanation:
                lines.append(line)

        # build return value
        self._simulated_clock.handle_event()
        simulated_datetime = self._simulated_clock.datetime
        action_identifier = self._action_identifiers.next_identifier(simulated_datetime)
        result = EnsemblePrediction(
            action_identifier=action_identifier,
            actual=actual,
            creation_event=creation_event,
            elapsed_wallclock_seconds=(datetime.datetime.now() - start_wallclock).total_seconds(),
            ensemble_prediction=ensemble_prediction,
            expert_accuracies=experts_accuracies_list,
            explanation=lines,
            feature_vector_for_actual=actual_feature_vector,
            feature_vector_for_query=query_feature_vector,
            importances=aggregate_importances,
            list_of_trained_experts=self._list_of_trained_experts,
            simulated_datetime=self._simulated_clock.datetime,
            standard_deviation=ensemble_standard_deviation,
        )
        return result, None

    def _maybe_train(self, creation_event, verbose=False):
        'return (Union[TrainedExperts, None], errs: Union[None,List[str]])'
        start_wallclock = datetime.datetime.now()

        errs = []
        time_since_last_training = self._simulated_clock.datetime - self._last_expert_training_time
        min_time = self._ensemble_hyperparameters.min_timedelta_between_training
        if time_since_last_training < min_time:
            err = '%s had not passed since last training' % min_time
            errs.append(err)

        if len(self._all_feature_vectors) < 2:
            err = 'need at least 2 feature vectors to train, had %d' % len(self._all_feature_vectors)
            errs.append(err)

        if len(errs) > 0:
            return None, errs

        def dt(index):
            'return datetime of the trace event for the primary cusip'
            return self._all_feature_vectors[index].payload['id_p_trace_event'].datetime()

        if verbose:
            print 'in _maybe_train: selecting training data'
            print 'all feature vectors'
            pp(self._all_feature_vectors)
        training_features = []
        training_targets = []
        err = None
        if verbose:
            print 'in _maybe_train: creating training data'
        for i in xrange(0, len(self._all_feature_vectors) - 1):
            training_features.append(self._all_feature_vectors[i].payload)
            features_datetime = dt(i)
            next_index = i + 1
            while (next_index < len(self._all_feature_vectors)) and dt(next_index) <= features_datetime:
                next_index += 1
            if next_index == len(self._all_feature_vectors):
                err = 'no future trade in same primary cusip to find target value'
                return None, [err]
                break
            target_feature_vector = self._all_feature_vectors[next_index]
            training_targets.append(self._select_target(target_feature_vector))
        if len(training_targets) == 0:
            err = 'no training targets were found for the feature vectors'
            return None, [err]

        assert len(training_features) == len(training_targets)

        print 'training experts on %d training samples ...' % len(training_features)
        grid = seven.HpGrids.construct_HpGridN(self._control.arg.hpset)
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
            model = model_constructor(model_spec, self._control.random_seed)
            try:
                if verbose:
                    print 'training expert', model.model_spec
                model.fit(training_features, training_targets)
            except seven.models2.ExceptionFit as e:
                seven.logging.warning('could not fit %s: %s' % (model_spec, e))
                continue
            trained_models[model_spec] = model  # the fitted model
        self._simulated_clock.handle_event()
        result = TrainedExperts(
            creation_event=creation_event,
            elapsed_wallclock_seconds=(datetime.datetime.now() - start_wallclock).total_seconds(),
            simulated_datetime=self._simulated_clock.datetime,
            trained_models=trained_models,
            training_features=training_features,
            training_targets=training_targets,
        )
        print 'trained %s experts on %d training samples in %f wallclock seconds; creation event=%s' % (
            len(trained_models),
            len(training_features),
            (datetime.datetime.now() - start_wallclock).total_seconds(),
            creation_event,
        )
        return result, None


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


def maybe_make_feature_vectorOLD(control, current_otr_cusip, event, event_feature_values):
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

    current_otr_cusip = ''

    output_actions = OutputActions(
        control.path['out_actions'],
    )
    output_experts = OutputExperts(
        control.path['out_experts'],
    )
    output_importances = OutputImportances(
        control.path['out_importances'],
        n_most_important_features=16,  # number of features in each feature vector
    )
    output_signals = OutputSignals(
        path=control.path['out_signal'],
    )
    output_trace = OutputTrace(
        path=control.path['out_trace'],
    )

    seven.logging.verbose_info = False
    seven.logging.verbose_warning = False
    simulated_clock = SimulatedClock()
    action_identifiers = ActionIdentifiers()
    events_at = collections.Counter()
    event_attribute_makers_trace = {}
    event_attribute_makers_not_trace = {}
    feature_vector_maker = FeatureVectorMaker(control.arg)
    total_wallclock_seconds = collections.Counter()

    def select_target(feature_vector, reclassified_trade_type):
        key = 'p_trace_%s_%s' % (reclassified_trade_type, control.arg.target)
        return feature_vector.payload[key]

    def select_target_B(feature_vector):
        return select_target(feature_vector, 'B')

    def select_target_S(feature_vector):
        return select_target(feature_vector, 'S')

    # construct a testing and training engine for each reclassified trade types
    # Each engine sees feature vectors for only B or S trades
    test_train = {
        'B': TestTrain(action_identifiers, control, ensemble_hyperparameters, select_target_B, simulated_clock),
        'S': TestTrain(action_identifiers, control, ensemble_hyperparameters, select_target_S, simulated_clock),
    }
    event_loop_wallclock_start = datetime.datetime.now()
    feature_vector = None
    print 'pretending that events before %s never happened' % control.arg.start_events
    while True:
        try:
            event = event_queue.next()
            counter['events read'] += 1
        except StopIteration:
            break  # all the event readers are empty

        events_at[event.datetime()] += 1

        if event.date() > control.arg.stop_predictions:
            print 'stopping event loop, because beyond stop predictions date of %s' % control.arg.stop_predictions
            print 'event=', event
            break

        if event.date() < control.arg.start_events:  # skip events before our simulated world starts
            counter['events ignored'] += 1
            continue

        # track events that occured at the exact same moment
        # In the future, such events need to be treated as a cluster and process together

        simulated_clock.see_event(event)  # remember current wallclock time
        seven.logging.verbose_info = True
        seven.logging.verbose_warning = True
        counter['events processed'] += 1
        print '\nprocessing event # %d: %s' % (counter['events processed'], event)
        # if counter['events processed'] > 12779:
        #     print 'near to it'
        #     pdb.set_trace()

        if True:  # build and accumulate feature vectors before we start predicting
            # if we waited to build feature vectors until we started predicting, the initial
            # predictions would have little training data
            source = event.source

            if True:  # lazily build the event_attribute_makers
                if source == 'trace':
                    cusip = event.cusip()
                    if cusip not in event_attribute_makers_trace:
                        event_attribute_makers_trace[cusip] = event.make_event_attributes_class(control.arg)
                    event_attributes, errs = event_attribute_makers_trace[cusip].make_attributes(event)
                else:
                    if source not in event_attribute_makers_not_trace:
                        print event.make_event_attributes_class
                        event_attribute_makers_not_trace[source] = event.make_event_attributes_class(control.arg)
                    event_attributes, errs = event_attribute_makers_not_trace[source].make_attributes(event)

                if errs is not None:
                    for err in errs:
                        irregularity.no_attribute_maker(err, event)
                    # continue for now
                    # later, once the event attribute makers are all running, stop with an error
                    continue

            if True:  # build attributes from the event
                if event.source == 'trace':
                    cusip = event.cusip()
                    if cusip == control.arg.cusip or cusip == current_otr_cusip:
                        # a cusip can be both a primary cusip and an OTR cusip
                        if cusip == control.arg.cusip:
                            feature_vector_maker.update_cusip_primary(event, event_attributes)
                            simulated_clock.handle_event()
                            output_trace.update_features_cusip_primary(simulated_clock.datetime, event, event_attributes)
                        if cusip == current_otr_cusip:
                            feature_vector_maker.update_cusip_otr(event, event_attributes)
                            simulated_clock.handle_event()
                            output_trace.update_features_cusip_otr(simulated_clock.datetime, event, event_attributes)
                    else:
                        irregularity.skipped_event(
                            'trace event for cusips %s, which is not primary (%s) nor otr (%s)' % (
                                cusip,
                                control.arg.cusip,
                                current_otr_cusip,
                            ),
                            event,
                        )
                        continue
                elif event.source == 'liq_flow_on_the_run':
                    current_otr_cusip = event_attributes['id_liq_flow_on_the_run_otr_cusip']
                    simulated_clock.handle_event()
                    output_trace.update_liq_flow_on_the_run(simulated_clock.datetime, event, event_attributes)
                else:
                    # for now, skip all other event sources
                    # later, add these to the feature vector
                    pdb.set_trace()
                    irregularity.skipped_event(
                        'event source %s not yet processed' % event.source,
                        event,
                    )
                    continue

            if True:  # trace cusip events for the primary and OTR cusips
                def has_cusip(cusip):
                    return event.is_trace_print_with_cusip(cusip)

                if has_cusip(control.arg.cusip) or has_cusip(current_otr_cusip):
                    output_signals.event_trace_print(event)

            if True:  # accumulate feature vectors
                if feature_vector_maker.have_all_event_attributes():
                    simulated_clock.handle_event()
                    feature_vector = FeatureVector(
                        creation_datetime=simulated_clock.datetime,
                        creation_event=event,
                        payload=feature_vector_maker.all_features(),
                    )
                    # print 'new feature vector', feature_vector
                    output_trace.new_feature_vector_created(feature_vector)
                    rtt = event.reclassified_trade_type()
                    test_train[rtt].accumulate_feature_vector(feature_vector)
                    # print 'new feature vector accumulated', rtt, test_train[rtt]

        if counter['events processed'] % 100 == 0:
            output_trace.event_loop(
                counter['events processed'] + 1,
                datetime.datetime.now() - event_loop_wallclock_start,
                total_wallclock_seconds,
            )

        if event.date() < control.arg.start_predictions:
            continue  # just accumulate feature vectors before the start date

        print str(test_train)
        print 'processed %d prior events in %0.2f wallclock minutes' % (
            counter['events processed'] - 1,
            control.timer.elapsed_wallclock_seconds() / 60.0,
        )

        counter['events on or after start-predictions date'] += 1
        if True:  # attempt to test and train
            if not event.is_trace_print_with_cusip(control.arg.cusip):
                irregularity.no_testing('event not trace print for primary cusip', event)
                continue
            if feature_vector is None:
                irregularity.no_feature_vector('missing attributes:' + feature_vector_maker.missing_attributes(), event)
                continue
            # event was for the primary cusip
            rtt = feature_vector.reclassified_trade_type()
            print 'about to call maybe_test_and_train', rtt
            ensemble_prediction, errs_test, trained_experts, errs_train = (
                test_train[rtt].maybe_test_and_train(
                    current_event=event,
                )
            )
            print 'returned from test_train.maybe_test_and_train', rtt
            print 'errs_test', errs_test
            print 'errs_train', errs_train
            simulated_clock.handle_event()
            if True:  # process any errors
                if errs_test is not None:
                    print 'reclassified trade type', rtt
                    for err in errs_test:
                        irregularity.no_test(err, event)
                        output_trace.no_test(simulated_clock.datetime, err, event)
                if errs_train is not None:
                    for err in errs_train:
                        irregularity.no_train(err, event)
                        output_trace.no_train(simulated_clock.datetime, err, event)
                # pdb.set_trace()
            if errs_test is None:  # process the test results (the ensemble prediction)
                print 'processing an ensemble prediction', rtt

                counter['ensemble predictions made'] += 1
                counter['ensemble predictions made %s' % rtt] += 1

                total_wallclock_seconds['test'] += ensemble_prediction.elapsed_wallclock_seconds
                output_actions.ensemble_prediction(ensemble_prediction, rtt)
                output_experts.expert_accuracy(ensemble_prediction)
                output_importances.ensemble_prediction(ensemble_prediction, rtt)
                output_signals.ensemble_prediction(ensemble_prediction, rtt)
                output_trace.ensemble_prediction(
                    ensemble_prediction,
                    counter['ensemble predictions made'],
                    rtt,
                    )

                # write the explanation (which is lines of plain text)
                explanation_filename = ('%s.txt' % ensemble_prediction.action_identifier)
                explanation_path = os.path.join(
                    control.path['dir_out'],
                    'ensemble-predictions',
                    explanation_filename,
                )
                with open(explanation_path, 'w') as f:
                    for line in ensemble_prediction.explanation:
                        f.write(line)
                        f.write('\n')

            if errs_train is None:  # process the training results (the trained_experts)
                print 'processing trained experts', rtt

                counter['experts trained'] += 1
                counter['experts trained %s' % rtt] += 1

                total_wallclock_seconds['train'] += trained_experts.elapsed_wallclock_seconds
                output_trace.trained_experts(
                    trained_experts,
                    rtt,
                )

        if counter['ensemble predictions made'] > 0:
            print 'for now, stopping early'
        gc.collect()
        continue

    output_actions.close()
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
    print '*********************'
    print 'events at same time with non-zero hour:minute:second'
    for k in sorted(events_at.keys()):
        v = events_at[k]
        if k.hour == 0 and k.minute == 0 and k.second == 0:
            pass
        else:
            if v > 1:
                print '%30s: %d' % (k, v)
    print
    print '************************************'
    for reclassified_trade_type in ('B', 'S'):
        print 'test train', reclassified_trade_type, 'ending state:'
        print '%s' % test_train[reclassified_trade_type]
        test_train[reclassified_trade_type].report()
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
