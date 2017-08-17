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
  python test_train.py AAPL 037833AJ9 oasspread grid5 2017-07-01 2017-07-01 --debug # run until end of events
  python test_train.py AAPL 037833AJ9 oasspread grid5 2017-04-01 2017-06-01 --debug --test # a few ensemble predictions

See build.py for input and output files.

APPROACH:
 Repeat until no more input:
    1. Read the next event.
        - For now, the events are in files, one file per source for the event.
        - Each record in each event file has a date or datetime stamp.
        - Read the records in datetime order.
        - For now, assume that a record with a date stamp and not a datetime stamp was generated at 00:00:00,
          so that it comes first on the date.
        - For now, arbitrarily select from records with the same datetime. This is a problem, as it
          assumes an ordering. In the future, treat all these events as having occured at the same time.
        - Keep track of the current OTR cusips. That is found by reading events from the
          liq_flow_on_the_run_{ticker}.csv event file.
        - After reading each event, create attributes for the event. The attributes may depend on
          the data in the event record and on data in prior event records. For example, an attribute
          could be the ratio of the current oasspread to the prior oasspread.
       - FIXME: The ratios and differences should be only for the same trade type.
    2. Create a feature vector, if we have enough new events to do so
        - For now, the events of interest are only the trace prints for the cusips, because no other
          events contribute features
        - For now, we create a new feature vector if and only if the most recent event is a trace print for
          the current OTR cusip or for the primary cusip. Later, we will create feature vectors if some other event
          has contributed new data that are used to construct the features.
        - For now, the feature vector has just attributes from events from the trace prints. Those event attributes
          become features in the feature vector.
        - Later, the feature vector will have a complex construction, namely:
          - First, all of the event attributes will be copied into the feature vector.
          - Then the feature vector constuctor will develop additional features that are functions
            of the features provided by the events: new_feature = f(event_attributes_1, event_attributes_2)
        -  A feature vector may contain information from events that were B or S trades. For example, a
           feature vector may be built from a primary cusip for a B trade and an OTR cusip for an S trade. Or
           the attributes for the primary cusip may have data from prior trades that were not B trades.
        - FIXME: one set of features for the Bs, one for the Ss; hence, for now, 12 features all together. Up the last
          feature vector, to update the B or S features, based on what the current trace print is. Call the feature
          vector a B or S based on the trace print trade type that caused it be updated. Use only trades on the primary
          CUSIP to set the trade type of the feature vector.
    3. If we cannot create a new feature vector, skip back to the start of the loop.
        - If the event is not for the primary cusip and is not for the current OTR cusip, we have no
          new information to create features, so we do not create the feature vector.
    4. Test the current experts
        - Test if and only if
          - We have a new feature vector; and
          - We have at least one set of trained experts (see just below for training); and
          - We have at least 2 feature vectors.
        - The testing is by making predictions and measuring their accuracies. We predict both the next B
          and S oasspread. Each set of experts is either for B or S trades.
        - The query vector is the feature vector created just before the most recent feature vector
          was created. We use it to predict the oasspread in the just-created feature vector.
        - The prediction is a weighted average of the predictions of the sets of experts. As the program
          runs, it will build up many sets of experts. The predictions of the experts are weighted
          by exp(-days * error), where days is how long ago the experts were trained and error is the error
          in the expert's prediction of the oasspread in the most recent feature vector.
    5. Train new experts.
       - Each possible hyperparameter setting leads to a distinct expert. For now, there are about 1,000 experts.
       - We group the experts into sets of experts. Each expert in the set was trained on a common set of training
         features and targets and at the same time.
       - The training data set consists of feature vectors created by primary events for B and S trades.
       - The training features are the B or S feature vectors in the current list of feature vectors,
         excluding the last one. Thus, the set of experts is good for predicting either B or S trades, but
         not both.
       - The targets are the oasspread in the next-in-time B or S feature vector.
       - The construction of the training data admits these possibilities:
         - Possibility 1:
           Suppose there are 3 events, all for B trades:
             event1 = primary1, for the primary cusip
             event2 = otr2, for an otr cusip
             event3 = otr3, for an otr cusip
           Suppose that 2 feature vectors have been constructed from the event attributes:
             fv1 = FeatureVector(primary1, otr2)
             fv2 = FeatureVector(primary1, otr3)
           Then, we have no training data, because fv1 has no future oasspread to become the target value.abs
         - Possibility 2:
           Suppose there are the events above plus this additional B trade:
            event4 = primary4, for the primary cusip
           We then have 3 feature vectors:
             fv1 = FeatureVector(primary1, otr2)
             fv2 = FeatureVector(primary1, otr3)
             fv3 = FeatureVector(prrimary2, otr3)
           Then
             fv1 is in the training set, with the target from fv3
             fv2 is in the training set, with the taget from fv3 (the same target!)

APPROACH (Alternative, not implemented):
 1. Construct two models
    modelB = model for oasspread B predictions := oasspread_predictions('B')
    modelS = model for oasspread S predictions := oasspread_predictions('S')
 2. Repeat until no more events
    a. Read next event
    b. Develop attributes for the event.
    c. Keep track of the current OTR cusip. It changes when certain events are processed.abs
    d. If have enough events to build a feature vector:
       (1) Update the feature vector fv. If the event causing the update was from a primary B or S event, then
           send the just-update feature vector to the B and S model (respectively).
       (2) For B or S model, executive signal := model.process_feature(fv) and save the
           predictions if any. The method process_features does the testing and training as described
           above. The signal consists of predictions and diagnostic information.

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
import seven.fit_predict_output
import seven.HpGrids
import seven.input_event
import seven.logging
import seven.make_event_attributes
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
    applied_data_science.dirutility.assure_exists(os.path.join(paths['dir_out'], 'ensemble-predictions'))

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
        'actual',
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

TrainedExperts = collections.namedtuple(
    'StateTrainedExpert', [
        'creation_event',
        'elapsed_wallclock_seconds',  # time for training all of the experts
        'simulated_datetime',
        'trained_models',             # Dict[model_spec, trained_model]
        'training_features',
        'training_targets',
    ],
    )


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
        self.max_n_trained_sets_of_experts = 100
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
                 payload,
                 reclassified_trade_type,
                 ):
        # these fields are part of the API
        self.creation_event = copy.copy(creation_event)
        self.reclassified_trade_type = reclassified_trade_type
        self.payload = payload
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
        return 'FeatureVector(creation_event=%s, rtt=%s, n features=%d)' % (
            self.creation_event,
            self.reclassified_trade_type,
            len(self.payload),
        )

    def __str__(self):
        'return fields in the Feature Vector'
        values = ''
        for key in sorted(self.payload.keys()):
            if not key.startswith('id_'):
                if len(values) > 0:
                    values += ', '
                values += '%s=%0.2f' % (key, self.payload[key])
        return 'FeatureVector(%s, %s)' % (self.reclassified_trade_type, values)


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

    def update_cusip_otr(self, event, event_attributes):
        self._event_attributes_cusip_otr = event_attributes

    def update_cusip_primary(self, event, event_attributes):
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

    def trace_event(self, event):
        self._check_datetime(event.datetime())
        row = {
            'simulated_datetime': str(event.datetime()),
            'event_source': event.source,
            'event_source_identifier': event.source_identifier,
            'event_cusip': event.cusip(),
            'actual_%s' % event.reclassified_trade_type(): event.oasspread(),
        }
        self._write_row(row)

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

    def ensemble_prediction(self, ensemble_prediction, reclassified_trade_type):
        assert isinstance(ensemble_prediction, EnsemblePrediction)
        d = {
            'simulated_datetime': ensemble_prediction.simulated_datetime,
            'time_description': 'simulated time + wallclock to create an ensemble prediction',
            'what_happened': 'the most recent accuracies were used to blend the predictions of recently-trained experts',
            'info': 'predicted next %s to be %f with standard deviation of %s using %d sets of experts; actual was %f' % (
                reclassified_trade_type,
                ensemble_prediction.ensemble_prediction,
                ensemble_prediction.standard_deviation,
                len(ensemble_prediction.list_of_trained_experts),
                ensemble_prediction.actual,
            ),
        }
        self._writerow(d)

    def event_features_created_for_source_not_trace(self, dt, event):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to determine create event features',
            'what_happened': 'data in an input event was converted to a uniform internal form',
            'info': str(event),
        }
        self._writerow(d)

    def event_features_created_for_source_trace(self, dt, event):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to determine create event features',
            'what_happened': 'data in an input event was converted to a uniform internal form',
            'info': str(event),
        }
        self._writerow(d)

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
        self._writerow(d)

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
        self._writerow(d)

    def new_feature_vector_created(self, dt, feature_vector):
        d = {
            'simulated_datetime': dt,
            'time_description': 'simulated time + wallclock to create feature vector',
            'what_happened': 'new feature vector created from relevant event features',
            'info': str(feature_vector),
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

    def maybe_test_and_train(self,
                             current_event,
                             feature_vector,
                             verbose=True,
                             ):
        'return (Union[EnsemblePrediction, None], Union[TrainedExperts, None], Union[List[err], None])'
        # if possible, test (and return an EnsemblePrediction that describes the accuracy of the test)
        # if possible, train the experts. They are used to do tests on subsequent calls
        self._all_feature_vectors.append(feature_vector)

        ensemble_prediction, errs_test = self._maybe_test(current_event)
        trained_experts, errs_train = self._maybe_train(current_event)
        if trained_experts is not None:
            self._list_of_trained_experts.append(trained_experts)  # so that _maybe_test can find them

        return ensemble_prediction, errs_test, trained_experts, errs_train

    def _make_actual(self, feature_vector):
        'return (actual, err)'
        try:
            raw_value = self._select_target(feature_vector)
            actual = float(raw_value)
        except ValueError:
            err = 'actual value not interpretable as a float: %s' % raw_value
            return None, err
        return actual, None

    def _maybe_test(self, creation_event, verbose=True):
        'return (EnsemblePredicition, errs: List[str])'
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

        print 'testing %d sets of experts ...' % len(self._list_of_trained_experts)
        actual_feature_vector = self._all_feature_vectors[-1]
        query_feature_vector = self._all_feature_vectors[-2]
        query_features = query_feature_vector.payload

        actual, err = self._make_actual(actual_feature_vector)
        if err is not None:
            return None, [err]

        # predict with the experts, tracking errors and building report lines
        lines = []

        def line(s):
            if verbose:
                print s
            lines.append(s)

        testing_datetime = actual_feature_vector.creation_event.datetime()

        # determine unnormalized weights for each expert's prediction using the query features
        line('using %d sets of trained experts to predict query at %s with actual target %s' % (
            len(self._list_of_trained_experts),
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
        for trained_experts_index, trained_experts in enumerate(self._list_of_trained_experts):
            training_datetime = trained_experts.creation_event.datetime()
            age_timedelta = testing_datetime - training_datetime
            age_days = age_timedelta.total_seconds() / (60.0 * 60.0 * 24.0)
            print training_datetime, testing_datetime, age_days
            ages_days[trained_experts_index] = age_days
            assert self._ensemble_hyperparameters.weight_units == 'days'
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

                factor = self._ensemble_hyperparameters.weight_temperature * age_days * absolute_error
                unnormalized_weight = math.exp(-factor)
                unnormalized_weights[trained_experts_index][model_spec] = unnormalized_weight

                sum_all_unnormalized_weights += unnormalized_weight

        # determine normalized the weights and the ensemble prediction and the weighted importances
        line('contribution of sets of experts to the ensemble prediction; actual = %6.2f' % actual)
        weighted_importances = collections.defaultdict(lambda: collections.defaultdict(float))
        ensemble_prediction = 0.0
        normalized_weights = collections.defaultdict(dict)
        for trained_experts_index, trained_experts in enumerate(self._list_of_trained_experts):
            line(
                'expert set %3d' % (trained_experts_index + 1) +
                ' trained on %d samples' % (len(trained_experts.training_features)) +
                ' at %s' % trained_experts.creation_event.datetime() +
                ''
            )
            # pdb.set_trace()  # normalized weight seems wrong
            for model_spec in sorted(trained_experts.trained_models.keys()):
                trained_expert = trained_experts.trained_models[model_spec]
                normalized_weight = unnormalized_weights[trained_experts_index][model_spec] / sum_all_unnormalized_weights
                normalized_weights[trained_experts_index][model_spec] = normalized_weight

                expert_prediction = expert_predictions[trained_experts_index][model_spec]
                contribution = normalized_weight * expert_prediction
                ensemble_prediction += contribution

                line(
                    ' %-30s' % model_spec +
                    ' predicted %10.6f' % expert_predictions[trained_experts_index][model_spec] +
                    ' abs error %10.6f' % absolute_errors[trained_experts_index][model_spec] +
                    ' weights unnormalized %10.6f' % unnormalized_weights[trained_experts_index][model_spec] +
                    ' normalized %10.6f' % normalized_weights[trained_experts_index][model_spec] +
                    ' normalized %10.6f' % normalized_weight +
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
        for trained_experts_index, trained_experts in enumerate(self._list_of_trained_experts):
            line(
                ' expert set %03d' % (trained_experts_index + 1) +
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
                    ' expert prediction %10.6f' % expert_prediction +
                    ' ensemble prediction %10.6f' % ensemble_prediction +
                    ' weighted delta %10.6f' % weighted_delta +
                    ''
                )
        standard_deviation = math.sqrt(weighted_variance)

        self._simulated_clock.handle_event()
        simulated_datetime = self._simulated_clock.datetime
        action_identifier = self._action_identifiers.next_identifier(simulated_datetime)
        result = EnsemblePrediction(
            actual=actual,
            action_identifier=action_identifier,
            creation_event=creation_event,
            elapsed_wallclock_seconds=(datetime.datetime.now() - start_wallclock).total_seconds(),
            ensemble_prediction=ensemble_prediction,
            explanation=lines,
            importances=weighted_importances,
            feature_vector_for_actual=actual_feature_vector,
            feature_vector_for_query=query_feature_vector,
            list_of_trained_experts=self._list_of_trained_experts,
            standard_deviation=standard_deviation,
            simulated_datetime=self._simulated_clock.datetime,
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
        print 'trained %s experts on %d training samples; creation event=%s' % (
            len(trained_models),
            len(training_features),
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


def maybe_test_ensemble(control,
                        ensemble_hyperparameters,
                        creation_event,
                        feature_vectors,
                        list_of_trained_experts,
                        verbose=True):
    'return (StateEnsemblePrediction, errs)'
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
        # pdb.set_trace()  # normalized weight seems wrong
        for model_spec in sorted(trained_experts.trained_models.keys()):
            trained_expert = trained_experts.trained_models[model_spec]
            normalized_weight = unnormalized_weights[trained_experts_index][model_spec] / sum_all_unnormalized_weights
            normalized_weights[trained_experts_index][model_spec] = normalized_weight

            expert_prediction = expert_predictions[trained_experts_index][model_spec]
            contribution = normalized_weight * expert_prediction
            ensemble_prediction += contribution

            line(
                ' %-30s' % model_spec +
                ' predicted %10.6f' % expert_predictions[trained_experts_index][model_spec] +
                ' abs error %10.6f' % absolute_errors[trained_experts_index][model_spec] +
                ' weights unnormalized %10.6f' % unnormalized_weights[trained_experts_index][model_spec] +
                ' normalized %10.6f' % normalized_weights[trained_experts_index][model_spec] +
                ' normalized %10.6f' % normalized_weight +
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
            ' expert set %03d' % (trained_experts_index + 1) +
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
                ' expert prediction %10.6f' % expert_prediction +
                ' ensemble prediction %10.6f' % ensemble_prediction +
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


def most_importance_features(ensemble_prediction, k):
    'return Dict[model_name, List(feature_name, importance_value)] for k most important features'
    pdb.set_trace()
    assert isinstance(ensemble_prediction, EnsemblePrediction)
    assert k >= 0
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
            if i == k:
                break
            ordered_pairs.append(name_importance)
        result[model_name] = ordered_pairs
    return result


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

    def to_datetime_date(s):
        year, month, day = s.split('-')
        return datetime.date(int(year), int(month), int(day))

    start_events = to_datetime_date(control.arg.start_events)
    start_predictions = to_datetime_date(control.arg.start_predictions)
    ignored = datetime.datetime(2017, 7, 1, 0, 0, 0)  # NOTE: must be at the start of a calendar quarter
    current_otr_cusip = ''

    output_actions = OutputActions(
        control.path['out_actions'],
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
    print 'pretending that events before %s never happened' % ignored
    while True:
        try:
            event = event_queue.next()
            counter['events read'] += 1
        except StopIteration:
            break  # all the event readers are empty

        events_at[event.datetime()] += 1

        if event.date() < start_events:  # skip events before our simulated world starts
            counter['events ignored'] += 1
            continue

        # track events that occured at the exact same moment
        # In the future, such events need to be treated as a cluster and process together

        simulated_clock.see_event(event)  # remember current wallclock time
        seven.logging.verbose_info = True
        seven.logging.verbose_warning = True
        counter['events processed'] += 1
        print '\nprocessing event # %d: %s' % (counter['events processed'], event)

        # print summary of global state
        if event.date() >= start_predictions:
            print str(test_train)
            print 'processed %d prior events in %0.2f wallclock minutes' % (
                counter['events processed'] - 1,
                control.timer.elapsed_wallclock_seconds() / 60.0,
            )

        # attempt to extract event-features from the event
        # keep track of the current OTR cusip

        if True:  # update the feature vector using the attributes from events
            # update the feature_vector, if the event has usable attributes
            # construct event attribute makers lazily
            # need to do this because we do not know all the cusips that may arrive and
            # we need an event attribute maker for each cusip
            source = event.source
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
                    irregularity.no_attributes(err, event)
                continue

            if event.source == 'trace':
                cusip = event.cusip()
                if cusip == control.arg.cusip:
                    feature_vector_maker.update_cusip_primary(event, event_attributes)
                elif cusip == current_otr_cusip:
                    feature_vector_maker.update_cusip_otr(event, event_attributes)
                else:
                    irregularity.skipped_event(
                        'trace event for cusips %s, which is not primary or otr' % cusip,
                        event,
                    )
                    continue
            elif event.source == 'liq_flow_on_the_run':
                current_otr_cusip = event_attributes['id_liq_flow_on_the_run_otr_cusip']
            else:
                # for now, skip all other event sources
                # later, add these to the feature vector
                irregularity.skipped_event(
                    'event source %s not yet processed' % event.source,
                    event,
                )
                continue

        # do no other work until we reach the start date
        if event.date() < start_predictions:
            msg = 'skipped more than event feature extraction because before start date %s' % control.arg.start_date
            counter[msg] += 1
            continue

        counter['events on or after start-predictions date'] += 1

        if True:  # attempt to create a feature vector and use it to test and train
            if not event.is_trace_print_with_cusip(control.arg.cusip):
                irregularity.no_testing('event not trace print for primary cusip', event)
                continue
            if not feature_vector_maker.have_all_event_attributes():
                irregularity.no_feature_vector('not all event attributes are available', event)
                continue
            # event was for the primary cusip
            feature_vector = FeatureVector(
                creation_event=event,
                payload=feature_vector_maker.all_features(),
                reclassified_trade_type=event_attributes.value['id_trace_reclassified_trade_type'],
            )
            rtt = feature_vector.reclassified_trade_type
            print 'about to call maybe_test_and_train', rtt
            ensemble_prediction, errs_test, trained_experts, errs_train = (
                test_train[rtt].maybe_test_and_train(
                    current_event=event,
                    feature_vector=feature_vector,
                )
            )
            if True:  # process any errors
                if errs_test is not None:
                    print 'reclassified trade type', rtt
                    for err in errs_test:
                        irregularity.no_test(err, event)
                if errs_train is not None:
                    for err in errs_train:
                        irregularity.no_train(err, event)
                # pdb.set_trace()
            if errs_test is None:  # process the test results (the ensemble prediction)
                print 'processing an ensemble prediction', rtt
                output_actions.ensemble_prediction(ensemble_prediction, rtt)
                output_importances.ensemble_prediction(ensemble_prediction, rtt)
                output_signals.ensemble_prediction(ensemble_prediction, rtt)
                output_trace.ensemble_prediction(ensemble_prediction, rtt)

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
                counter['ensemble predictions made'] += 1
                counter['ensemble predictions made %s' % rtt] += 1
            if errs_train is None:  # process the training results (the trained_experts)
                print 'processing trained experts', rtt
                output_trace.trained_experts(
                    trained_experts,
                    rtt,
                )
                counter['experts trained'] += 1
                counter['experts trained %s' % rtt] += 1

        if counter['ensemble predictions made'] > 0:
            print 'for now, stopping early'
        gc.collect()
        continue
        if False:  # OLD VERSION START HERE
            # set current_otr_cusip for use below
            event_feature_makers = None  # avoid pyflake warnings
            event_feature_values = None
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
                simulated_clock.handle_event()
                output_trace.event_features_created_for_source_trace(simulated_clock.datetime, event)
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
                simulated_clock.handle_event()
                output_trace.event_features_created_for_source_not_trace(simulated_clock.datetime, event)
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
            state_all_feature_vectors = None  # avoid pyflake warning
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
            simulated_clock.handle_event()
            output_trace.new_feature_vector_created(
                simulated_clock.datetime,
                feature_vector,
            )
            counter['feature vectors created'] += 1

        if True:  # maybe test the ensemble model
            state_all_tested_ensembles = None  # avoid pyflake warnings
            state_all_trained_sets_of_experts = None
            for trade_type in ('B', 'S'):
                start_wallclock = datetime.datetime.now()
                tested_ensemble1, errs = maybe_test_ensemble(
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
                    simulated_clock.handle_event()
                    action_identifier = action_identifiers.next_identifier(simulated_clock.datetime)
                    tested_ensemble = tested_ensemble1._replace(
                        action_identifier=action_identifier,
                        elapsed_wallclock_seconds=(datetime.datetime.now() - start_wallclock).total_seconds(),
                        simulated_datetime=simulated_clock.datetime,
                    )
                    state_all_tested_ensembles.append(
                        trade_type,
                        tested_ensemble,
                    )
                    output_actions.ensemble_prediction(tested_ensemble, trade_type)
                    output_signals.ensemble_prediction(tested_ensemble, trade_type)
                    output_trace.test_ensemble(
                        simulated_clock.datetime,
                        tested_ensemble,
                        trade_type,
                    )
                    # write the explanation (which is lines of plain text)
                    explanation_filename = ('%s.txt' % action_identifier)
                    explanation_path = os.path.join(control.path['dir_out'], 'ensemble-predictions', explanation_filename)
                    with open(explanation_path, 'w') as f:
                        for line in tested_ensemble.explanation:
                            f.write(line)
                            f.write('\n')
                    counter['tested %s ensemble' % trade_type] += 1

        if True:  # maybe train the experts
            state_all_trained_sets_of_experts = None
            last_expert_training_time = None
            for trade_type in ('B', 'S'):
                start_wallclock = datetime.datetime.now()
                trained_experts1, errs = maybe_train_experts(
                    control=control,
                    ensemble_hyperparameters=ensemble_hyperparameters,
                    creation_event=event,
                    feature_vectors=state_all_feature_vectors[trade_type],
                    last_expert_training_time=last_expert_training_time,
                    simulated_datetime=simulated_clock.datetime,
                )
                if errs is not None:
                    for err in errs:
                        irregularity.no_training(err, event)
                else:
                    trained_experts = trained_experts1._replace(
                        elapsed_wallclock_seconds=(datetime.datetime.now() - start_wallclock).total_seconds(),
                    )
                    state_all_trained_sets_of_experts.append(
                        trade_type,
                        trained_experts,
                    )
                    simulated_clock.handle_event()
                    output_trace.trained_experts(
                        simulated_clock.datetime,
                        trained_experts,
                        trade_type,
                    )
                    last_expert_training_time = simulated_clock.datetime
                    counter['sets of %s experts trained' % trade_type] += 1

        if False and counter['sets of B experts trained'] >= 1:
            print 'for now, stopping base on number of accuracies determined'
            break

        if False and counter['tested B ensemble'] >= 1:
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
