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
    def __init__(self, control, creation_event, last_created_features_not_trace, last_created_features_trace):
        assert isinstance(control, Bunch)
        assert isinstance(creation_event, Event)
        assert isinstance(last_created_features_not_trace, dict)
        assert isinstance(last_created_features_trace, dict)

        self.creation_event = copy.copy(creation_event)

        # build up the features from all the events except for the trace prints
        self._reclassified_trade_type = None
        all_features = {}
        for event_reader_name, event_and_features in last_created_features_not_trace.iteritems():
            input_event, features = event_and_features
            for k, v in features.iteritems():
                all_features[k] = v

        # buid up the features for all the events for the trace print primary and OTR cusips
        # for now, just append the features for the primary cusip
        # TODO: also include features for the OTR cusip
        input_event, features = last_created_features_trace[control.arg.cusip]
        for k, v in features.iteritems():
            if k == 'id_trace_event_payload':
                self._reclassified_trade_type = v['reclassified_trade_type']
            all_features[k] = v
        assert self._reclassified_trade_type is not None
        self.payload = all_features

    def __repr__(self):
        return 'FeatureVector(creation_event.id=%s, rct=%s, n features=%d)' % (
            self.creation_event.id,
            self._reclassified_trade_type,
            len(self.payload),
        )

    def reclassified_trade_type(self):
        'return the reclassified trade type from the trace print event for the query cusip'
        return self._reclassified_trade_type


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

    def predictions(self, event, prediction_b, prediction_s):
        self.actions.action(event, 'prediction_b', prediction_b)
        self.actions.action(event, 'prediction_s', prediction_s)
        self.signal.predictions(event, prediction_b, prediction_s)

    def actual_b(self, event, actual_b):
        # self.actions.action(event, 'actual_b', actual_b)
        # TODO: also record standard deviation
        self.signal.actual_b(event, actual_b)

    def actual_s(self, event, actual_s):
        # self.actions.action(event, 'actual_s', actual_s)
        # TODO: also record standard deviation
        self.signal.actual_s(event, actual_s)

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
        event_id = str(event.id)
        if event_id == self.last_event_id:
            self.last_action_suffix += 1
        else:
            self.last_action_suffix = 1
        action_id = 'ml_%s_%d' % (event_id, self.last_action_suffix)
        d = {
            'action_id': action_id,
            'action_type': action_type,
            'action_value': action_value,
        }
        self.dict_writer.writerow(d)
        self.last_event_id = event_id

    def close(self):
        self.file.close()


class Signal(object):
    'produce the signal as a CSV file'
    def __init__(self, path):
        self.path = path
        self.file = open(path, 'wb')
        self.field_names = [
            'datetime', 'event_source', 'event_source_id',  # these columns are dense
            'prediction_b', 'prediction_s',                 # thse columns are sparse
            'actual_b', 'actual_s',                         # these columns are sparse
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

    def predictions(self, event, prediction_b, prediction_s):
        d = self._event(event)
        d.update({
            'prediction_b': prediction_b,
            'prediction_s': prediction_s,
        })
        self.dict_writer.writerow(d)

    def actual_b(self, event, actual):
        d = self._event(event)
        d.update({
            'actual_b': actual
        })
        self.dict_writer.writerow(d)

    def actual_s(self, event, actual):
        d = self._event(event)
        d.update({
            'actual_s': actual
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

    last_created_features_not_trace = {}   # Dict[event reader class name, dict]
    last_created_features_trace = {}       # Dict[cusip, dict]
    trace_feature_maker = {}               # Dict[cusip, feature_maker2.Trace]
    total_debt_feature_maker = seven.feature_makers2.TotalDebt(control.arg.issuer, control.arg.cusip)

    # determine number of historic trades needed to train all the models
    current_otr_cusip = None

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
    while True:
        try:
            event = event_queue.next()
            counter['events read']
        except StopIteration:
            break  # all the event readers are empty

        if event.id.datetime() < start_date:
            msg = 'skipped because before start date'
            if counter[msg] == 0:
                print 'skipping events before the start date %s' % control.arg.start_date
            counter[msg] += 1
            continue

        print 'next event', event,
        if event.id.source.startswith('trace_'):
            print 'cusip=%s' % event.payload['cusip']
        else:
            print
        counter['events examined'] += 1

        # print summary of global state
        def print_state_list(name, value):
            print '%30s B %6d S %6d' % (
                name,
                len(value['B']),
                len(value['S']),
            )

        print_state_list('ensemble_predictions', ensemble_predictions)
        print_state_list('expert_accuracies', expert_accuracies)
        print_state_list('feature_vectors', feature_vectors)
        print_state_list('trained_experts', trained_experts)
        print 'processed %d events in %0.2f wallclock minutes' % (
            counter['events examined'] - 1,
            control.timer.elapsed_wallclock_seconds() / 60.0,
        )

        # extract features from the event
        errs = None
        if isinstance(event.id, seven.EventId.TraceEventId):
            # build features for all cusips because we don't know which will be the OTR cusips
            # until we see OTR events
            cusip = event.payload['cusip']
            debug = False
            if cusip not in trace_feature_maker:
                # the feature makers accumulate state
                # we need one for each cusip
                trace_feature_maker[cusip] = seven.feature_makers2.Trace(control.arg.issuer, cusip)
            features, errs = trace_feature_maker[cusip].make_features(event.id, event.payload, debug)
            if errs is None:
                last_created_features_trace[cusip] = (event, features)
                counter['cusip feature set created'] += 1
            else:
                event_not_usable(errs, event)
                continue

            # determine accuracy of the experts, if:
            # - we have predictions from the experts
            # - the trace event is for the primary cusip
            # build expert_accuracies[model_spec, float] that sum to one across mmodel_specs

            # determine the variance in the experts' predictions relative to the ensemble prediction
            # determine variances, by this algorithm
            # source: Dennis Shasha, "weighting the variances", email Jun 23, 2017 and subsequent discussions in email
            # let m1, m2, ..., mk = the models
            # let p1, p2, ..., pk = their predictions
            # let w1, w2, ..., wk = their weights (that sum to 1)
            # let e = prediction of ensemble = (w1*p1 + w2*p2 + ... + wk*pk) = weighted mean
            # let delta_k = pk - e
            # let variance = w1 * delta_1^2 + w2 * delta_2^2 + ... + wk * delta_k^2)
            # let standard deviation = sqrt(variance)

            reclassified_trade_type = event.payload['reclassified_trade_type']
            relevant_expert_predictions = expert_predictions[reclassified_trade_type]
            if len(relevant_expert_predictions) > 0 and event.payload['cusip'] == control.arg.cusip:
                # detemine unnormalized weights
                actual = float(event.payload[control.arg.target])
                temperature_expert = ensemble_hyperparameters.weight_temperature
                unnormalized_weights = collections.defaultdict(float)  # Dict[model_spec, float]
                sum_unnormalized_weights = 0.0
                for expert in relevant_expert_predictions:
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

                # normalize the weights
                normalized_weights = {}  # Dict[model_spec, float]
                for model_spec, unnormalized_weight in unnormalized_weights.iteritems():
                    normalized_weights[model_spec] = unnormalized_weight / sum_unnormalized_weights
                if True:
                    print 'model_spec, where normalized weights > mean normalized weight'
                    mean_normalized_weight = 1.0 / len(normalized_weights)
                    sum_normalized_weights = 0.0
                    for model_spec, normalized_weight in normalized_weights.iteritems():
                        sum_normalized_weights += normalized_weight
                        if normalized_weight > mean_normalized_weight:
                            print '%30s %8.6f' % (str(model_spec), normalized_weight)
                    print 'sum of normalized weights', sum_normalized_weights

                expert_accuracies[reclassified_trade_type] = ExpertAccuracies(
                    event=event,
                    dictionary=copy.copy(normalized_weights),
                )
                # # produce the ensemble prediction
                # ensemble_prediction = 0.0
                # for expert in relevant_expert_predictions:
                #     for model_spec, prediction in expert.expert_predictions.iteritems():
                #         ensemble_prediction = (
                #             weight_experts[expert.event.id] *
                #             normalized_weights[model_spec] *
                #             prediction )
                #     pdb.set_trace()
                # pdb.set_trace()
                # # determine variance of experts relative to the ensemble prediction
                # variance = 0.0
                # for expert in relevant_expert_predictions:
                #     for model_spec, prediction in expert.expert_predictions.iteritems():
                #         delta = ensemble_prediction - prediction
                #         weighted_delta = (
                #             weight_experts[expert.event.it] *
                #             normalized_weights[model_spec] *
                #             delta
                #         )
                #         variance += weighted_delta * weighted_delta
                #     pdb.set_trace()
                # pdb.set_trace()
                # standard_deviation = math.sqrt(variance)
                # # produce the signal

                seven.logging.info('new accuracties from event id %s' % event.id)

            # actual = None
            # try:
            #     actual = float(event.payload['oasspread'])
            # except:
            #     no_accuracy('actual oasspread is not a float', event)
            # if actual is not None and event.payload['cusip'] == control.arg.cusip:
            #     # determine accuracy, if we have predictions
            #     trade_type = event.payload['trade_type']
            #     if trade_type == 'B':
            #         if prediction_b is None:
            #             no_accuracy('no predicted B oasspread', event)
            #         else:
            #             error = actual - prediction_b
            #             errors['B'] = error
            #             seven.logging.info('ACTUAL B %f ERROR B %f as of event %s' % (actual, error, event.id))
            #             action_signal.actual_b(event, actual)
            #             counter['accuracies determined b'] += 1
            #     elif trade_type == 'S':
            #         if prediction_s is None:
            #             no_accuracy('no predicted S oasspraead', event)
            #         else:
            #             error = actual - prediction_s
            #             errors['S'] = error
            #             seven.logging.info('ACTUAL S %f ERROR S %f as of event %s' % (actual, error, event.id))
            #             action_signal.actual_s(event, actual)
            #             counter['accuracies determined s'] += 1

        elif isinstance(event.id, seven.EventId.OtrCusipEventId):
            # change the on-the-run cusip, if the event is for the primary cusip
            otr_cusip = event.payload['otr_cusip']
            primary_cusip = event.payload['primary_cusip']
            if primary_cusip == control.arg.cusip:
                current_otr_cusip = otr_cusip
                counter['otr cusips identified for primary cusip'] += 1
            else:
                err = 'ignoring new OTR cusip %s for non-query cusip %s' % (otr_cusip, primary_cusip)
                event_not_usable([err], event)
                continue
        elif isinstance(event.id, seven.EventId.TotalDebtEventId):
            # this is a regularly-handled event
            features, errs = total_debt_feature_maker.make_features(event.id, event.payload)
            if errs is None:
                # TODO: generalize TotalDebt, so that code works for all non-trace events
                # source of event is the key (event.id.source)
                last_created_features_not_trace['TotalDebtEventReader'] = (event, features)
                counter['total debt feature set created'] += 1
            else:
                event_not_usable(errs, event)
                continue
        else:
            print 'not yet handling', event.id
            pdb.set_trace()
        counter['event features created'] += 1

        # create feature vectors only for the query cusip
        # event_cusip = event.payload['cusip']
        # if event_cusip != control.arg.cusip:
        #     counter['skipped after event features created, as wrong cusip %s' % event_cusip] += 1
        #     continue

        # check whether we have all the data needed to create the feature vector
        if control.arg.cusip not in last_created_features_trace:
            # we must know the features from the query cusip
            no_feature_vector('no features for the query cusip %s' % control.arg.cusip, event)
            continue
        if current_otr_cusip is None:
            # we must know the OTR cusip
            no_feature_vector('no otr cusip identified as of event id %s' % event.id, event)
            continue
        if current_otr_cusip not in last_created_features_trace:
            # we must have features for the OTR cusip
            no_feature_vector('no features for OTR cusip %s' % current_otr_cusip, event)
            continue
        if 'TotalDebtEventReader' not in last_created_features_not_trace:
            no_feature_vector('no features for TotalDebtEventReader', event)
            continue

        # create feature vector
        feature_vector = FeatureVector(
            control,
            event,
            last_created_features_not_trace,
            last_created_features_trace,
        )

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
            # TODO: create ensemble predictions, if we have accuracies for the experts
            if len(expert_accuracies[reclassified_trade_type]) > 0:
                EnsemblePrediction(None, None)  # TODO: generate the ensemble prediction using the accuracies

            # OLD BELOW ME
            # assert trained_experts == 'naive'
            # # predict, using the naive model
            # # the naive model predicts the most recent oasspread will be the next oasspread we see
            # # predit only if we have the query cusip
            # assumed_market_spread = 6  # ref: Kristina 170802 for the AAPL cusip ... AJ9
            # if feature_vector['trace_trade_type_is_B'] == 1:
            #     prediction_b = feature_vector['trace_oasspread']
            #     seven.logging.info('PREDICTION B %f as of event %s' % (prediction_b, event.id))
            #     action_signal.predictions(event, prediction_b, prediction_b + assumed_market_spread)
            #     counter['predictions made b'] += 1
            # elif feature_vector['trace_trade_type_is_S'] == 1:
            #     prediction_s = feature_vector['trace_oasspread']
            #     seven.logging.info('PREDICTION S %f as of event %s' % (prediction_s, event.id))
            #     action_signal.predictions(event, prediction_s - assumed_market_spread, prediction_s)
            #     counter['predictions made s'] += 1
            # else:
            #     # it was a D trade
            #     # for now, we don't know the reclassified trade type
            #     no_prediction('D trade', event)

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
        print '%-50s: %6d' % (k, counter[k])
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
