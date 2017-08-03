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
    def __init__(self, control, event, last_created_features_not_trace, last_created_features_trace):
        assert isinstance(control, Bunch)
        assert isinstance(event, Event)
        assert isinstance(last_created_features_not_trace, dict)
        assert isinstance(last_created_features_trace, dict)

        self.creation_event = copy.copy(event)
        self.payload = copy.copy(last_created_features_trace[control.arg.cusip])[1]

    def reclassified_trade_type(self):
        # TODO: use the actual reclassified trade type, when it exists
        assert self.creation_event.id.source.startswith('trace_')
        return self.creation_event.payload['trade_type']

    def __repr__(self):
        return 'FeatureVector(creation_event.id=%s)' % self.creation_event.id


class TargetVector(object):
    def __init__(self, event, target_value):
        assert isinstance(event, Event)
        assert isinstance(target_value, float)
        self.creation_event = copy.copy(event)
        self.payload = target_value

    def __repr__(self):
        return 'TargetVector(creation_event.id=%s, payload=%f)' % (self.creation_event.id, self.payload)


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
        return 'Event(%s, %d values)' % (self.id, len(self.payload))


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

        path = seven.path.sort_trace_file(issuer)
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


def select_relevant(target, training_feature_vectors):
    'return (List[training_feature_vector], List[target_vector]'
    assert target == 'oasspread'
    last_reclassified_trade_type = training_feature_vectors[-1].reclassified_trade_type()
    feature_vectors = [
        training_feature_vector
        for training_feature_vector in training_feature_vectors
        if training_feature_vector.reclassified_trade_type() == last_reclassified_trade_type
    ]
    # the targets have the oasspread of the next trade
    target_vectors = []
    for i in range(0, len(feature_vectors) - 1):
        for j in range(i + 1, len(feature_vectors)):
            oasspread = float(feature_vectors[j].creation_event.payload['oasspread'])
            target_vector = TargetVector(feature_vectors[j].creation_event, oasspread)
            target_vectors.append(target_vector)
    return feature_vectors[:-1], target_vectors


def make_trained_expert_models(control, training_feature_vectors, training_target_vectors):
    'return Dict[model_spec, fitted_model]'
    pdb.set_trace()
    assert control.arg.target == 'oasspread'

    grid = seven.HpGrids.construct_HpGridN(control.arg.hpset)
    result = {}
    for model_spec in grid.iter_model_specs():
        model_constructor = (
            seven.models2.ModelNaive if model_spec.name == 'n' else
            seven.models2.ModelElasticNet if model_spec.name == 'en' else
            seven.models2.ModelRandomForests if model_spec.name == 'rf' else
            None
        )
        model = model_constructor(model_spec, control.random_seed)
        try:
            model.fit(training_feature_vectors, training_target_vectors)
        except seven.models2.ExceptionFit as e:
            seven.logging.warning('could not fit %s: %s' % (model_spec, e))
            continue
        result[model_spec] = model  # the fitted model

    pdb.set_trace()
    pass


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

    last_created_features_not_trace = {}   # Dict[event reader class name, dict]
    last_created_features_trace = {}       # Dict[cusip, dict]
    trace_feature_maker = {}               # Dict[cusip, feature_maker2.Trace]
    total_debt_feature_maker = seven.feature_makers2.TotalDebt(control.arg.issuer, control.arg.cusip)

    # determine number of historic trades needed to train all the models
    current_otr_cusip = None
    # TODO: determine queue length from the hpset
    errors = {}  # Dict[trade_type, float]
    training_feature_vectors = collections.deque(
        [],
        maxlen=10 * make_max_n_trades_back(control.arg.hpset),  # keep a lot of recent feature vectors
    )
    trained_experts = collections.deque(  # List[(event, Dict[model_spec, fitted model])]
        [],
        maxlen=100,
    )
    prediction_b = None
    prediction_s = None
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

        # extract features from the event
        errs = None
        if isinstance(event.id, seven.EventId.TraceEventId):
            # determine accuracy of the ensemble model prediction, if they are available
            # for now, there is one model and it has put its predictions in prediction_b and prediction_s
            actual = None
            try:
                actual = float(event.payload['oasspread'])
            except:
                no_accuracy('actual oasspread is not a float', event)
            if actual is not None and event.payload['cusip'] == control.arg.cusip:
                # determine accuracy, if we have predictions
                trade_type = event.payload['trade_type']
                if trade_type == 'B':
                    if prediction_b is None:
                        no_accuracy('no predicted B oasspread', event)
                    else:
                        error = actual - prediction_b
                        errors['B'] = error
                        seven.logging.info('ACTUAL B %f ERROR B %f as of event %s' % (actual, error, event.id))
                        action_signal.actual_b(event, actual)
                        counter['accuracies determined b'] += 1
                elif trade_type == 'S':
                    if prediction_s is None:
                        no_accuracy('no predicted S oasspraead', event)
                    else:
                        error = actual - prediction_s
                        errors['S'] = error
                        seven.logging.info('ACTUAL S %f ERROR S %f as of event %s' % (actual, error, event.id))
                        action_signal.actual_s(event, actual)
                        counter['accuracies determined s'] += 1

            # build features for all cusips because we don't know which will be the OTR cusips
            # until we see OTR events
            cusip = event.payload['cusip']
            debug = False
            if cusip not in trace_feature_maker:
                # the feature makers accumulate state
                # we need one for each cusip
                trace_feature_maker[cusip] = seven.feature_makers2.Trace(control.arg.issuer, cusip)
            features, errs = trace_feature_maker[cusip].make_features(event.id, event.payload, debug)
            assert (isinstance(features, dict) and errs is None) or (features is None and isinstance(errs, list))
            if errs is None:
                last_created_features_trace[cusip] = (event, features)
                counter['cusip feature set created'] += 1
            else:
                event_not_usable(errs, event)
                continue
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

        # test (=predict), if we have a model
        if len(trained_experts) > 0 and False:  # TODO: re-enable this code, it should predict
            assert trained_experts == 'naive'
            # predict, using the naive model
            # the naive model predicts the most recent oasspread will be the next oasspread we see
            # predit only if we have the query cusip
            assumed_market_spread = 6  # ref: Kristina 170802 for the AAPL cusip ... AJ9
            if feature_vector['trace_trade_type_is_B'] == 1:
                prediction_b = feature_vector['trace_oasspread']
                seven.logging.info('PREDICTION B %f as of event %s' % (prediction_b, event.id))
                action_signal.predictions(event, prediction_b, prediction_b + assumed_market_spread)
                counter['predictions made b'] += 1
            elif feature_vector['trace_trade_type_is_S'] == 1:
                prediction_s = feature_vector['trace_oasspread']
                seven.logging.info('PREDICTION S %f as of event %s' % (prediction_s, event.id))
                action_signal.predictions(event, prediction_s - assumed_market_spread, prediction_s)
                counter['predictions made s'] += 1
            else:
                # it was a D trade
                # for now, we don't know the reclassified trade type
                no_prediction('D trade', event)

        # train new models using the new feature vector
        # the models decide how much history to use in their training
        # for now, train whenever we can
        # later, we may decide to train occassionally
        # the training could run assynchronously and then feed a trained-model event into the system
        ce = feature_vector.creation_event
        if ce.id.source.startswith('trace_') and ce.payload['cusip'] == control.arg.cusip:
            pass
        else:
            no_training('event from source %s is not trace print of query cusip' % ce.id.source, event)
            continue
        training_feature_vectors.append(feature_vector)

        relevant_training_features, relevant_training_targets = select_relevant(
            control.arg.target,
            training_feature_vectors,
        )
        assert len(relevant_training_features) == len(relevant_training_targets)

        if len(relevant_training_features) == 0:
            no_training(
                'zero training vectors for %s' % feature_vector,
                event,
            )
            continue

        pdb.set_trace()
        new_expert_models = make_trained_expert_models(
            control,
            relevant_training_features,
            relevant_training_targets,
        )
        trained_experts.append((event, new_expert_models))
        counter['trainings completed'] += 1

        counter['event loops completed'] += 1
        print 'finished %d complete event loops' % counter['event loops completed']

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
