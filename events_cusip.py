'''read queue events.{cusip}, write feature set to queue features.{cusip}.{expert}

One instance of this program reads one event queue. That event queue must containing
all of the trace prints for a primary cusip as well as all of the trace prints for
any on-the-run (OTR) cusips for the primary cusip.

The program starts and then reads the configuration file specified on its invocation
command line. Included in the configuration file is the primary cusip for which the
program is constructing feature vectors.

The configuration file specifies the input source, which is either
- a file in the file system (used for development and some testing), or
- a queue in RabbitMQ (used for some testing and production).

The program reads the input until end of file. The messages in the input cause the_date
program to take the actions indicated below.

- BackToZero
    Reset the program's execution state to be what it was 
    just after the configuration parameters were read.

- SetPrimaryOTRs
    Remember the primay cusip on 0 to N OTR cusips for it.
    The feature vectors have information from a primary cusip and from
    0 or more OTR cusips. The number of OTR cusips can vary by primary
    cusip but most not change once the program is invoked, or the feature
    vector will get messed up.

- SetVersion
    The program passes this message downstream. The upstream
    program uses it to declare the version of the data or program (or
    anything else) that it is using.

- TracePrint
    This message contains all the information from trace prints that is actually
    used. The functions defined in the module features.py use the TracePrint
    messages to create portions of features vectors. Those portions are sticked
    together by this program to create an entire feature vector. Multiple feature
    vectors are acumulated to form the training data. Each feature vector is used
    to test the experts by asking them to predict and determining the accuracy
    of their predictions.

    The TracePrint message can replace a previous trace print message simply by
    using the issuepriceid of the previous message. That is how corrections are
    handled.

   Each TracePrint message has a field "cancellation_probability". If the value
   of this field is above a threshold specified in the configuration file,

- TracePrintCancel
    This message cancels a trace print message by naming its issuepriceid. Any
    feature vectors

- (other events that create features)
    For now, only trace prints create features. Later, other events, like equity tickers
    can also create features.

This program writes the following messages to queues features.{cusip}.{expert}

- SetVersion
    Any SetVersion messages from upstream are passed along without modification.
    An addition SetVersion message is created that contains the version of the
    machine learning algorithm.

- TestFeatureVector
    This message contains a test vector that is passed to experts. The experts
    use their trained models to make a prediction. The experts pass those productions
    on to the ensemble model. The ensemble model determines the accuracy of the
    predictions.

- TrainingFeatureVectors
    This message contains a list of feature vectors. The list is passed to the
    experts which then train themselves on the data. The trained experts are then
    able to make predictions.

- {various messages containing diagnostic information}
    These messages are the log file for the program. The messages are used
    for debugging and performance assessments.

HOW TO ADD A FEATURE

Follow these steps to add a feature (such as equity ticker prices)

1. Design and build a new message and define it in message.py. For example, the
   the message might be message.EquityTicker. All message types are classes.

2. Design and write the feature-creation function in features.py. This function
   reads a list of all the messages seens so far and produces a portion of a
   feature vector. When creating the portion of a feature vector, it has access
   to all the features read so far, both the new content in the message
   just designed and all of the previous messages. Another output of the function
   is the list of messages not used to buid the feature vector. The caller uses
   this to determine which messages are not used by any of the feature builders.
   Those messages are thrown away, so as to keep the memory requirements of this
   program roughly constant.

3. Modify this program to use the new feature. The code contains a list of features
   that are being used. Find that statement and modify it to include the new features.
   (Later, this program can be modified to read the feature sets from the configuration
   file.)

4. Run test cases (back tests) to assess the extent to which the new feature
   improves prediction accuracy.
'''
import copy
import datetime
import pdb
from pprint import pprint as pp
import sys
import typing

import exception
import features
import HpGrids
import log
import machine_learning
import message
import shared_message
import shared_queue


def old_trace_attributes():
    event_attributes = EventAttributes.EventAttributes(
        id_trace_event=copy.copy(event),
        id_trace_reclassified_trade_type=reclassified_trade_type,  # from the event record
        trace_B_oasspread=self._oasspreads['B'][1],
        trace_B_oasspread_divided_by_prior_B=self._oasspreads['B'][1] / self._oasspreads['B'][0],
        trace_B_oasspread_less_prior_B=self._oasspreads['B'][1] - self._oasspreads['B'][0],
        trace_S_oasspread=self._oasspreads['S'][1],
        trace_S_oasspread_divided_by_prior_S=self._oasspreads['S'][1] / self._oasspreads['S'][0],
        trace_S_oasspread_less_prior_S=self._oasspreads['S'][1] - self._oasspreads['S'][0],
    )
    return event_attributes, None


class OldFeatureVectorMaker(object):
    def __init__(self, invocation_args):
        self._invocation_args = invocation_args

        self._event_attributes_cusip_otr = None
        self._event_attributes_cusip_primary = None
        self._reclassified_trade_type = None

    def all_features(self):
        'return dict with all the features'
        # check that there are no duplicate feature names
        def append(all_features, tag, attributes):
            for k, v in attributes.value.items():
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

    def reclassified_trade_type(self):
        assert self._reclassified_trade_type is not None  # throws, if there was no primary cusip
        return self._reclassified_trade_type

    def update_cusip_otr(self, event, event_attributes):
        assert isinstance(event_attributes, seven.EventAttributes.EventAttributes)
        self._event_attributes_cusip_otr = event_attributes

    def update_cusip_primary(self, event, event_attributes):
        assert isinstance(event_attributes, seven.EventAttributes.EventAttributes)
        self._reclassified_trade_type = event.reclassified_trade_type()
        self._event_attributes_cusip_primary = event_attributes

        
class TracePrintSequence:
    def __init__(self):
        self._msgs = []

    def accumulate(self, msg):
        assert isinstance(msg, message.TracePrint)
        print('todo: handle replacement messages')
        # a replacement message has the same issuepriceid
        self._msgs.append(msg)
        # for now, save everything; later, discard msg that we will never use

    def feature_vectors(self,
                        cusips: typing.List[str],  # primary, otr1, otr2, ...
                        n_feature_vectors: int,
                        required_reclassified_trade_type: str,
                        trace=False,
                        ):
        'return List[feature_vectors] with length up to n_feature_vectors'
        set_trace = machine_learning.make_set_trace(trace)
        
        def find_cusip(msgs, cusip):
            'return first message with the specified cusip'
            if False and trace:
                pdb.set_trace()
            for msg in msgs:
                if msg.cusip == cusip:
                    return msg
            pdb.set_trace()
            raise exception.NoMessageWithCusip(
                cusip=cusip,
                msgs=msgs,
            )
            
        def find_cusip_rtt(msgs, cusip, rtt):
            'return first msg with cusip and remaining messages'
            if False and trace:
                pdb.set_trace()
            for i, msg in enumerate(msgs):
                if msg.cusip == cusip and msg.reclassified_trade_type == rtt:
                    return msg, msgs[i + 1:]
            pdb.set_trace()
            raise exception.NoPriorEventWithCusipAndRtt(
                cusip=cusip,
                rtt=rtt,
                msgs=msgs,
            )
        
        def cusip_features(msgs, cusip):
            'return dict, unused_msgs'
            if False and trace:
                pdb.set_trace()
            result_dict = {}
            result_unused_messages = msgs
            for rtt in ('B', 'S'):
                first_msg, first_other_messages = find_cusip_rtt(msgs, cusip, rtt)
                second_msg, second_other_messages = find_cusip_rtt(first_other_messages, cusip, rtt)
                result_dict['id_first_%s_message_source' % rtt] = first_msg.source
                result_dict['id_first_%s_message_identifier' % rtt] = first_msg.identifier
                result_dict['id_second_%s_message_source' % rtt] = second_msg.source
                result_dict['id_second_%s_message_identifier' % rtt] = second_msg.identifier
                result_dict['trace_%s_oasspread' % rtt] = first_msg.oasspread
                result_dict['trace_%s_oasspread_less_prior_%s' % (rtt, rtt)] = (
                    first_msg.oasspread - second_msg.oasspread
                    )
                result_dict['trace_%s_oasspread_divided_by_prior_%s' % (rtt, rtt)] = (
                        first_msg.oasspread / second_msg.oasspread if second_msg.oasspread != 0.0 else
                        100.0
                        )
                if len(second_other_messages) < len(result_unused_messages):
                    result_unused_messages = copy.copy(second_other_messages)
            return result_dict, result_unused_messages
            
        def feature_vector(msgs, cusips, required_reclassified_trade_type):
            'return (feature_vector, unused messages)'
            def key_with_cusip_info(k, i):
                first, *others = k.split('_')
                return '%s_%s_%s' % (
                    first,
                    'primary' if i == 0 else 'otr%d' % i,
                    k[len(first) + 1:],
                    )
            
            if False and trace:
                pdb.set_trace()
            vp = machine_learning.make_verbose_print(False)
            result_unused_messages = msgs
            # NOTE: these field names are used by message.FeatureVectors.__repr__()
            # Don't change them here unless you also change them there
            trace_print_with_oasspread, _ = find_cusip_rtt(
                msgs,
                cusips[0],
                required_reclassified_trade_type,
                )
            vp('trace_print_with_oasspread', trace_print_with_oasspread)
            result_feature_vector = {
                'id_target_oasspread': trace_print_with_oasspread.oasspread,
                'id_target_reclassified_trade_type': required_reclassified_trade_type,
                'id_trigger_source': msgs[0].source,
                'id_trigger_identifier': msgs[0].identifier,
                'id_trigger_reclassified_trade_type': msgs[0].reclassified_trade_type,
                'id_trigger_event_datetime': msgs[0].datetime,
                }
            for i, cusip in enumerate(cusips):
                cf, unused_messages = cusip_features(msgs, cusip)
                vp('cusip_features result', i, cusip, len(cf), len(unused_messages))
                result_feature_vector['id_feature_vector_%s' % ('primary' if i == 0 else 'otr%d' % i)] = cusip
                for k, v in cf.items():
                    # adjust the keys to reflect whether the features are from the primary or OTR cusip
                    result_feature_vector[key_with_cusip_info(k, i)] = v
                if len(unused_messages) < len(result_unused_messages):
                    result_unused_messages = copy.copy(unused_messages)
            return result_feature_vector, result_unused_messages
        
        def loop(msgs):
            'return (feature_vectors, unused messages)'
            vp = machine_learning.make_verbose_print(False)
            set_trace()
            feature_creators = (
                ('trace_print', features.trace_print),
                )
            result_feature_vectors = []
            result_unused = msgs
            pdb.set_trace()
            for i in range(0, n_feature_vectors, 1):
                msgs_to_be_used = msgs[i:]
                all_features = features.Features()
                for feature_creator in feature_creators:
                    for cusip in cusips:
                        try:
                            cusip_features, unused = feature_creator[1](msgs_to_be_used, cusip)
                        except exception.NoFeatures as e:
                            raise exception.Features('cusip %s, %s' % (cusip, e.msg))
                        if len(unused) < len(result_unused):
                            result_unused = copy.copy(unused)
                        # update feature names to incorporate the cusip
                        for k, v in cusip_features.items():
                            key = (
                                'id_%s_%s' (cusip, k[3:]) if k.startwith('id_') else
                                '%s_%s_%s' (feature_creator[0], cusip, k)
                            )
                            all_features.add(key, v)
                continue   # bypass old code, for now
                # try:
                #     fv, unused = feature_vector(msgs_to_be_used, cusips, required_reclassified_trade_type)
                #     vp('loop %d: fv trigger identifier: %s len(msgs): %d, len(unused): %d' % (
                #         i,
                #         fv['id_trigger_identifier'],
                #         len(msgs_to_be_used),
                #         len(unused),
                #     ))
                #     if False and i % 10 == 1:
                #         pdb.set_trace()
                #     result_feature_vectors.append(fv)
                #     if len(unused) < len(result_unused):
                #         result_unused = copy.copy(unused)
                # except exception.NoPriorEventWithCusipAndRtt as e:
                #     vp('stub: handle exception %s' % e)
                #     break
                # except exception.NoMessageWithCusip as e:
                #     vp('stub: handle exception %s' % e)
                #     break
            set_trace()
            return list(reversed(result_feature_vectors)), result_unused

        set_trace()
        assert n_feature_vectors >= 0
        result, unused = loop(list(reversed(self._msgs)))
        set_trace()
        self._msgs = self._msgs[len(unused):]
        if True:
            print('check that we get same results using possibly fewer messages')
            set_trace()
            # test: should get same feature vectors (the result)
            result2, unused2 = loop(list(reversed(self._msgs)))
            assert len(result) == len(result2)
            assert len(unused2) == 0
            for i, item in enumerate(result):
                item2 = result2[i]
                for k, v in item.items():
                    assert item2[k] == v
            set_trace()
        return result

        
def test_tps_make_feature_vectors():
    return  # the test code is deprecated
    # test 3 OTRs and consumption of entire set of messages
    vp = machine_learning.make_verbose_print(False)
    set_trace = machine_learning.make_set_trace(True)
    
    def make_trace_print_message(tp_info):
        index, cusip, rtt = tp_info
        return message.TracePrint(
            source='test_tps_make_feature_vectors',
            identifier=str(index),
            cusip=cusip,
            issuepriceid=str(index),
            datetime=datetime.datetime.now(),
            oasspread=float(index),
            trade_type=None,
            reclassified_trade_type=rtt,
            cancellation_probability=0.0,
            )
    
    trace_prints = (
        (0, 'p', 'B'),
        (1, 'o1', 'S'),
        (2, 'o2', 'S'),
        (3, 'p', 'S'),
        (4, 'o1', 'B'),
        (5, 'p', 'S'),
        (6, 'o2', 'B'),
        (7, 'o1', 'S'),
        (8, 'o1', 'B'),
        (9, 'p', 'S'),
        (10, 'o2', 'B'),
        (11, 'p', 'B'),
        (12, 'o2', 'S'),
    )
    tps = TracePrintSequence()
    for tp_info in trace_prints:
        msg = make_trace_print_message(tp_info)
        tps.accumulate(msg)
    assert len(tps._msgs) == len(trace_prints)
    set_trace()
    for rtt in ('B', 'S'):
        feature_vectors = tps.feature_vectors(
            cusips=('p', 'o1', 'o2'),
            n_feature_vectors=2,
            required_reclassified_trade_type=rtt,
            trace=False,
        )
        set_trace()
        assert len(feature_vectors) == 1
        for i, fv in enumerate(feature_vectors):
            print(rtt, i, fv['id_trigger_identifier'], fv['id_target_oasspread'])
            vp(fv)

    
def unittest(config):
    'run unit tests'
    test_tps_make_feature_vectors()


class Identifier:
    def __init__(self):
        self._last_datetime = datetime.datetime.now()
        self._suffix = 1

    def get_next(self):
        current_dt = datetime.datetime.now()
        if current_dt == self._last_datetime:
            self._suffix += 1
        else:
            self._suffix = 1
        self._last_datetime = current_dt
        return '%s_%02d' % (self._last_datetime, self._suffix)


def make_path_for_input(queue: str) -> str:
    pass


class InputPathMaker:
    def __init__(self, path_prefix):
        self._path_prefix = path_prefix  # the path identifies the exchange

    def make_path_for_input(self, queue):
        return '%s.%s.txt' % (self._path_prefix, queue)

    
class OutputPathMaker:
    def __init__(self, path_prefix):
        self._path_prefix = path_prefix

    def make_paths_for_output(self, exchange, routing_key):
        # put all routing keys into one file
        return ['%s.%s.txt' % (
            self._path_prefix,
            exchange,
            )]

    
def do_work(config):
    set_trace = machine_learning.make_set_trace(True)
    set_trace()
    n_required_feature_vectors = max(HpGrids.HpGrid5().n_trades_back_choices)
    feature_vector_identifiers = Identifier()
    n_required_feature_vectors = 300  # TODO: set based on model_specs
    creating_output = False
    all_trace_prints = []
    cusips = []  # [0] = primary, [1] = otr 1, [2] = otr 2, ...

    connection = shared_queue.PrimitiveBlockingConnection(
        path_for_input=InputPathMaker(config.get('in_path_prefix')).make_path_for_input,
        paths_for_output=OutputPathMaker(config.get('out_path_prefix')).make_paths_for_output,
        )
    channel = connection.channel()
    exchange = config.get('out_exchange')
    routing_key = 'events.%s.*' % config.get('primary_cusip')  # send every message to all of the experts
    input_queue = 'events.%s' % (config.get('primary_cusip'))
    channel.publish(
        exchange=exchange,
        routing_key=routing_key,
        body=str(shared_message.SetVersion(
            source='events_cusips.py',
            identifier=str(datetime.datetime.now()),
            what='machine_learning',
            version='1.0.0.0',
        )),
        )
    while True:
        try:
            s = channel.consume(input_queue)
        except StopIteration:
            break
        msg = shared_message.from_string(s)
        print('\n%r' % msg)
        if isinstance(msg, shared_message.BackToZero):
            pdb.set_trace()
            print('todo: implement BackToZero')
        elif isinstance(msg, shared_message.SetPrimaryOTRs):
            cusips = [msg.primary_cusip]
            for otr_cusip in msg.otr_cusips:
                cusips.append(otr_cusip)
        elif isinstance(msg, shared_message.SetVersion):
            channel.publish(
                exchange=exchange,
                routing_key=routing_key,
                body=str(msg),
                )
        elif isinstance(msg, shared_message.TracePrint):
            def make_feature_vector(msgs, cusips):
                'return (FeatureVector, unused msgs) or raise NoFeatures'
                if len(cusips) == 0:
                    log_msg = (
                        'A SetCusipOTRs message was not received before a StartOutput message was received'
                    )
                    log.critical(log_msg)
                    assert False, log_msg
                all_features = features.FeatureVector()
                shortest_unused = msgs
                for cusip in cusips:
                    for name, maker in (('trace_print', features.trace_print),):
                        try:
                            fv, unused = maker(msgs, cusip)
                        except exception.NoFeatures as e:
                            raise e
                        if len(unused) < len(shortest_unused):
                            shortest_unused = copy.copy(unused)
                        # rename the feature to use the name
                        # that makes the features unique
                        for k, v in fv.items():
                            key = (
                                'id_%s_%s_%s' % (name, cusip, k[3:]) if k.startswith('id_') else
                                '%s_%s_%s' % (name, cusip, k)
                                )
                            all_features[key] = v
                # create a unique identifier for the feature vector
                all_features['id_feature_vector'] = feature_vector_identifiers.get_next()
                # add id info from TracePrint for the first primary cusip
                for msg in msgs:
                    if msg.cusip == cusips[0]:
                        # found first trade for the primary cusip
                        all_features['id_primary_cusip'] = msg.cusip
                        all_features['id_primary_cusip_issuepriceid'] = msg.issuepriceid
                        all_features['id_primary_cusip_oasspread'] = msg.oasspread
                        all_features['id_primary_cusip_reclassified_trade_type'] = msg.reclassified_trade_type
                        break
                return all_features, shortest_unused

            # handle corrections
            made_correction = False
            for i, old_trace_print in enumerate(all_trace_prints):
                if old_trace_print.issuepriceid == msg.issuepriceid:
                    all_trace_prints[i] = msg  # replace the previous trace print message
                    made_correction = True
                    break
            if not made_correction:
                all_trace_prints.append(msg)

            if creating_output:
                all_msgs = list(reversed(all_trace_prints))
                try:
                    last_feature_vector, shortest_unused = make_feature_vector(all_msgs, cusips)
                except exception.NoFeatures as e:
                    # info ==> things are working as expected
                    log.info('unable to create even one feature vector: %s' % str(e))
                    continue
                channel.publish(
                    exchange=exchange,
                    routing_key='%s.expert.*' % config.get('primary_cusip'),  # {cusip}.expert.{model_spec}
                    body=str(message.Test(
                        source='events_cusip.py',
                        identifier=last_feature_vector['id_feature_vector'],
                        feature_vector=last_feature_vector,
                        )),
                    )

                pdb.set_trace()
                many_feature_vectors = [last_feature_vector]
                for i in range(1, len(all_msgs), 1):
                    try:
                        another_feature_vector, unused = make_feature_vector(all_msgs[i:])
                    except exception.NoFeatures as e:
                        log.info('unable to create %dth feature vector: %s' % (i, str(e)))
                        break
                    many_feature_vectors.append(another_feature_vector)
                    if len(unused) < len(shortest_unused):
                        shorted_unused = copy.copy(unused)
                pdb.set_trace()
                if len(many_feature_vectors) >= n_required_feature_vectors:
                    pdb.set_trace()
                    print('create B and S targets')
                    
                    print('reconcile targets with features')
                    channel.publish(
                        exchange=exchange,
                        routing_key='%s.expert.*' % config.get('primary_cusip'),
                        body=str(message.Train(
                            source='events_cusip.py',
                            identifier=last_feature_vector['id_feature_vector'],
                            feature_vectors=many_feature_vectors,  # TODO: include both B and S targets
                        )),
                    )
                # discard TracePrint objects that will never be used
                all_trace_prints = all_trace_prints[len(shorted_unused):]
        elif isinstance(msg, shared_message.TracePrintCancel):
            pdb.set_trace()
            print('todo: implement TracePrintCancel')
        elif isinstance(msg, shared_message.OutputStart):
            creating_output = True
        elif isinstance(msg, shared_message.OutputStop):
            pdb.set_trace()
            creating_output = False
        else:
            print(msg)
            print('%r' % msg)
            print('unrecognized input message type')
            pdb.set_trace()
                
    print('have read all messages')
    pdb.set_trace()
    connection.close()
                              
    
if __name__ == '__main__':
    if False:
        pdb
        pp
    machine_learning.main(
        argv=sys.argv,
        program='events_cusip.py',
        unittest=unittest,
        do_work=do_work,
        out_log='out_log',
        )
