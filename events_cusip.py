'''read queue events.{cusip}, write feature set to queue features.{cusip}.{expert}'''
import collections
import copy
import datetime
import pdb
from pprint import pprint as pp
import sys
import typing

import configuration
import exception
import machine_learning
import message
import queue
import verbose


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
    def __init__(self, n_feature_vectors=2):
        print('fixme: correctly set n_feature_vectors')
        self._n_feature_vectors = n_feature_vectors
        self._msgs = []

    def accumulate(self, msg):
        assert isinstance(msg, message.TracePrint)
        self._msgs.append(msg)
        # for now, save everything; later, discard msg that we will never use

    def feature_vectors(self,
                        cusips: typing.List[str],  # primary, otr1, otr2, ...
                        n_feature_vectors: int,
                        ):
        'return List[feature_vector] and truncate self._msgs, OR raise exception'
        def find_cusip_rtt(msgs, cusip, rtt):
            'return first msg with cusip and remaining messages'
            for i, msg in enumerate(msgs):
                if msg.cusip == cusip and msg.reclassified_trade_type == rtt:
                    return msg, msgs[i + 1:]
            raise exception.NoMessageWithCusipAndRtt(
                cusip=cusip,
                rtt=rtt,
                msgs=msgs,
            )
        
        def cusip_features(msgs, cusip):
            'return dict, unused_msgs'
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
            
        def find_cusip(msgs, cusip):
            'return first message with the specified cusip'
            for msg in msgs:
                if msg.cusip == cusip:
                    return msg
            else:
                raise exception.NoMessageWithCusip(
                    cusip=cusip,
                    msgs=msgs,
                )
            
        def feature_vector(msgs, cusips):
            'return (feature_vector, unused messages)'
            vp = verbose.make_verbose_print(True)
            result_unused_messages = msgs
            result_feature_vector = {
                'id_trigger_source': msgs[0].source,
                'id_trigger_identifier': msgs[0].identifier,
                'id_trigger_reclassified_trade_type': msgs[0].reclassified_trade_type,
                'id_target_oasspread': find_cusip(msgs, cusips[0]).oasspread,  # for most recent primary cusip
                }
            for i, cusip in enumerate(cusips):
                cf, unused_messages = cusip_features(msgs, cusip)
                vp('cusip_features result', i, cusip, len(cf), len(unused_messages))
                if i == 0:
                    result_feature_vector['id_primary_cusip'] = cusip
                else:
                    result_feature_vector['id_otr_%s_cusip' % i] = cusip
                for k, v in cf.items():
                    # adjust the keys to reflect whether the features are from the primary or OTR cusip
                    if k.startswith('id_'):
                        key = 'id_%s_%s' % ('primary' if i == 0 else 'otr_%s' % i, k[3:])
                    else:
                        assert not cusip.startswith('id_')
                        # NOTE: do not put the cusip in the feature names, becuase
                        # then the accuracy metrics will be very difficult to interpret
                        key = '%s_%s' % (
                            'primary' if i == 0 else 'otr_%s' % i,
                            k,
                            )
                    result_feature_vector[key] = v
                if len(unused_messages) < len(result_unused_messages):
                    result_unused_msgs = copy.copy(unused_messages)
            return result_feature_vector, result_unused_msgs
        
        def loop(msgs):
            'return (feature_vectors, unused messages)'
            result_feature_vectors = []
            result_unused = copy.copy(msgs)
            while len(result_feature_vectors) < n_feature_vectors:
                fv, unused = feature_vector(msgs, cusips)
                result_feature_vectors.append(fv)
                if len(unused) < len(result_unused):
                    result_unused = copy.copy(unused)
            return result_feature_vectors, result_unused

        pdb.set_trace()
        assert n_feature_vectors >= 0
        result, unused = loop(list(reversed(self._msgs)))
        print('todo: delete unused')
        return result

        
def test_tps_make_feature_vectors():
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
    
    pdb.set_trace()
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
    tps = TracePrintSequence(
        n_feature_vectors=1,
        )
    for tp_info in trace_prints:
        msg = make_trace_print_message(tp_info)
        tps.accumulate(msg)
    feature_vectors = tps.feature_vectors(
        cusips=('p', 'o1', 'o2'),
        n_feature_vectors=1,
        )
    for feature_vector in feature_vectors:
        print(feature_vector)
    assert len(feature_vectors) == 1

    
def unittest():
    'run unit tests'
    test_tps_make_feature_vectors()


class Identifier:
    def __init__(self):
        pdb.set_trace()
        self._last_datetime = datetime.datetime.now()
        self._sufix = 1

    def next(self):
        self.set_trace()
        current_dt = datetime.datetime.now()
        if current_dt == self._last_datetime:
            self._suffix += 1
        else:
            self._suffix = 1
        self._last_datetime = current_dt
        return '%s_%02d' % (self._last_datetime, self._suffix)

    
def do_work(config):
    def make_feature_vector_identifier():
        current_dt = datetime.datetime.now()
        
    def write_all(msg):
        'write msg to all output queues'
        # for now, just the one output file
        writer.write(
            routing_key='all_experts',
            message=msg,
            )
        
    pdb.set_trace()
    feature_vector_identifiers = Identifier()
    creating_output = False
    tps = TracePrintSequence()
    cusips = []  # [0] = primary, [1] = otr 1, [2] = otr 2, ...
    reader = queue.ReaderFile(config.get('in_messages'))
    writer = queue.WriterFile(config.get('out_messages'))
    write_all(message.SetVersion(
        what='machine_learning',
        version='1.0.0.0',
        ))
    while True:
        try:
            s = reader.__next__()
        except StopIteration:
            break
        msg = message.from_string(s)
        if isinstance(msg, message.BackToZero):
            pdb.set_trace()
            print('todo: implement BackToZero')
        elif isinstance(msg, message.SetCusipOtr):
            pdb.set_trace()
            while len(cusips) < msg.otr_level:
                cusips.append('')
            cusips[msg.otr_level] = msg.otr_cusip
        elif isinstance(msg, message.SetCusipPrimary):
            pdb.set_trace()
            if len(cusips) < 1:
                cusips.append('')
            cusips[0] = msg.primary_cusip
        elif isinstance(msg, message.SetVersion):
            pdb.set_trace()
            write_all(msg)
        elif isinstance(msg, message.TracePrint):
            pdb.set_trace()
            tps.accumulate(msg)
            if creating_output:
                pdb.set_trace()
                try:
                    feature_vectors = tps.features_vectors()
                except exception.MachineLearningException as e:
                    print('exception', e)
                    pdb.set_trace()  # for now, just print; later, send it downstream
                write_all(message.FeatureVectors(
                    source='events_cusip.py',
                    identifier=feature_vector_identifiers.next(),
                    datetime=datetime.datetime.now(),
                    feature_vectors=feature_vectors,
                    ))
            elif isinstance(msg, message.TracePrintCancel):
                pdb.set_trace()
                print('todo: implement TracePrintCancel')
            elif isinstance(msg, message.OutputStart):
                pdb.set_trace()
                creating_output = True
            elif isinstance(msg, message.OutputStop):
                pdb.set_trace()
                creating_output = False
            else:
                print(msg)
                print('unrecognized input message type')
                pdb.set_trace()
                
    print('have read all messages')
    pdb.set_trace()
    reader.close()
    writer.close()
                              
    
def main(argv):
    pdb.set_trace()
    program = 'events_cusip.py'
    config = configuration.make(
        program=program,
        argv=argv[1:],
        )
    print('sarted %s with configuration' % program)
    print(str(config))
    if config.get('debug', False):
        # enter pdb if run-time error
        import debug
        if False:
            debug.info
    unittest()
    do_work(config)

    
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
