'''read queue events.{cusip}, write feature set to queue features.{cusip}.{expert}'''
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
        self._msgs.append(msg)
        # for now, save everything; later, discard msg that we will never use

    def feature_vectors(self,
                        cusips: typing.List[str],  # primary, otr1, otr2, ...
                        n_feature_vectors: int,
                        trace=False,
                        ):
        'return List[feature_vector] and truncate self._msgs, OR raise exception'
        set_trace = machine_learning.make_set_trace(trace)
        
        def find_cusip(msgs, cusip):
            'return first message with the specified cusip'
            if False and trace:
                pdb.set_trace()
            for msg in msgs:
                if msg.cusip == cusip:
                    return msg
            else:
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
            raise exception.NoMessageWithCusipAndRtt(
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
            
        def feature_vector(msgs, cusips):
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
            result_feature_vector = {
                'id_trigger_source': msgs[0].source,
                'id_trigger_identifier': msgs[0].identifier,
                'id_trigger_reclassified_trade_type': msgs[0].reclassified_trade_type,
                'id_target_oasspread': find_cusip(msgs, cusips[0]).oasspread,  # for most recent primary cusip
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
            set_trace()
            result_feature_vectors = []
            result_unused = msgs
            for i in range(0, n_feature_vectors, 1):
                msgs_to_be_used = msgs[i:]
                fv, unused = feature_vector(msgs_to_be_used, cusips)
                print('loop: fv trigger identifier: %s len(msgs): %d, len(unused): %d' % (
                    fv['id_trigger_identifier'],
                    len(msgs_to_be_used),
                    len(unused),
                    ))
                result_feature_vectors.append(fv)
                if len(unused) < len(result_unused):
                    result_unused = copy.copy(unused)
            set_trace()
            return list(reversed(result_feature_vectors)), result_unused

        set_trace()
        assert n_feature_vectors >= 0
        result, unused = loop(list(reversed(self._msgs)))
        set_trace()
        print('todo: delete unused')
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
    # test 3 OTRs and consumption of entire set of messages
    vp = machine_learning.make_verbose_print(False)
    set_trace = machine_learning.make_set_trace(False)
    
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
    set_trace()
    feature_vectors = tps.feature_vectors(
        cusips=('p', 'o1', 'o2'),
        n_feature_vectors=1,
        trace=False,
        )
    assert len(tps._msgs) == 13  # this test uses all of the messages
    for feature_vector in feature_vectors:
        vp(feature_vector)
    assert len(feature_vectors) == 1

    
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

    
def do_work(config):
    def write_all(msg):
        'write msg to all output queues'
        # for now, just the one output file
        writer.write(
            routing_key='all_experts',
            message=msg,
            )

    set_trace = machine_learning.make_set_trace(True)
    set_trace()
    feature_vector_identifiers = Identifier()
    creating_output = False
    tps = TracePrintSequence()
    cusips = []  # [0] = primary, [1] = otr 1, [2] = otr 2, ...
    reader = queue.ReaderFile(config.get('in_messages'))
    writer = queue.WriterFile(config.get('out_messages'))
    write_all(message.SetVersion(
        source='events_cusips.py',
        identifier=str(datetime.datetime.now()),
        what='machine_learning',
        version='1.0.0.0',
        ))
    while True:
        try:
            s = reader.__next__()
        except StopIteration:
            break
        msg = message.from_string(s)
        print('\n%r' % msg)
        if isinstance(msg, message.BackToZero):
            pdb.set_trace()
            print('todo: implement BackToZero')
        elif isinstance(msg, message.SetCusipOtr):
            while len(cusips) <= msg.otr_level:
                cusips.append('')
            cusips[msg.otr_level] = msg.cusip
        elif isinstance(msg, message.SetCusipPrimary):
            if len(cusips) < 1:
                cusips.append('')
            cusips[0] = msg.cusip
        elif isinstance(msg, message.SetVersion):
            write_all(msg)
        elif isinstance(msg, message.TracePrint):
            tps.accumulate(msg)
            if creating_output:
                # verify that the message stream has consistently set all cusips
                # NOTE: to keep space used bounded, create feature vectors periodically
                # including after every TracePrint message is read
                # NOTE; if creating feature vectors, the input message queue must have declared
                # the primary and OTR cusips, as these are needed to create feature vectors
                assert len(cusips) >= 2, 'messages did not declare any OTR cusips; cusips=%s' % cusips
                for i, cusip in enumerate(cusips):
                    if i > 0:
                        assert cusip != ' ', 'messages did not declare OTR cusip for level %d; cusips=%s' % (
                            i + 1,
                            cusips,
                        )
                try:
                    set_trace()
                    feature_vectors = tps.feature_vectors(
                        cusips=cusips,
                        n_feature_vectors=20,
                        trace=True,
                    )
                except exception.MachineLearningException as e:
                    print('exception whle creating feature vectors:', e)
                    pdb.set_trace()  # for now, just print; later, send some exceptions downstream
                write_all(message.FeatureVectors(
                    source='events_cusip.py',
                    identifier=feature_vector_identifiers.get_next(),
                    datetime=datetime.datetime.now(),
                    feature_vectors=feature_vectors,
                ))
        elif isinstance(msg, message.TracePrintCancel):
            pdb.set_trace()
            print('todo: implement TracePrintCancel')
        elif isinstance(msg, message.OutputStart):
            creating_output = True
        elif isinstance(msg, message.OutputStop):
            pdb.set_trace()
            creating_output = False
        else:
            print(msg)
            print('%r' % msg)
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
