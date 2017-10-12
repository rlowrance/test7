'''read queue events.{cusip}, write feature set to queue features.{cusip}.{expert}'''
import collections
import pdb
from pprint import pprint as pp
import sys

import configuration
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


class FeaturesTargets:
    def __init__(self, max_n_otrs):
        self._max_n_otrs = max_n_otrs
        
        self._otr_cusips = {}
        self._primary_cusip = None

    def set_cusip_otr(self, msg):
        if self._primary_cusip is not None:
            assert self._primary_cusip == msg.primary_cusip  # attempt to set OTR for other than primary
        self._otr_cusips[msg.otr_cusip] = {
            'otr_level': msg.otr_level,
            'primary_cusip': msg.primary_cusip,
        }
        assert len(self._otr_cusips) <= self._max_n_otrs
        
    def set_cusip_primary(self, msg):
        if self._primary_cusip is not None:
            assert self._primary_cusip == msg.primary_cusip  # attempt to change primary cusip
        self._primary_cusip = msg.primary_cusip

    def trace_print(self, msg):
        pdb.set_trace()

    def has_feature_vector_set(self):
        print('stub: has_feature_vector_set')
        return False
    

class TracePrintSequence:
    def __init__(self):
        print('stub: TracePrintSequence.__init__')

    def accumulate(self, msg):
        assert isinstance(msg, message.TracePrint)
        print('stub: TracePrintSequence.accumulate')


def do_work(config):
    def write_all(msg):
        'write msg to all routing keys'
        for routing_key in routing_keys:
            writer.write(
                routing_key=routing_key,
                message=msg,
                )

    reader = queue.ReaderFile(config.get('in_messages'))
    assert config.get('output_to') == 'one'  # write the features just once (for debugging)
    routing_keys = ('features.n',)  # naive model  TODO: replace with keys for all experts
    writer = queue.WriterFile(config.get('out_messages'))
    write_all(message.SetVersion(
        what='features',
        version='1.2.0.0',  # feature set 1 (as for test_train) implementation 2 (rabbit MQ)
    ))
    ft = FeaturesTargets(
        max_n_otrs=1,  # max n of OTR cusips (1 for feature set 1)
    )
    counter = collections.Counter()
    pdb.set_trace()
    primary_cusip = ''
    tps_primary = None
    tps_otrs = {}  # Dict[otr_level, (otr_cusip, TracePrintSequence)]
    produce_output = False
    input_message_counter = 0
    trace_print_messages = []
    while True:
        try:
            s = next(reader)
            input_message_counter += 1
        except StopIteration:
            break
        # handle the message
        print(input_message_counter, s)
        if len(s) == 0:
            counter['empty message'] += 1
            continue
        msg = message.from_string(s)
        print('\nnext message', msg)
        counter['message_type %s' % msg.message_type] += 1
        if msg.message_type == 'SetCusipOtr':
            # all the OTR cusips must be for the most recent primary cusip
            assert msg.primary_cusip == primary_cusip
            # a otr cusip can have multiple otr levels
            if msg.otr_level in tps_otrs and msg.otr_cusip == tps_otrs[msg.otr_level][0]:
                pass
            else:
                assert len(tps_otrs) <= config.get('n_otr_levels')
                tps_otrs[msg.otr_level] = (msg.otr_cusip, TracePrintSequence())
        elif msg.message_type == 'SetCusipPrimary':
            # possible the primary cusip replaces the existing primary cusip
            if msg.primary_cusip == primary_cusip:
                pass
            else:
                primary_cusip = msg.primary_cusip
                tps_primary = TracePrintSequence()
        elif msg.message_type == 'SetVersion':
            write_all(msg)  # pass along the version from upstream
        elif msg.message_type == 'TracePrint':
            # a cusip can be both primary and OTR
            trace_print_messages.append(msg)
            if msg.cusip == primary_cusip:
                tps_primary.accumulate(msg)
                counter['TracePrint primary accumulated'] += 1
            for tps_otr_key, (tps_cusip, tps_trace_print_sequence) in tps_otrs.items():
                if tps_cusip == msg.cusip:
                    tps_trace_print_sequence.accumulate(msg)
                    counter['TracePrint otr %s accumulated' % tps_cusip] += 1
            if produce_output:
                counter['considered producing output'] += 1
                print('stub: sometimes create and write feature vectors')
                # make sure we have trace prints from all the OTR cusips
                pdb.set_trace()
        elif msg.message_type == 'OutputStart':
            pdb.set_trace()
            produce_output = True
        elif msg.message_type == 'OutputStop':
            pdb.set_trace()
            produce_output = False
        else:
            print('\n*** unexpected input message type %s' % msg.message_type)
            print('unexpected message:', msg)
            pdb.set_trace()
    print('\nFinished processing the input queue')
    print('counters')
    for k in sorted(counter.keys()):
        print('%-40s: %5d' % (k, counter[k]))
    print('\nall trace print messages')
    for trace_print_message in trace_print_messages:
        print(trace_print_message)
    writer.close()


def main(argv):
    config = configuration.make(
        program='events_cusip.py',
        argv=argv[1:],  # ignore program name
    )
    print('started with configuration')
    print(str(config))
    if config.get('debug', False):
        # enter pdb if run-time error
        # (useful during development)
        import debug
        if False:
            debug.info
    do_work(config)


if __name__ == '__main__':
    if False:
        pdb
        pp
    main(sys.argv)
