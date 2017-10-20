import datetime
import pdb
import typing

import exception
import machine_learning
import message


class FeatureVector(dict):
    'a FeatureVector is a dict with restrictions on the values of certain items'
    # Idea;
    # - keys of the form 'id_*' are identiers that are not really features
    # - all if the other keys are really features, and they must have a floating point value
    # ref: stack overflow at "how to perfectly override a dict"
    def __init__(self, *args, **kwargs):
        super(FeatureVector, self).__init__(*args, **kwargs)

    def __setitem__(self, key, item):
        if key.startswith('id_'):
            # value can be anything
            super(FeatureVector, self).__setitem__(key, item)
        else:
            if isinstance(item, float):
                super(FeatureVector, self).__setitem__(key, item)
            else:
                raise exception.FeatureVector('item %s has type %s, not a float value (key=%s)' % (
                    item,
                    type(item),
                    key,
                ))


def trace_print(msgs: typing.List[message.Message], cusip: str) -> FeatureVector:
    'return (Features from msgs, unused messages) or raise NoFeatures'
    # create features from a trace print and the prior trace print
    # return empty feature if not able to create features
    # the features are formed from two trace prints
    def find_messages(msgs, cusip, reclassified_trade_type):
        'return list of first 2 messages with the cusip and reclassified trade type and unused messages'
        result = []
        for i, msg in enumerate(msgs):
            if msg.cusip == cusip and msg.reclassified_trade_type == reclassified_trade_type:
                result.append(msg)
                if len(result) == 2:
                    return result, msgs[i + 1:]
        return result, msgs[i:]

    def add_features(result, trade_type, msgs):
        assert len(msgs) == 2
        msg0 = msgs[0]  # most recent message
        msg1 = msgs[1]  # message just before the most recent message
        result['id_trace_print_%s_msg0_issuepriceid' % trade_type] = msg0.issuepriceid
        result['id_trace_print_%s_msg1_issuepriceid' % trade_type] = msg1.issuepriceid
        result['trace_print_%s_oasspread' % trade_type] = msg0.oasspread
        result['trace_print_%s_oasspread_less_prior' % trade_type] = msg0.oasspread - msg1.oasspread
        result['trace_print_%s_oasspread_divided_by_prior' % trade_type] = (
            100.0 if msg1.oasspread == 0.0 else msg0.oasspread / msg1.oasspread
            )
        return result

    set_trace = machine_learning.make_set_trace(False)
    vp = machine_learning.make_verbose_print(False)
    set_trace()
    B_messages, B_unused = find_messages(msgs, cusip, 'B')
    if len(B_messages) != 2:
        raise exception.NoFeatures('not 2 B messages')
    S_messages, S_unused = find_messages(msgs, cusip, 'S')
    if len(S_messages) != 2:
        raise exception.NoFeature('not 2 S messages')
    features_B = add_features(FeatureVector(), 'B', B_messages)
    vp(features_B)
    features_B_and_S = add_features(features_B, 'S', S_messages)
    vp('features_B_and_S', features_B_and_S)
    set_trace()
    return features_B_and_S, B_unused if len(B_unused) < len(S_unused) else S_unused
                

def test_FeatureVector():
    set_trace = machine_learning.make_set_trace(False)
    vp = machine_learning.make_verbose_print(False)
    
    def test1():
        set_trace()
        ok = FeatureVector()
        ok['id_trace_print'] = 'abc'
        ok['a'] = 10.0

    def test2():
        set_trace()
        bad = FeatureVector()
        try:
            bad['a'] = 1  # must be a float, but is not
            assert False, 'should have raised an exception'
        except exception.FeatureVector as e:
            vp('exception', e)
        except Exception as e:
            print('raised unexpected exception', e)
            assert False, 'should have rased exception.FeatureVector'
    
    test1()
    test2()


def test_trace_print():
    set_trace = machine_learning.make_set_trace(False)
    vp = machine_learning.make_verbose_print(False)
    
    def make_messages(tests):
        def make_message(test):
            cusip, info, rtt = test
            return message.TracePrint(
                source='trace_print_test',
                identifier=str(info),
                cusip=cusip,
                issuepriceid=str(info),
                datetime=datetime.datetime.now(),
                oasspread=float(info),
                trade_type=rtt,
                reclassified_trade_type=rtt,
                cancellation_probability=0.0,
                )

        msgs = []
        for test in tests:
            msgs.append(make_message(test))
        return msgs

    def test_ok():
        msgs = make_messages((
            ('a', 1, 'B'),
            ('a', 2, 'S'),
            ('a', 3, 'B'),
            ('a', 4, 'S'),
            ('b', 5, 'B'),
            ))
        set_trace()
        r = trace_print(msgs, 'a')
        vp('test_ok', r)
        set_trace()
        try:
            r = trace_print(msgs, 'b')
            assert False, 'should raise an exception'
        except exception.NoFeatures as e:
            vp('raised', e)
            set_trace()

    def test_bad():
        msgs = make_messages((
            ('a', 1, 'B'),
            ('a', 2, 'S'),
            ('a', 3, 'B'),
            ('a', 4, 'S'),
            ('b', 5, 'B'),
            ))
        set_trace()
        try:
            r = trace_print(msgs, 'b')
            assert False, 'should have raised'
        except exception.NoFeatures as e:
            vp(e)
            # expect to be here

    test_ok()
    test_bad()
    
    
def unittests():
    test_FeatureVector()
    test_trace_print()
    

if __name__ == '__main__':
    unittests()
