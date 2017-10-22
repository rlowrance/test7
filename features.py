import datetime
import typing

import exception
import machine_learning
import shared_message


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
            # value must be a float
            if isinstance(item, float):
                super(FeatureVector, self).__setitem__(key, item)
            else:
                raise exception.FeatureVector('item %s has type %s, not a float value (key=%s)' % (
                    item,
                    type(item),
                    key,
                ))


def trace_print(msgs: typing.List[shared_message.Message], cusip: str, debug=False) -> FeatureVector:
    'return (Features from msgs, unused messages) or raise NoFeatures'
    # create features from a trace print and the prior trace print
    # return empty feature if not able to create features
    # the features are formed from two trace prints
    # The caller modifies the feature vector keys to include the name of this functions
    # in those keys, so that the keys will be unique across all features. So DO NOT
    # include the name of this function in the keys of the feature vector.
    def find_messages(msgs, cusip, reclassified_trade_type) -> typing.List[shared_message.Message]:
        'attempt to find first 2 messages with specified attributes'
        'return list of first 2 messages with the cusip and reclassified trade type and unused messages'
        result = []
        for i, msg in enumerate(msgs):
            if msg.cusip == cusip and msg.reclassified_trade_type == reclassified_trade_type:
                result.append(msg)
                if len(result) == 2:
                    return result, msgs[i + 1:]
        raise exception.NoFeatures('features.trace_print: not 2 %s %s messages' % (cusip, reclassified_trade_type))

    def add_features(result: FeatureVector, rtt: str, msgs: typing.List[shared_message.Message]):
        # mutate result by adding features from 2 trace print messages'
        assert len(msgs) == 2
        msg0 = msgs[0]  # most recent message
        msg1 = msgs[1]  # message just before the most recent message
        result['id_%s_msg0_issuepriceid' % rtt] = msg0.issuepriceid
        result['id_%s_msg1_issuepriceid' % rtt] = msg1.issuepriceid
        result['%s_oasspread' % rtt] = msg0.oasspread
        result['%s_oasspread_less_prior' % rtt] = msg0.oasspread - msg1.oasspread
        result['%s_oasspread_divided_by_prior' % rtt] = (
            100.0 if msg1.oasspread == 0.0 else msg0.oasspread / msg1.oasspread
            )

    set_trace = machine_learning.make_set_trace(debug)
    vp = machine_learning.make_verbose_print(debug)
    vpp = machine_learning.make_verbose_pp(debug)
    set_trace()
    B_messages, B_unused = find_messages(msgs, cusip, 'B')
    S_messages, S_unused = find_messages(msgs, cusip, 'S')
    result = FeatureVector()
    add_features(result, 'B', B_messages)
    add_features(result, 'S', S_messages)
    vp('features_B_and_S')
    vpp(result)
    set_trace()
    return result, B_unused if len(B_unused) < len(S_unused) else S_unused
                

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
    
    def make_messages(*tests):
        def make_message(test):
            cusip, info, rtt = test
            return shared_message.TracePrint(
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

    def make_messages_1():
        return make_messages(
            ('a', 1, 'B'),
            ('a', 2, 'S'),
            ('a', 3, 'B'),
            ('a', 4, 'S'),
            ('b', 5, 'B'),
        )

    def test_1a():
        msgs = make_messages_1()
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

    def test_1b():
        msgs = make_messages_1()
        set_trace()
        try:
            r = trace_print(msgs, 'b')
            assert False, 'should have raised'
        except exception.NoFeatures as e:
            vp(e)
            # expect to be here

    def make_messages_2():
        return make_messages(
            ('o2', 12, 'S'),
            ('p', 11, 'B'),
            ('o2', 10, 'B'),
            ('p', 9, 'S'),
            ('o1', 8, 'B'),
            ('o1', 7, 'S'),
            ('o2', 6, 'B'),
            ('p', 5, 'S'),
            ('o1', 4, 'B'),
            ('p', 3, 'S'),
            ('o2', 2, 'S'),
            ('o1', 1, 'S'),
            ('p', 0, 'B'),
        )
    
    def len_features(fv):
        result = 0
        for k, v in fv.items():
            if k.startswith('id_'):
                pass
            else:
                result += 1
        return result
        
    def test_2a():
        debug = False
        vp = machine_learning.make_verbose_print(debug)
        vpp = machine_learning.make_verbose_pp(debug)
        set_trace = machine_learning.make_set_trace(debug)

        set_trace()
        msgs = make_messages_2()
        fv, unused = trace_print(msgs, 'p', debug=False)
        assert len(unused) == 0
        assert len_features(fv) == 6

        fv, unused = trace_print(msgs, 'o1')
        assert len(unused) == 1
        assert len_features(fv) == 6

        fv, unused = trace_print(msgs, 'o2')
        assert len(unused) == 2
        assert len_features(fv) == 6

    def test_2b():
        debug = False
        vp = machine_learning.make_verbose_print(debug)
        vpp = machine_learning.make_verbose_pp(debug)
        set_trace = machine_learning.make_set_trace(debug)

        set_trace()
        msgs = make_messages_2()[1:]  # start at send message

        fv, unused = trace_print(msgs, 'p')
        assert len(unused) == 0
        assert len_features(fv) == 6

        fv, unused = trace_print(msgs, 'o1')
        assert len(unused) == 1
        assert len_features(fv) == 6

        try:
            fv, unused = trace_print(msgs, 'o2')
            assert False, 'should have raised exception'
        except exception.NoFeatures as e:
            vp('expected exception', e)
    test_1a()
    test_1b()
    test_2a()
    test_2b()

    
def unittests():
    test_FeatureVector()
    test_trace_print()
    

if __name__ == '__main__':
    unittests()
