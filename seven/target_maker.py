'''make targets from features

Example 3:
 Given these features
        oas    trade
 time   spread type quantity
 10:00     100 B    100
 10:01     101 D    200
 10:02     102 S    300

 Expect these targets
        next next next
 time    B   D   S
 10:00 nan 101 102
 10:01 nan nan 102

Example 4:
 Given these features
        oas    trade
 time   spread type quantity
 10:00     100 B    100
 10:01     101 D    200
 10:02     102 S    300
 10:03     103 B    400
 10:03     105 B    400
 10:03     105 D    600
 10:03     106 S    700

 Expect these targets
        next next next
 time    B   D   S
 10:00 104 101 102
 10:01 104 105 102
 10:02 104 105 106
'''
import collections
import datetime
import numpy as np
import pandas as pd
import pdb
import unittest


def make_training_targets(features, verbose=False, trace=False):
    'return DataFrame with next oasspread weighted-average prices'
    # NOTE: the index of ther return DataFrame cannot be the same as the features dataframe
    # because the returned DataFrame has one row for each effectivedatetime and the features dataframe
    # may have multiple rows on the same effectivedatetime

    def isnan(x):
        return x != x

    def make_weighted_average_features(features):
        'return Dict[effectivedatetime, Dict[trade_type, weighted_average_oasspread]]'
        result = collections.defaultdict(dict)
        for effectivedatetime in set(features['effectivedatetime']):
            at_effectivedatetime = features.loc[features['effectivedatetime'] == effectivedatetime]
            for trade_type in ('B', 'D', 'S'):
                for_trade_type = at_effectivedatetime.loc[at_effectivedatetime['trade_type'] == trade_type]
                if len(for_trade_type) == 0:
                    result[effectivedatetime][trade_type] = np.nan
                else:
                    weighted_sum = (for_trade_type['quantity'] * for_trade_type['oasspread']).sum()
                    weighted_average = weighted_sum / for_trade_type['quantity'].sum()
                    result[effectivedatetime][trade_type] = weighted_average
        return result

    if verbose:
        print 'make_training_targets: features'
        print features
    if trace:
        pdb.set_trace()
    weighted_average_features = make_weighted_average_features(features)
    effectivedatetimes = sorted(weighted_average_features.keys())
    d = collections.defaultdict(list)
    for i, effectivedatetime in enumerate(effectivedatetimes[:-1]):  # don't look at the last time
        # search forwad to find the next trade for each trade_type
        next_wa_oasspread = {}
        next_wa_effectivedatetime = {}
        for trade_type in ('B', 'D', 'S'):
            # search forward until we maybe find the next oasspread for the trade_type
            for k in xrange(i + 1, len(effectivedatetimes), 1):
                candidate_effectivedatetime = effectivedatetimes[k]
                wa = weighted_average_features[candidate_effectivedatetime][trade_type]
                if not isnan(wa):
                    next_wa_oasspread[trade_type] = wa
                    next_wa_effectivedatetime[trade_type] = candidate_effectivedatetime
                    break
        d['id_effectivedatetime'].append(effectivedatetime)
        d['target_next_oasspread_B'].append(next_wa_oasspread.get('B', np.nan))
        d['target_next_oasspread_D'].append(next_wa_oasspread.get('D', np.nan))
        d['target_next_oasspread_S'].append(next_wa_oasspread.get('S', np.nan))
        d['info_target_next_effectivedatetime_B'].append(next_wa_effectivedatetime.get('B', np.nan))
        d['info_target_next_effectivedatetime_D'].append(next_wa_effectivedatetime.get('D', np.nan))
        d['info_target_next_effectivedatetime_S'].append(next_wa_effectivedatetime.get('S', np.nan))
    if trace:
        pdb.set_trace()
    if len(d) == 0:
        # we created no records
        result = pd.DataFrame(
            columns=[
                'id_effectivedatetime',
                'target_next_oasspread_B',
                'target_next_oasspread_D',
                'target_next_oasspread_S',
                'info_target_next_effectivedatetime_B',
                'info_target_next_effectivedatetime_D',
                'info_target_next_effectivedatetime_S',
            ],
        )
        return result
    else:
        result = pd.DataFrame(data=d)
        if verbose:
            print 'make_training_targets: result'
        return result


class TestAllTargets(unittest.TestCase):
    def test_multiple_trades_at_same_time(self):
        def isnan(x):
            return x != x

        def make_effectivedatetime(time):
            'convert hours.minutes to datetime.datetime'
            hours = int(time)
            minutes = int(round(100 * (time - hours)))
            return datetime.datetime(2016, 11, 1, hours, minutes)

        def assert_equal(expected, actual):
            # print 'assert_equal:', expected, actual
            self.assertEqual(len(expected), len(actual))
            for i in xrange(len(expected)):
                # print expected[i], actual[i]
                if isnan(expected[i]):
                    self.assertTrue(isnan(actual[i]))
                else:
                    self.assertEqual(expected[i], actual[i])

        def make_expected(rows):
            'return DataFrame'
            result = pd.DataFrame(
                columns=[
                    'id_effectivedatetime',
                    'target_next_oasspread_B',
                    'target_next_oasspread_D',
                    'target_next_oasspread_S',
                ],
            )
            for row in rows:
                test_time, b, d, s = row
                result.loc[len(result)] = (make_effectivedatetime(test_time), b, d, s)
            return result

        def make_features(highest_feature_time, features_data):
            'return DataFrame'
            result = pd.DataFrame(
                columns=['effectivedatetime', 'oasspread', 'trade_type', 'quantity'],
            )
            for trace_time, trace_prints in features_data:
                for trace_print in trace_prints:
                    oasspread, trade_type, quantity = trace_print
                    result.loc[len(result)] = (make_effectivedatetime(trace_time), oasspread, trade_type, quantity)
                if trace_time == highest_feature_time:
                    break
            return result

        verbose = False
        features_data = (
            (10.00, [(100, 'B', 100)]),
            (10.01, [(101, 'D', 200)]),
            (10.02, [(102, 'S', 300)]),
            (10.03, [(103, 'B', 400), (105, 'B', 400), (105, 'D', 600), (106, 'S', 700)]),
            (10.04, [(107, 'B', 800)]),
        )
        nan = np.nan
        tests = (  # (highest feature time, [rows])
            (10.00, []),
            (10.01, [(10.00, nan, 101, nan)]),
            (10.02, [(10.00, nan, 101, 102),
                     (10.01, nan, nan, 102)]),
            (10.03, [(10.00, 104, 101, 102),
                     (10.01, 104, 105, 102),
                     (10.02, 104, 105, 106)]),
            (10.04, [(10.00, 104, 101, 102),
                     (10.01, 104, 105, 102),
                     (10.02, 104, 105, 106),
                     (10.03, 107, nan, nan)]),
        )
        for test in tests:
            highest_feature_time, rows = test
            expected = make_expected(rows)
            features = make_features(highest_feature_time, features_data)
            actual = make_training_targets(features)
            if verbose:
                print 'actual'
                print actual
                print 'expected'
                print expected
                pdb.set_trace()
            for column_name in expected.columns:
                # NOTE: ignore the info columns in this test
                assert_equal(expected[column_name], actual[column_name])


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
