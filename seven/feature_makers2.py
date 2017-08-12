'''class to create features from the input files

Rules for FeatureMakers
1. Each FeatureMaker is a class associated with one input file
2. Calling its update method with an input record returns a dictionary of features. The dictionary
   has as keys strings, which are the feature names, and floating point values, the values
   of the features.
3. The feature names from each feature maker must be unique across all feature makers.
4. Any string can be a feature name. However, features names ending in "_size" are transformed
   if the model_spec.transform_x value is True.
5. If a feature name begins with "id_", its not actually a feature name that is put into the
   models. Its just an identifier.
6. If a feature name begins with "p_", its a feature for the print (the trade).
7. If a feature name begins with "otr1", its a feature name for the closest on-the-run bond.

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a license.
'''
from __future__ import division

import abc
import collections
import copy
import datetime
import math
import numbers
import pandas as pd
import pdb
from pprint import pprint
import unittest

# imports from seven/
import input_event
import OrderImbalance4
import read_csv

pp = pprint


def make_effectivedatetime(df, effectivedate_column='effectivedate', effectivetime_column='effectivetime'):
    '''create new column that combines the effectivedate and effective time

    example:
    df['effectivedatetime'] = make_effectivedatetime(df)
    '''
    values = []
    for the_date, the_time in zip(df[effectivedate_column], df[effectivetime_column]):
        values.append(datetime.datetime(
            the_date.year,
            the_date.month,
            the_date.day,
            the_time.hour,
            the_time.minute,
            the_time.second,
        ))
    return pd.Series(values, index=df.index)


class FeatureMaker(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, args, event, name):
        self._name = name
        print 'constructing FeatureMaker %s for %s %s' % (name, args.issuer, args.cusip)


class Etf(FeatureMaker):
    def __init__(self, df=None, name=None):
        'construct'
        # name is "weight {security_kind} {etf_name}"
        assert df is not None
        assert name is not None
        super(Etf, self).__init__('etf ' + name)  # sets self.name
        table = collections.defaultdict(dict)
        for index, row in df.iterrows():
            timestamp, cusip = index
            date = timestamp.date()
            weight = row['weight_of_cusip_pct']
            assert 0 <= weight <= 100.0
            table[date][cusip] = weight
        df.table = table
        self.security_kind, self.security_id, self.etf_name = name.split(' ')[1:]
        self.second_index_name = None  # overridden by subclass

    def make_features(self, trace_index, trace_record):
        'return (None, err) or (DataFrame, None)'
        date = trace_record['effectivedate']
        second_index_value = trace_record[self.second_index_name]
        pdb.set_trace()
        if date not in self.table:
            return None, 'date %d not in input file for %s' % (date, self.name)
        table_date = self.table[date]
        if second_index_value not in table_date:
            return None, 'security id %s not in input file %s for date %s' % (second_index_value, self.name, date)
        weight = table_date[second_index_value]
        features = {
            'p_weight_etf_pct_%s_%s_size' % (self.security_kind, self.etf_name): weight,
        }
        return features, None


class EtfCusip(Etf):
    def __init__(self, df=None, name=None):
        print 'construction FeatureMakerEtfCusip'
        super(EtfCusip, self). __init__(df=df, name=name)
        self.second_index_name = 'cusip'


class EtfCusipTest(unittest.TestCase):
    def test(self):
        return  # for now, unstub when testing the EFT feature makers
        Test = collections.namedtuple('Test', 'logical_name cusip date, expected_featurename, expected_weight')
        tests = (
            Test('weight cusip agg', '00184AAG0', datetime.date(2010, 1, 29), 'p_weight_etf_pct_cusip_agg_size', 0.152722039314371),
        )
        for test in tests:
            fm = EtfCusip(
                df=read_csv.input(issuer=None, logical_name=test.logical_name),
                name=test.logical_name,
            )
            trace_record = {
                'effectivedate': test.date,
                'cusip': test.cusip
            }
            features, err = fm.make_features(0, trace_record)
            self.assertIsNone(err)
            self.assertEqual(1, len(features))
            for k, v in features.iteritems():
                self.assertEqual(test.expected_featurename, k)
                self.assertAlmostEqual(test.expected_weight, v)


class FundamentalsOLD(FeatureMaker):
    def __init__(self, issuer):
        super(Fundamentals, self).__init__('Fundamentals(issuer=%s)' % issuer)
        self.issuer = issuer
        self.file_logical_names = (  # feature names are the same as the logical names
            'expected_interest_coverage',
            'gross_leverage',
            'LTM_EBITDA',
            'mkt_cap',
            'mkt_gross_leverage',
            'reported_interest_coverage',
            'total_assets',
            'total_debt',
        )
        # create Dict[content_name: str, Dict[datetime.date, content_value:float]]
        self.data = self._read_files()
        return

    def _read_file(self, logical_name):
        'return the contents of the CSV file as a pd.DataFrame'
        df = read_csv.input(issuer=self.issuer, logical_name=logical_name)
        if len(df) == 0:
            print 'df has zero length', logical_name
            pdb.set_trace()
        result = {}
        for timestamp, row in df.iterrows():
            result[timestamp.date()] = row[0]
        return result

    def _read_files(self):
        'return Dict[datetime.date, Dict[content_name:str, content_value:float]]'
        result = {}
        for logical_name in self.file_logical_names:
            result[logical_name] = self._read_file(logical_name)
        return result

    def _get(self, date, logical_name):
            'return (value on or just before the specified date, None) or (None, err)'
            # a logical name and feature name are the same thing for this function
            data = self.data[logical_name]
            for sorted_date in sorted(data.keys(), reverse=True):
                if date >= sorted_date:
                    result = data[sorted_date]
                    return (result, None)
            return (
                None,
                'date %s not in fundamentals for issuer %s content %s' % (
                    date,
                    self.issuer,
                    logical_name))

    def make_features(self, trace_index, trace_record, extra):
        'return Dict[feature_name, feature_value], errs'
        def check_no_negatives(d):
            for k, v in d.iteritems():
                assert v >= 0.0

        date = trace_record['effectivedate'].date()

        # build basic features directly from the fundamentals file data
        result = {}
        errors = []
        for feature_name in self.file_logical_names:
            value, err = self._get(date, feature_name)
            if err is not None:
                errors.append(err)
            else:
                result['%s_size' % feature_name] = value

        # add in derived features, which for now are ratios of the basic features
        # all of the basic features are in the result dict

        def maybe_add_ratio(feature_name, numerator_name, denominator_name):
            numerator = result[numerator_name]
            denominator = result[denominator_name]
            if denominator is 0:
                errors.append('feature %s had zero denominator' % feature_name)
            value = numerator * 1.0 / denominator
            if value != value:
                errors.append('feature %s was NaN' % feature_name)
            else:
                result[feature_name] = value

        maybe_add_ratio('debt_to_market_cap_size', 'total_debt_size', 'mkt_cap_size')
        maybe_add_ratio('debt_to_ltm_ebitda_size', 'total_debt_size', 'LTM_EBITDA_size')
        if len(errors) is not None:
            return (None, errors)

        check_no_negatives(result)
        if len(errors) == 0:
            return (result, None)
        else:
            return (None, errors)


class HistoryOasspread(FeatureMaker):
    'create historic oasspread features'

    # The caller will want to create these additional features:
    #  p_reclassified_trade_type_is_{B|C} with value 0 or 1
    #  p_oasspread with the value in the trace_record
    def __init__(self, history_length):
        super(HistoryOasspread, self).__init__('OasspreadHistory(history_length=%s)' % history_length)
        self.history_length = history_length  # number of historic B and S oasspreads in the feature vector
        self.recognized_trade_types = ('B', 'S')

        self.history = {}
        for trade_type in self.recognized_trade_types:
            self.history[trade_type] = collections.deque(maxlen=history_length)

    def make_features(self, trace_index, trace_record, extra, verbose=False):
        'return (features, err)'
        def accumulate_history():
            oasspread = trace_record['oasspread']
            if isinstance(oasspread, numbers.Number):
                self.history[reclassified_trade_type].append(oasspread)
            else:
                return (None, 'oasspread is %s, which is not a number' % oasspread)

        reclassified_trade_type = extra.get('id_reclassified_trade_type', None)
        if reclassified_trade_type is None:
            return (None, 'no reclassified trade type')
        if reclassified_trade_type not in self.recognized_trade_types:
            return (None, 'no history is created for reclassified trade types %s' % reclassified_trade_type)

        # determine whether we have enough history to build the features
        for trade_type in self.recognized_trade_types:
            if len(self.history[trade_type]) < self.history_length:
                err = 'history for trade type %s has length less than %s' % (
                    trade_type,
                    self.history_length,
                )
                accumulate_history()
                return (None, err)

        # if the history has a zero value, we cannot create ratios
        for trade_type in self.recognized_trade_types:
            for k in range(self.history_length):
                if self.history[trade_type][k] == 0.0:
                    err = 'history of oasspread contains a zero value'
                    accumulate_history()
                    return (None, err)

        # we have enough history and no zero values (which would mess up the ratios)

        # create the oasspread history features
        features = {}
        for trade_type in self.recognized_trade_types:
            for k in range(self.history_length):
                index2 = self.history_length - k  # user-visible index
                key = 'oasspread_%s_back_%02d' % (trade_type, index2)
                features[key] = self.history[trade_type][k]

        # use the back_N features to create the ratio features
        if False and len(self.history['B']) == 3:
            print self.history
            pp(features)
            pdb.set_trace()
        for trade_type in self.recognized_trade_types:
            for k in range(self.history_length - 1):
                index2 = self.history_length - k
                index1 = index2 - 1
                key = 'oasspread_ratio_%s_back_%02d_to_%02d' % (trade_type, index1, index2)
                index1_key = 'oasspread_%s_back_%02d' % (trade_type, index1)
                index2_key = 'oasspread_%s_back_%02d' % (trade_type, index2)
                if verbose:
                    print key
                    print index1_key, features[index1_key]
                    print index2_key, features[index2_key]
                features[key] = features[index1_key] / features[index2_key]

        # ratio of current spread to last spread of same reclassified trade type
        features['id_oasspread_ratio'] = (
            trace_record['oasspread'] / self.history['B'][-1] if extra['id_reclassified_trade_type'] == 'B' else
            trace_record['oasspread'] / self.history['S'][-1]
        )

        accumulate_history()
        return (features, None)


class HistoryOasspreadTest(unittest.TestCase):
    def _has_none(self, items):
        for item in items:
            if item is None:
                return True
        return False

    def test_k_2_without_ratios(self):
        verbose = False

        def make_trace_record(trace_index, test):
            'return a pandas.Series with the bare minimum fields set'
            return pd.Series(
                data={
                    'oasspread': test.oasspread,
                },
            )
        Test = collections.namedtuple(
            'Test',
            'reclassified_trade_type oasspread b02 b01 s02 s01',
        )
        tests = (  # k = 2
            Test('B', 100, None, None, None, None),
            Test('S', 103, 100, None, None, None),
            Test('S', 104, 100, None, 103, None),
            Test('B', 101, 100, None, 103, 104),
            Test('B', 102, 100, 101, 103, 104),
            Test('S', 105, 101, 102, 103, 104),
            Test('S', 106, 101, 102, 104, 105),
            Test('B', 103, 101, 102, 105, 106),
            Test('B', 105, 102, 103, 105, 106),
        )
        feature_maker = HistoryOasspread(history_length=2)
        for trace_index, test in enumerate(tests):
            if verbose:
                print 'TestOasSpreads.test_1: trace_index', trace_index
                print 'TestOasSpreads.test_1: test', test
            features, err = feature_maker.make_features(
                trace_index,
                make_trace_record(trace_index, test),
                {'id_reclassified_trade_type': test.reclassified_trade_type},
            )
            if self._has_none(test):
                self.assertTrue(features is None)
                self.assertTrue(err is not None)
            else:
                self.assertTrue(features is not None)
                self.assertTrue(err is None)
                self.assertEqual(features['oasspread_B_back_01'], test.b01)
                self.assertEqual(features['oasspread_B_back_02'], test.b02)
                self.assertEqual(features['oasspread_S_back_01'], test.s01)
                self.assertEqual(features['oasspread_S_back_02'], test.s02)

    def test_k_3_with_ratios(self):
        verbose = False

        def has_any_none(items):
            for item in items:
                if isinstance(item, collections.Iterable):
                    if self._has_none(item):
                        return True
                else:
                    if item is None:
                        return True
            return False

        def make_trace_record(index, test):
            return pd.Series(
                data={'oasspread': test.oasspread})

        Test = collections.namedtuple(
            'Test',
            'reclassified_trade_type oasspread target b_history s_history b_ratios s_ratios',
        )
        # assume k = 3 ; columns
        #                   [target]  [b history]         [s history]
        #                             [b ratios]         [s ratios]
        tests = (
            Test('B', 100,  None,    (None, None, None),  (None, None, None),
                                     (None, None),        (None, None)),
            Test('S', 103,  None,    (100,  None, None),  (None, None, None),
                                     (None, None),        (None, None)),
            Test('S', 104,  104/103, (100,  None, None),  (103, None, None),
                                     (None, None),        (None, None)),
            Test('B', 101,  101/100, (100,  None, None),  (104,  103, None),
                                     (None, None),        (104/103, None)),
            Test('B', 102,  102/101, (101,  100,  None),  (104,  103, None),
                                     (101/100, None),     (104/103, None)),
            Test('S', 105,  105/104, (102,  101,  100),   (104,  103, None),
                                     (102/101, 101/100),  (104/103, None)),
            Test('S', 106,  106/105, (102,  101,  100),   (105,  104,  103),
                                     (102/101, 101/100),  (105/104, 104/103)),
            Test('B', 103,  103/102, (102,  101,  100),   (106,  105,  104),
                                     (102/101, 101/100),  (106/105, 105/104)),
            Test('B',  99,   99/103, (103,  102,  101),   (106,  105,  104),
                                     (103/102, 102/101),  (106/105, 105/104)),
        )
        feature_maker = HistoryOasspread(history_length=3)
        for i, test in enumerate(tests):
            if verbose:
                print test
            features, err = feature_maker.make_features(
                trace_index=i,
                trace_record=make_trace_record(i, test),
                extra={'id_reclassified_trade_type': test.reclassified_trade_type},
            )
            if has_any_none(test):
                self.assertTrue(features is None)
                self.assertTrue(err is not None)
            else:
                if verbose:
                    print features
                self.assertTrue(features is not None)
                self.assertTrue(err is None)
                for i, h in enumerate(test.b_history):
                    self.assertAlmostEqual(h, features['oasspread_B_back_%02d' % (i + 1)])
                for i, h in enumerate(test.s_history):
                    self.assertAlmostEqual(h, features['oasspread_S_back_%02d' % (i + 1)])
                self.assertAlmostEqual(test.b_ratios[0], features['oasspread_ratio_B_back_01_to_02'])
                self.assertAlmostEqual(test.b_ratios[1], features['oasspread_ratio_B_back_02_to_03'])
                self.assertAlmostEqual(test.s_ratios[0], features['oasspread_ratio_S_back_01_to_02'])
                self.assertAlmostEqual(test.s_ratios[1], features['oasspread_ratio_S_back_02_to_03'])
                self.assertAlmostEqual(test.target, features['id_oasspread_ratio'])


class HistoryPrice(FeatureMaker):
    def __init__(self, history_length):
        super(HistoryPrice, self).__init__('HistoryPrice(history_length=%s)' % history_length)
        self.history_length = history_length
        self.deque = collections.deque(maxlen=history_length)

    def make_features(self, trace_index, trace_record, extra):
        'return (features, err)'
        def accumulate_history():
            price = trace_record['price']
            if isinstance(price, numbers.Number):
                self.deque.append(price)
            else:
                return (None, 'price is %s, which is not a number' % price)

        if len(self.deque) < self.history_length:
            accumulate_history()
            return (None, 'not %d historic prices' % self.history_length)

        features = {}
        for k in range(self.history_length):
            key = 'price_back_%02d' % (self.history_length - k)
            features[key] = self.deque[k]

        accumulate_history()
        return (features, None)


class HistoryQuantity(FeatureMaker):
    def __init__(self, history_length):
        super(HistoryQuantity, self).__init__('HistoryQuantity(history_length=%s)' % history_length)
        self.history_length = history_length
        self.deque = collections.deque(maxlen=history_length)

    def make_features(self, trace_index, trace_record, extra):
        'return (features, err)'
        def accumulate_history():
            quantity = trace_record['quantity']
            if isinstance(quantity, numbers.Number):
                self.deque.append(quantity)
            else:
                return (None, 'quantity is %s, which is not a number' % quantity)

        if len(self.deque) < self.history_length:
            accumulate_history()
            return (None, 'not %d historic prices' % self.history_length)

        features = {}
        for k in range(self.history_length):
            key = 'quantity_back_%02d' % (self.history_length - k)
            features[key] = self.deque[k]

        accumulate_history()
        return (features, None)


class HistoryQuantityWeightedAverageSpread(FeatureMaker):
    'weighted average oasspread including the current trade'
    def __init__(self, history_length):
        super(HistoryQuantityWeightedAverageSpread, self).__init__(
            'VolumeWeightedAverageSpread(history_length=%s)' % history_length
        )
        self.history_length = history_length
        self.deque = collections.deque(maxlen=history_length)

    def make_features(self, trace_index, trace_record, extra):
        'return (features, err)'
        Item = collections.namedtuple('Item', 'oasspread quantity')

        def accumulate_history():
            oasspread = trace_record['oasspread']
            if not isinstance(oasspread, numbers.Number):
                return (None, 'oasspread is %s, which is not a number' % oasspread)
            quantity = trace_record['quantity']
            if not isinstance(quantity, numbers.Number):
                return (None, 'quantity is %s, which is not a number' % quantity)
            item = Item(
                oasspread=oasspread,
                quantity=quantity,
            )
            self.deque.append(item)

        accumulate_history()  # include the current trade in the weighted average

        if len(self.deque) < self.history_length:
            return (None, 'not %d historic oasspread-quantities' % self.history_length)

        key = 'quantity_weighted_average_oasspread_for_last_%02d_trace_prints' % self.history_length
        total_quantity = sum([item.quantity for item in self.deque])
        weighted_sum = sum([item.oasspread * item.quantity / total_quantity for item in self.deque])
        features = {
            key: weighted_sum
        }
        return (features, None)


class QuantityWeightedAverageSpreadTest(unittest.TestCase):
    def test(self):
        verbose = False
        Test = collections.namedtuple('Test', 'quantity oasspread expected')

        def make_trace_record(test):
            return pd.Series(
                data={
                    'quantity': test.quantity,
                    'oasspread': test.oasspread,
                })

        tests = (
            Test(1, 100, None),
            Test(2, 200, (1*100 + 2*200) / 3),
            Test(3, 300, (2*200 + 3*300) / 5),
            Test(4, 100, (3*300 + 4*100) / 7),
        )
        average = HistoryQuantityWeightedAverageSpread(history_length=2)
        for i, test in enumerate(tests):
            if verbose:
                print test
            trace_record = make_trace_record(test)
            features, err = average.make_features(
                trace_index=0,
                trace_record=trace_record,
                extra={},
            )
            if verbose:
                print features
                print err
            if test.expected is None:
                self.assertTrue(features is None)
                self.assertTrue(isinstance(err, str))
            else:
                self.assertTrue(features is not None)
                self.assertTrue(err is None)
                for k, v in features.iteritems():
                    self.assertAlmostEqual(test.expected, v)  # just one feature


class InterarrivalTime(FeatureMaker):
    def __init__(self):
        super(InterarrivalTime, self).__init__('InterarrivalTime')
        self.last_effectivedatetime = None

    def make_features(self, trace_index, trace_record, extra):
        'return (features, err)'
        def accumulate_history():
            self.last_effectivedatetime = trace_record['effectivedatetime']

        if self.last_effectivedatetime is None:
            accumulate_history()
            return (None, 'no prior trace record')
        else:
            interval = trace_record['effectivedatetime'] - self.last_effectivedatetime
            # interval: Timedelta, a subclass of datetime.timedelta
            # attributes of a datetime.timedelta are days, seconds, microseconds
            interarrival_seconds = (interval.days * 24.0 * 60.0 * 60.0) + (interval.seconds * 1.0)
            assert interarrival_seconds >= 0.0  # trace print file not sorted in ascending datetime order
            features = {
                'interarrival_seconds_size': interarrival_seconds,
            }
            accumulate_history()
            return (features, None)


class InterarrivalTimeTest(unittest.TestCase):
    def test1(self):
        Test = collections.namedtuple('Test', 'minute second expected_interval')
        tests = (  # (minute, second, expected_interval)
            Test(10, 0, None),
            Test(10, 0, 0),
            Test(10, 1, 1),
            Test(10, 20, 19),
            Test(10, 20, 0),
        )
        iat = InterarrivalTime()
        for test in tests:
            trace_record = {}
            trace_record['effectivedatetime'] = datetime.datetime(2016, 11, 3, 6, test.minute, test.second)
            features, err = iat.make_features(
                trace_index=None,
                trace_record=trace_record,
                extra={},
            )
            if test.expected_interval is None:
                self.assertTrue(features is None)
                self.assertTrue(isinstance(err, str))  # it contains an error message
            else:
                self.assertEqual(test.expected_interval, features['interarrival_seconds_size'])
                self.assertTrue(err is None)


class Ohlc(FeatureMaker):
    'ratio_days of delta ticker / delta spx for closing prices'
    def __init__(self, df_ticker=None, df_spx=None, verbose=False):
        'precompute all results'
        pdb.set_trace()
        super(Ohlc, self).__init__('ohlc')

        self.df_ticker = df_ticker  # a DataFreame
        self.df_spx = df_spx        # a DataFrame
        self.skipped_reasons = collections.Counter()

        self.days_back = [
            1 + days_back
            for days_back in xrange(30)
        ]
        self.ratio_day, self.dates_list = self._make_ratio_day(df_ticker, df_spx)
        self.dates_set = set(self.dates_list)

    def make_features(self, trace_index, trace_record, extra):
        'return Dict[feature_name, feature_value], err'
        pdb.set_trace()
        date = trace_record['effectivedatetime'].date()
        if date not in self.dates_set:
            return False, 'ohlc: date %s not in ticker and spx input files' % date
        result = {}
        feature_name_template = 'p_price_delta_ratio_back_%s_%02d'
        for days_back in self.days_back:
            key = (date, days_back)
            if key not in self.ratio_day:
                return False, 'ohlc: missing date %s days_back %s in ticker and spx input files' % (
                    date,
                    days_back,
                )
            feature_name = feature_name_template % ('days', days_back)
            result[feature_name] = self.ratio_day[key]
        return result, None

    def _make_ratio_day(self, df_ticker, df_spx):
        'return Dict[date, ratio]'
        verbose = False

        closing_price_spx = {}         # Dict[date, closing_price]
        closing_price_ticker = {}
        dates_list = []
        ratio = {}
        for timestamp in sorted(df_ticker.index):
            ticker = df_ticker.loc[timestamp]
            if timestamp.date() not in df_spx.index:
                msg = 'FeatureMakerOhlc: skipping index %s is in ticker but not spx' % timestamp
                self.skipped_reasons[msg] += 1
                continue
            spx = df_spx.loc[timestamp]

            date = timestamp.date()
            dates_list.append(date)
            closing_price_spx[date] = spx.Close
            closing_price_ticker[date] = ticker.Close
            if verbose:
                print 'trades', date, 'ticker', ticker.Close, 'spx', spx.Close
            for days_back in self.days_back:
                # detemine for calendar days (which might fail)
                # assume that the calendar dates are a subset of the market dates
                stop_date = self._adjust_date(dates_list, 2)
                start_date = self._adjust_date(dates_list, 2 + days_back)
                if stop_date is None or start_date is None:
                    msg = 'no valid date for trade date %s days_back %d' % (date, days_back)
                    if verbose:
                        print msg
                    self.skipped_reasons[msg] += 1
                    continue
                ratio[(date, days_back)] = self._ratio_day(
                    closing_price_spx,
                    closing_price_ticker,
                    start_date,
                    stop_date,
                )
                if verbose:
                    print 'ratio', days_back, start_date, stop_date, date, ratio[(date, days_back)]
        return ratio, dates_list

    def _adjust_date(self, dates_list, days_back):
        'return date if its in valid_dates, else None'
        try:
            return dates_list[-days_back]
        except:
            return None

    def _ratio_day(self, prices_spx, prices_ticker, start, stop):
        delta_ticker = prices_ticker[start] * 1.0 / prices_ticker[stop]
        delta_spx = prices_spx[start] * 1.0 / prices_spx[stop]
        ratio_day = delta_ticker * 1.0 / delta_spx
        return ratio_day


class OrderImbalance(FeatureMaker):
    'features related to the order imbalance'
    def __init__(self, lookback=None, typical_bid_offer=None, proximity_cutoff=None):
        super(OrderImbalance, self).__init__(
            'OrderImbalance(lookback=%s, typical_bid_offer=%s, proximity_cutff=%s)' % (
                lookback,
                typical_bid_offer,
                proximity_cutoff,
            ))
        self.order_imbalance4_object = OrderImbalance4.OrderImbalance4(
            lookback=lookback,
            typical_bid_offer=typical_bid_offer,
            proximity_cutoff=proximity_cutoff,
        )
        self.order_imbalance4 = None
        self.all_trade_types = ('B', 'D', 'S')

    def make_features(self, trace_index, trace_record, extras, verbose=False):
        'return (Dict, err)'
        'return None or error message'
        oasspread = trace_record['oasspread']
        price = trace_record['price']
        quantity = trace_record['quantity']
        trade_type = trace_record['trade_type']

        assert trade_type in self.all_trade_types
        # check for NaN values
        if oasspread != oasspread:
            return (None, 'oasspread is NaN')
        if price != price:
            return (None, 'price is NaN')
        if quantity != quantity:
            return (None, 'quantity is NaN')
        if trade_type not in self.all_trade_types:
            return (None, 'unexpected trade type %d for trace print %s' % (
                trade_type,
                trace_index,
                ))

        order_imbalance4_result = self.order_imbalance4_object.imbalance(
            trade_type=trade_type,
            trade_quantity=quantity,
            trade_price=price,
        )
        order_imbalance, reclassified_trade_type, err = order_imbalance4_result
        if err is not None:
            return (None, 'trade type not reclassfied: ' + err)

        features = {
            'orderimbalance': order_imbalance,
            'id_reclassified_trade_type': reclassified_trade_type,
            'reclassified_trade_type_is_B': 1 if reclassified_trade_type is 'B' else 0,
            'reclassified_trade_type_is_S': 1 if reclassified_trade_type is 'S' else 0,
        }
        return (features, None)


class SecurityMaster(FeatureMaker):
    def __init__(self, df):
        super(SecurityMaster, self).__init__('SecurityMaster')
        self.df = df  # the security master records

    def make_features(self, trace_index, trace_record, extra):
        'return (Dict[feature_name, feature_value], None) or (None, err)'
        cusip = trace_record['cusip']
        if cusip not in self.df.index:
            return (None, 'cusip %s not in security master' % cusip)
        security = self.df.loc[cusip]

        result = {
            'coupon_type_is_fixed_rate': 1 if security['coupon_type'] == 'Fixed rate' else 0,
            'coupon_type_is_floating_rate': 1 if security['coupon_type'] == 'Floating rate' else 0,
            'original_amount_issued_size': security['original_amount_issued'],
            'months_to_maturity_size': self._months_from_until(trace_record['effectivedate'], security['maturity_date']),
            'months_of_life_size': self._months_from_until(security['issue_date'], security['maturity_date']),
            'is_callable': 1 if security['is_callable'] else 0,
            'is_puttable': 1 if security['is_puttable'] else 0,
        }
        return result, None

    def _months_from_until(self, a, b):
        'return months from date a to date b'
        delta_days = (b - a).days
        return delta_days / 30.0


class TimeVolumeWeightedAverage(object):
    def __init__(self, k):
        assert k > 0
        self.k = k
        self.history = collections.deque([], k)

    def weighted_average(self, amount, volume, timestamp):
        'accumulate amount and volume and return (weighted_average: float, err)'
        def as_days(timedelta):
            'convert pandas Timedelta to number of days'
            seconds_per_day = 24.0 * 60.0 * 60.0
            return (
                timedelta.components.days +
                timedelta.components.hours / 24.0 +
                timedelta.components.minutes / (24.0 * 60.0) +
                timedelta.components.seconds / seconds_per_day +
                timedelta.components.milliseconds / (seconds_per_day * 1e3) +
                timedelta.components.microseconds / (seconds_per_day * 1e6) +
                timedelta.components.nanoseconds / (seconds_per_day * 1e9)
            )

        self.history.append((amount, volume, timestamp))
        if len(self.history) != self.k:
            return None, 'not yet k=%d observations' % self.k
        weighted_amount_sum = 0.0
        weighted_volume_sum = 0.0
        for amount, volume, ts in self.history:
            days_back = as_days((ts - timestamp))  # in fractions of a day
            assert days_back <= 0.0
            weight = math.exp(days_back)
            weighted_amount_sum += amount * volume * weight
            weighted_volume_sum += volume * weight
        if weighted_volume_sum == 0.0:
            return None, 'time-weighted volumes sum to zero'
        return weighted_amount_sum / weighted_volume_sum, None


class TimeVolumeWeightedAverageTest(unittest.TestCase):
    def test(self):
        def t(hour, minute):
            'return datetime.date'
            return pd.Timestamp(2016, 11, 1, hour, minute, 0)

        TestCase = collections.namedtuple('TestCase', 'k spreads_quantities expected')
        data = ((t(10, 00), 100, 10), (t(10, 25), 200, 20), (t(10, 30), 300, 30))  # [(time, value, quantity)]
        tests = (
            TestCase(1, data, 300),
            TestCase(2, data, 240.06),
            TestCase(3, data, (100 * 10 * 0.979 + 200 * 20 * 0.996 + 300 * 30) / (10 * 0.979 + 20 * 0.996 + 30)),
            TestCase(4, data, None),
        )
        for test in tests:
            if test.k != 3:
                continue
            rwa = TimeVolumeWeightedAverage(test.k)
            for spread_quantity in test.spreads_quantities:
                time, spread, quantity = spread_quantity
                actual, err = rwa.weighted_average(spread, quantity, time)
                # we check only the last result
            if test.expected is None:
                self.assertEqual(actual, None)
                self.assertTrue(err is not None)
            else:
                self.assertAlmostEqual(test.expected, actual, 1)
                self.assertTrue(err is None)


class TraceTradetypeContext(object):
    # has been replaced by OrderImbalance, HistoryPrice, and HitoryQuantity
    # todo: DELETE ME
    'accumulate running info from trace prints for each trade type'
    def __init__(self, lookback=None, typical_bid_offer=None, proximity_cutoff=None):
        print 'deprecated: use OrderImbalance, PriceHistory, QuantityHistory instead'
        self.order_imbalance4_object = OrderImbalance4.OrderImbalance4(
            lookback=lookback,
            typical_bid_offer=typical_bid_offer,
            proximity_cutoff=proximity_cutoff,
        )
        self.order_imbalance4 = None

        # each has type Dict[trade_type, number]
        self.prior_oasspread = {}
        self.prior_price = {}
        self.prior_quantity = {}

        self.all_trade_types = ('B', 'D', 'S')

    def update(self, trace_record, verbose=False):
        'return (Dict, err)'
        'return None or error message'
        oasspread = trace_record['oasspread']
        price = trace_record['price']
        quantity = trace_record['quantity']
        trade_type = trace_record['trade_type']

        if verbose:
            print 'context', trace_record['cusip'], trade_type, oasspread

        assert trade_type in self.all_trade_types
        # check for NaN values
        if oasspread != oasspread:
            return (None, 'oasspread is NaN')
        if price != price:
            return (None, 'price is NaN')
        if quantity != quantity:
            return (None, 'quantity is NaN')

        order_imbalance4_result = self.order_imbalance4_object.imbalance(
            trade_type=trade_type,
            trade_quantity=quantity,
            trade_price=price,
        )
        order_imbalance, reclassified_trade_type, err = order_imbalance4_result
        if err is not None:
            return (None, 'trace print not reclassfied: ' + err)

        assert order_imbalance is not None
        assert reclassified_trade_type in ('B', 'S')
        # the updated values could be missing, in which case, they are np.nan values
        self.prior_oasspread[trade_type] = oasspread
        self.prior_price[trade_type] = price
        self.prior_quantity[trade_type] = quantity

        # assure that we know the prior values the caller is looking for
        for trade_type in self.all_trade_types:
            if trade_type not in self.prior_oasspread:
                return (None, 'no prior oasspread for trade type %s' % trade_type)
            if trade_type not in self.prior_price:
                return (None, 'no prior price for trade type %s' % trade_type)
            if trade_type not in self.prior_quantity:
                return (None, 'no prior quantity for trade type %s' % trade_type)

        d = {
            'orderimbalance': order_imbalance,
            'reclassified_trade_type': reclassified_trade_type,
            'prior_price': self.prior_price,
            'prior_quantity': self.prior_quantity,
        }
        return (d, None)

    def missing_any_prior_oasspread(self):
        'return None or error'
        missing = []
        for trade_type in self.all_trade_types:
            if trade_type not in self.prior_oasspread:
                missing.append(trade_type)
        return (
            None if len(missing) == 0 else
            'missing prior oasspread for trade type %s' % missing
        )


class VolumeWeightedAverage(object):
    def __init__(self, k):
        assert False, 'deprecated: use HistorYQuantityWeightedAverageSpread instead'
        assert k > 0
        self.k = k
        self.history = collections.deque([], k)

    def weighted_average(self, amount, volume):
        'accumulate amount and volume and return (weighted_average: float, err)'
        self.history.append((amount, volume))
        if len(self.history) != self.k:
            return None, 'not yet k=%d observations' % self.k
        weighted_amount_sum = 0.0
        weighted_volume_sum = 0.0
        for amount, volume in self.history:
            weighted_amount_sum += amount * volume
            weighted_volume_sum += volume
        if weighted_volume_sum == 0.0:
            return None, 'volums sum to zero'
        return weighted_amount_sum / weighted_volume_sum, None


class VolumeWeightedAverageTest(unittest.TestCase):
    def test(self):
        return
        TestCase = collections.namedtuple('TestCase', 'k spreads_quantities expected')
        data = ((100, 10), (200, 20), (300, 30))  # [(value, quantity)]
        tests = (
            TestCase(1, data, 300),
            TestCase(2, data, 260),
            TestCase(3, data, 233.33),
            TestCase(4, data, None),
        )
        for test in tests:
            rwa = VolumeWeightedAverage(test.k)
            for spread_quantity in test.spreads_quantities:
                spread, quantity = spread_quantity
                actual, err = rwa.weighted_average(spread, quantity)
                # we check only the last result
            if test.expected is None:
                self.assertEqual(actual, None)
                self.assertTrue(err is not None)
            else:
                self.assertAlmostEqual(test.expected, actual, 2)
                self.assertTrue(err is None)

############################################
# features file in MidPredictor/automatic_feeds
# in order listed in that directory
############################################


class Fundamentals(FeatureMaker):
    'DEPREACTED. Use History as a base class instead'
    __metaclass__ = abc.ABCMeta

    def __init__(self, args, event, numeric_fields_not_size, numeric_fields_size, trace=False):
        self._args = args
        self._creation_event = copy.copy(event)
        self._numeric_fields_not_size = copy.copy(numeric_fields_not_size)
        self._numeric_fields_size = copy.copy(numeric_fields_size)
        self._trace = trace
        super(Fundamentals, self).__init__(args, event, event.source)

    def make_features(self, event):
        'return (event_features, errs)'
        if self._trace:
            print 'make_features', self
            pp(event.payload)
            pdb.set_trace()
        d = {
            'id_event': copy.copy(event),
        }
        for k, v in event.payload.iteritems():
            if k in self._numeric_fields_not_size or k in self._numeric_fields_size:
                try:
                    value = float(v)
                except ValueError:
                    err = 'value %s is not convertable to a float' % v
                    return None, [err]
                if k in self._numeric_fields_not_size:
                    d['%s' % k] = value
                elif k in self._numeric_fields_size:
                    d['%s_size' % k] = value
        if self._trace:
            print 'event_features'
            pp(d)
            pdb.set_trace()
        return input_event.EventFeatures(d), None


class Etf(FeatureMaker):
    __metaclass__ = abc.ABCMeta

    def __init__(self, args, event, event_is_relevant, extract_weight):
        pdb.set_trace()
        self._args = args
        self._creation_event = copy.copy(event)
        self._event_is_relevant = event_is_relevant
        self._extract_weight = extract_weight
        super(Etf, self).__init(args, event, event.source)

    def make_features(self, event):
        'return (event_features, errs)'
        # make feature_events only for the query cusip
        pdb.set_trace()
        if self._event_is_relevant(event):
            pdb.set_trace
            try: 
                s = self._extract_weight(event)
                value = float(s)
            except ValueError:
                err = 'value %s is not convertable to a float' % s
                return None, [err]
            d = {
                'id_event': copy.copy(event),
                'weight_size': value,
            }
            return input_event.EventFeatures(d), None
        else:
            err = 'event not relevant'
            return None, [err]


class History(FeatureMaker):
    'maintain history of an event-feature and its ratios'
    # features createed are
    #   {feature_name}_prior_0   (value from most recent relevant event)
    #   {feature_name}_prior_1   (value from the just prior relevant event)
    #   {feature_name}_prior_2
    #   ...
    #   {feature_name}_ratio_0_to_1  (ratio of current value to just-prior value)
    #   {feature_name}_ratio_1_to_2

    def __init__(self, args, event, k, feature_name, event_is_relevant, extract_feature_value):
        assert k >= 1
        self._args = args
        self._event = copy.copy(event)
        self._k = k
        self._feature_name = feature_name
        self._event_is_relevant = event_is_relevant
        self._extract_feature_value = extract_feature_value

        self._prior_values = collections.deque([], k)  # keep k most recent values

    def make_features(self, event):
        'return (event_features, errs)'
        if self._event_is_relevant(event):
            try:
                s = self._extract_feature_value(event.payload)
                value = float(s)
            except ValueError:
                err = 'string %s was not converted to a float' % s
                return None, [err]
            self._prior_values.append(value)

            if len(self._prior_values) < self._k:
                err = 'have %d values, not yet the required %d' % (
                    len(self._prior_values),
                    self._k,
                )
                return None, [err]

            # have k values, so produce the features
            d = {
                'id_event': copy.copy(event),
            }
            ordered_values = list(reversed(list(self._prior_values)))  # reversed return an iterable
            i = 0
            while True:
                key1 = '%s_prior_%d' % (self._feature_name, i)
                d[key1] = ordered_values[i]
                if i + 1 == len(ordered_values):
                    break
                if ordered_values[i + 1] == 0.0:
                    err = 'value in prior %k is zero' % i + 1
                    return None, [err]
                key2 = '%s_ratio_%d_to_%d' % (self._feature_name, i, i + 1)
                d[key2] = ordered_values[i] / (1.0 * ordered_values[i + 1])
                i += 1
            return input_event.EventFeatures(d), None
        else:
            pdb.set_trace()
            err = 'event is not relevant'
            return None, [err]


class EtfWeight(FeatureMaker):
    pass


class AmtOutstandingHistory(FeatureMaker):  # Note: file not in same format as fundamentals files
    pass


class CurrentCoupon(FeatureMaker):  # Note: file not in same format as fundamentals files
    pass


class EtfWeightOfCusipPctAgg(Etf):
    def __init__(self, args, event):
        pdb.set_trace()
        super(EtfWeightOfCusipPctAgg, self).__init__(
            args,
            event,
            event_is_relevant=lambda row: row['cusip'] == args.cusip,
            extract_weight=lambda row: row['weight'],
            trace=True,
        )


class EtfWeightOfCusipPctLqd(EtfWeight):
    pass


class EtfWeightOfIssuerPctAgg(EtfWeight):
    pass


class EtfWeightOfIssuerPctLqd(EtfWeight):
    pass


class EtfWeightOfSectorPctAgg(EtfWeight):
    pass


class EtfWeightOfSectorPctLqd(EtfWeight):
    pass


class FunExpectedInterestCoverage(History):
    def __init__(self, args, event):
        super(FunExpectedInterestCoverage, self).__init__(
            args=args,
            event=event,
            k=2,
            feature_name='interest_coverage',
            event_is_relevant=lambda row: True,
            extract_feature_value=lambda row: row['interest_coverage'],
        )


class FunGrossLeverage(History):
    def __init__(self, args, event):
        super(FunGrossLeverage, self).__init__(
            args=args,
            event=event,
            k=2,
            feature_name='gross_leverage',
            event_is_relevant=lambda row: True,
            extract_feature_value=lambda row: row['gross_leverage'],
        )


class FunLtmEbitda(History):
    def __init__(self, args, event):
        super(FunLtmEbitda, self).__init__(
            args=args,
            event=event,
            k=2,
            feature_name='LTM_EBITDA',
            event_is_relevant=lambda row: True,
            extract_feature_value=lambda row: row['LTM_EBITDA'],
        )


class FunMktCap(History):
    def __init__(self, args, event):
        super(FunMktCap, self).__init__(
            args=args,
            event=event,
            k=2,
            feature_name='mkt_cap',
            event_is_relevant=lambda row: True,
            extract_feature_value=lambda row: row['mkt_cap'],
        )


class FunMktGrossLeverage(History):
    def __init__(self, args, event):
        super(FunMktGrossLeverage, self).__init__(
            args=args,
            event=event,
            k=2,
            feature_name='mkt_gross_leverage',
            event_is_relevant=lambda row: True,
            extract_feature_value=lambda row: row['mkt_gross_leverage'],
        )


class FunReportedInterestCoverage(History):
    def __init__(self, args, event):
        super(FunReportedInterestCoverage, self).__init__(
            args=args,
            event=event,
            k=2,
            feature_name='interest_coverage',
            event_is_relevant=lambda row: True,
            extract_feature_value=lambda row: row['interest_coverage'],
        )


class FunTotalAssets(History):
    def __init__(self, args, event):
        super(FunTotalAssets, self).__init__(
            args=args,
            event=event,
            k=2,
            feature_name='total_assets',
            event_is_relevant=lambda row: True,
            extract_feature_value=lambda row: row['total_assets'],
        )


class FunTotalDebt(History):
    def __init__(self, args, event):
        super(FunTotalDebt, self).__init__(
            args=args,
            event=event,
            k=2,
            feature_name='total_debt',
            event_is_relevant=lambda row: True,
            extract_feature_value=lambda row: row['total_debt'],
        )


class HistEquityPrices(FeatureMaker):  # NOTE: not in same format as fundamentals files
    pass


class LiqFlowOnTheRun(FeatureMaker):
    def __init__(self, arg, event):
        self._arg = arg
        self._construction_event = copy.copy(event)

        self._query_cusip = arg.cusip

    def make_features(self, event):
        'return (event.EventFeatures, errs)'
        # return all the fields as identifiers
        primary_cusip = event.payload['primary_cusip']
        if primary_cusip != self._query_cusip:
            err = 'otr for %s, not the query cusip %s' % (primary_cusip, self._query_cusip)
            return None, [err]
        d = {
            'id_liq_flow_on_the_run_event': copy.copy(event),
            'liq_flow_on_the_run_otr_cusip': event.payload['otr_cusip'],
        }
        return input_event.EventFeatures(d), None


class SecMaster(Fundamentals):
    pass
    # CHEN: Please remind that this set of event features and others are not of the issuer,
    # but are of the cusip. (The calling program needs to handle them separately.)


####################################################
# trace print event features
####################################################

class Trace(FeatureMaker):
    'accumlate features for a specific cusip'
    def __init__(self, args, event):
        self._args = args
        self._construction_event = copy.copy(event)

        self._construction_cusip = event.payload['cusip']
        self._prior_oasspread = None
    
    def make_features(self, event):
        'return (event_features, errs)'
        assert event.cusip() == self._construction_cusip
        try:
            s = event.payload['oasspread']
            oasspread = float(s)
        except ValueError:
            err = 'oasspread string %s was not converted to a float' % s
            return None, [err]
        prior_oasspread = self._prior_oasspread
        self._prior_oasspread = oasspread

        if prior_oasspread is None:
            err = 'no prior trace event for cusip %s' % self._construction_cusip
            return None, [err]
        if prior_oasspread == 0.0:
            err = 'prior oasspread has zero value'
            return None, [err]
        d = {
            'id_trace_event': copy.copy(event),
            'trace_oasspread': oasspread,
            'trace_oaspread_divided_by_prior': oasspread / prior_oasspread,
            'trace_oasspread_less_prior': oasspread - prior_oasspread,
        }
        return input_event.EventFeatures(d), None


class TraceOLD(FeatureMaker):
    'create features from last k trace print events'
    def __init__(self, issuer, cusip):
        pdb.set_trace()
        super(Trace, self).__init__(issuer, cusip, name='Trace')
        self.k = 2
        self.prior_features = collections.deque([], self.k)

        self.coded_field_values = {
            'salescondcode': ['', '@', 'A', 'C', 'N', 'R', 'S', 'T', 'U', 'W', 'Z'],
            'secondmodifier': ['A', 'S', 'W', 'Z'],
            'wiflag': ['N'],
            'commissionflag': ['N', 'Y'],
            'asofflag': ['A', 'R', 'S'],
            'specialpriceindicator': ['N', 'Y'],
            'yielddirection': ['-', '+', 'S'],
            'halt': ['N'],
            'cancelflag': ['N', 'Y'],
            'correctionflag': ['N', 'Y'],
            'trade_type': ['D', 'B', 'S'],
            'is_suspect': ['N', 'Y'],
            'reclassified_trade_type': ['D', 'B', 'S']
        }
        self.found_field_values = collections.defaultdict(set)
        self.numeric_field_values = {  # value is whether the feature is a _size feature
            'price': True,
            'yield': True,
            'quantity': True,
            'estimatedquantity': True,
            'mka_oasspread': False,
            'oasspread': False,
            'convexity': True
        }

        self.initial_cusip = None

    def make_features(self, id, payload, debug=False):
        'return (features, errs)'
        assert isinstance(id, EventId.TraceEventId)
        assert isinstance(payload, dict)

        def add_coded_features(payload, features, errors):
            'mutate features to include 1-of-K encoding for each coded feature'
            for field_name, expected_field_values in self.coded_field_values.iteritems():
                field_value = payload[field_name]
                if field_value in expected_field_values:
                    for expected_field_value in expected_field_values[1:]:
                        # [1:] ==> do not encode the first value
                        # if you do, linear methods without regularizers will see colinear features
                        # collinear features cause some linear methods to fail to fit
                        key = '%s_is_%s' % (
                            field_name,
                            expected_field_value
                        )
                        value = 1 if field_value == expected_field_value else 0
                        features[key] = value
                else:
                    errors.append('unexpected value %s in field %s' % (field_value, field_name))

        def add_numeric_features(payload, features, errors):
            'mutate features to include the numeric values'
            for field_name, is_size_feature in self.numeric_field_values.iteritems():
                field_value_str = payload[field_name]
                try:
                    field_value = float(field_value_str)
                except:
                    errors.append('field %s=%s, which is not a float' % (field_name, field_value_str))
                    continue
                if is_size_feature:
                    features[field_name + '_size'] = field_value
                else:
                    features[field_name] = field_value

        def make_features_errors(id, payload):
            'return (features, errors)'
            features = {}
            errors = []
            add_coded_features(payload, features, errors)
            add_numeric_features(payload, features, errors)
            return (features, errors)

        cusip = payload['cusip']
        if self.initial_cusip is None:
            self.initial_cusip == cusip
        else:
            # caller must call me always for the same cusip, because
            # I accumulate info about cusips
            assert self.cusip == self.initial_cusip
        features, errors = make_features_errors(id, payload)
        if len(errors) > 0:
            return (None, errors)
        self.prior_features.append(features)
        if len(self.prior_features) < self.prior_features.maxlen:
            err = 'have not seen %d no-error trace events from cusip % s' % (
                self.prior_features.maxlen,
                payload['cusip'],
            )
            return (None, [err])

        # rename the features
        renamed_features = {}
        for i, prior_features in enumerate(self.prior_features):
            for feature_name, feature_value in prior_features.iteritems():
                new_name = 'trace_prior_%d_%s' % (len(self.prior_features) - (i + 1), feature_name)
                renamed_features[new_name] = feature_value

        # add identifiers
        if debug:
            pdb.set_trace()
        renamed_features['id_trace_event_id'] = id
        renamed_features['id_trace_event_payload'] = payload
        return (renamed_features, None)


if __name__ == '__main__':
    unittest.main()


if False:
    # avoid errors from linter
    pdb
    pprint
