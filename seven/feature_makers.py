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
'''

from __future__ import division

import collections
import copy
import datetime
import math
import numbers
import pandas as pd
import pdb
from pprint import pprint
import unittest


from applied_data_science.timeseries import FeatureMaker

# import seven.OrderImbalance4
# import seven.read_csv
import OrderImbalance4
import read_csv


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


class Skipped(object):
    def __init__(self):
        self.counter = collections.Counter()
        self.n_skipped = 0

    def skip(self, reason):
        self.counter[reason] += 1
        self.n_skipped += 1


class AllFeatures(object):
    def __init__(self, ticker, primary_cusip, otr_cusips):
        self.ticker = ticker
        self.primary_cusip = primary_cusip
        self.otr_cusips = otr_cusips

        self.last_effectivedatetime = None

        # Dict[cusip, OrderedDict[trace_index, Dict[feature, value]]  about 50 features
        self.primary_features = collections.defaultdict(lambda: collections.OrderedDict())
        # Dict[cusip, Dict[trace_index_of_cusip, (otr_cusip, index_in_otr_cusip)]]
        self.otr_index = collections.defaultdict(dict)

        self.features_ticker_cusip = {}
        for otr_cusip in otr_cusips.union(set(primary_cusip)):
            self.features_ticker_cusip[otr_cusip] = FeaturesTickerCusip(ticker, otr_cusip)

    def append_features(self, trace_index, trace_record):
        'return None or error_message:str; mutate self.features_ticker_cusip[cusip]'
        # assure that datetimes are in non-decreasing order
        effectivedatetime = trace_record['effectivedatetime']
        if self.last_effectivedatetime is not None:
            assert self.last_effectivedatetime <= effectivedatetime
        self.last_effectivedatetime = effectivedatetime

        cusip = trace_record['cusip']
        assert cusip == self.primary_cusip or cusip in self.otr_cusips
        features_values, err = self.features_ticker_cusip[cusip].make_features(trace_index, trace_record)
        if err is not None:
            return err
        self.primary_features[cusip][trace_index] = features_values
        if cusip == self.primary_cusip:
            # save the cusip and index of the features for the related on-the-run cusip
            # NOTE: a cusip can be both primary and on the run; that is common
            otr_cusip = trace_record['cusip1']
            if otr_cusip not in self.primary_features:
                return 'have not yet seen OTR cusip %s for primary cusip %s' % (otr_cusip, cusip)
            otr_indices = self.primary_features[otr_cusip].keys()
            assert len(otr_indices) > 0  # because we always add a record when creating the item
            otr_last_index = otr_indices[-1]
            self.otr_index[cusip][trace_index] = (otr_cusip, otr_last_index)
        return None

    def get_primary_features_dataframe(self):
        'return pd.DataFrame for the primary cusip'
        def append_features(data, feature_values):
            'append primary and OTR features to the data'
            for feature_name, feature_value in features_values.iteritems():
                data[feature_name].append(feature_value)
            otr_index_dict = self.otr_index[self.primary_cusip]
            otr_cusip, index_in_otr_cusip = otr_index_dict[trace_index]
            otr_features_values = self.primary_features[otr_cusip][index_in_otr_cusip]
            for feature_name, feature_value in otr_features_values.iteritems():
                if feature_name.startswith('p_'):
                    new_feature_name = 'otr1_' + feature_name[2:]
                    data[new_feature_name].append(feature_value)
            return None

        def append_index(indices, trace_index):
            'append the trace_index to all the indices'
            indices.append(trace_index)
            return None

        data = collections.defaultdict(list)  # Dict[feature_name, [feature_values]]
        indices = []
        for trace_index, features_values in self.primary_features[self.primary_cusip].iteritems():
            append_features(data, features_values)
            append_index(indices, trace_index)
        result = pd.DataFrame(
            data=data,
            index=indices,
        )
        return result


class FeaturesTickerCusip(object):
    'create features for a single CUSIP for a specfied ticker'
    def __init__(self, ticker, cusip):
        self.ticker = ticker
        self.cusip = cusip
        self.all_feature_makers = None  # will be a tuple after lazy initialization
        # NOTE: callers rely on the definitions of skipped and features
        self.skipped = collections.Counter()
        self.features = pd.DataFrame()  # The API guarantees this feature

    def __str__(self):
        return 'FeaturesTickerCusip(target=%s,cusip%s)' % (self.ticker, self.cusip)

    def _initialize_all_feature_makers(self):
        'return initialized feature makers'
        # the lazy intiailization is done because we need a copy operation and eager initialization requires reading files
        result = (
            FeatureMakerEtf(
                df=read_csv.input(self.ticker, 'etf agg'),
                name='agg',
            ),
            FeatureMakerEtf(
                df=read_csv.input(self.ticker, 'etf lqd'),
                name='lqd',
            ),
            FeatureMakerFund(
                df=read_csv.input(self.ticker, 'fund'),
            ),
            FeatureMakerTradeId(),
            FeatureMakerOhlc(
                df_ticker=read_csv.input(self.ticker, 'ohlc ticker'),
                df_spx=read_csv.input(self.ticker, 'ohlc spx'),
            ),
            FeatureMakerSecurityMaster(
                df=read_csv.input(self.ticker, 'security master'),
            ),
            FeatureMakerTrace(
                order_imbalance4_hps={
                    'lookback': 10,
                    'typical_bid_offer': 2,
                    'proximity_cutoff': 20,
                },
            ),
        )
        return result

    def __deepcopy__(self, memo_dict):
        'return new AllFeatures instance'
        # ref: http://stackoverflow.com/questions/15214404/how-can-i-copy-an-immutable-object-like-tuple-in-python
        other = AllFeatures(self.ticker, self.cusip)
        other.all_feature_makers = copy.deepcopy(self.all_feature_makers, memo_dict)
        other.skipped = copy.deepcopy(self.skipped, memo_dict)
        other.features = copy.deepcopy(self.features, memo_dict)
        return other

    def make_features(self, trace_index, trace_record, verbose=False):
        'return (Dict[feature_name, feature_value], err)'
        if self.all_feature_makers is None:
            self.all_feature_makers = self._initialize_all_feature_makers()
        # assure all the trace_records are for the same CUSIP
        assert self.ticker == trace_record['ticker']
        assert self.cusip == trace_record['cusip']

        # accumulate features from the feature makers
        # stop on the first error from a feature maker
        all_features = {}  # Dict[feature_name, feature_value]
        for feature_maker in self.all_feature_makers:
            if verbose:
                print feature_maker.name
            features, error = feature_maker.make_features(trace_index, trace_record)
            if error is not None:
                self.skipped[error] += 1
                return None, error
            for k, v in features.iteritems():
                if k in all_features:
                    print 'duplicate feature', k
                    pdb.set_trace()
                all_features[k] = v

        return all_features, None

    def append_features(self, trace_index, trace_record, verbose=False):
        'return None or string with error message'
        if self.all_feature_makers is None:
            self.all_feature_makers = self._initialize_all_feature_makers()
        # assure all the trace_records are for the same CUSIP
        assert self.ticker == trace_record['ticker']
        assert self.cusip == trace_record['cusip']

        # accumulate features from the feature makers
        # stop on the first error from a feature maker
        all_features = {}  # Dict[feature_name, feature_value]
        for feature_maker in self.all_feature_makers:
            if verbose:
                print feature_maker.name
            features, error = feature_maker.make_features(trace_index, trace_record)
            if error is not None:
                self.skipped[error] += 1
                return error
            for k, v in features.iteritems():
                if k in all_features:
                    print 'duplicate feature', k
                    pdb.set_trace()
                all_features[k] = v

        df = pd.DataFrame(
            data=all_features,
            index=[trace_index]
        )
        self.features = self.features.append(df)
        return None


class FeatureMakerEtf(FeatureMaker):
    def __init__(self, df=None, name=None):
        # the df comes from a CSV file for the ticker and etf name
        # df format
        #  each row is a date as specified in the index
        #  each column is an isin as specified in the column names
        #  each df[date, isin] is the weight in an exchanged trade fund
        def invalid(isin):
            'return Bool'
            # the input files appear to duplicate the isin, by having two columns for each
            # the second column ends in ".1"
            # there are also some unnamed colums
            return isin.endswith('.1') or isin.startswith('Unnamed:')

        self.short_name = name
        super(FeatureMakerEtf, self).__init__('etf_' + name)
        self.df = df.copy(deep=True)
        return

    def make_features(self, ticker_index, ticker_record):
        'return Dict[feature_name, feature_value], err'
        date = ticker_record['effectivedate']
        isin = ticker_record['isin']  # international security identification number'
        try:
            weight = self.df.loc[date, isin]
            assert 0.0 <= weight <= 1.0, (weight, date, isin)
            return {
                    'p_weight_etf_%s_size' % self.short_name: weight,
                }, None
        except KeyError:
            msg = '.loc[date: %s, isin: %s] not in eff file %s' % (date, isin, self.name)
            return None, msg


class FeatureMakerFund(FeatureMaker):
    def __init__(self, df=None):
        super(FeatureMakerFund, self).__init__('fund')
        # precompute all the features
        # convert the Excel date in column "Date" to a python.datetime
        # create Dict[date: datetime.date, features: Dict[name:str, value]]
        self.sorted_df = df.sort_values('date', ascending=False)
        self.sorted_df.index = self.sorted_df['date']
        return

    def make_features(self, ticker_index, ticker_record):
        'return Dict[feature_name, feature_value], err'
        'return (True, Dict[feature_name, feature_value]) or (False, error_message)'
        # find correct date for the fundamentals
        # this code assumes there is no lag in reporting the values
        ticker_record_effective_date = ticker_record['effectivedate']
        for reported_date, row in self.sorted_df.iterrows():
            if reported_date <= ticker_record_effective_date:
                return {
                    'p_debt_to_market_cap': row['total_debt'] / row['mkt_cap'],
                    'p_debt_to_ltm_ebitda': row['total_debt'] / row['LTM_EBITDA'],
                    'p_gross_leverage': row['total_assets'] / row['total_debt'],
                    'p_interest_coverage': row['interest_coverage'],
                    'p_ltm_ebitda': row['LTM_EBITDA'],  # not a size, since could be negative
                    # OLD VERSION (note: different column names; some features were precomputed)
                    # 'p_debt_to_market_cap': series['Total Debt BS'] / series['Market capitalization'],
                    # 'p_debt_to_ltm_ebitda': series['Debt / LTM EBITDA'],
                    # 'p_gross_leverage': series['Debt / Mkt Cap'],
                    # 'p_interest_coverage': series['INTEREST_COVERAGE_RATIO'],
                    # 'p_ltm_ebitda_size': series['LTM EBITDA'],
                }, None
        msg = 'date %s not found in fundamentals files %s' % (ticker_record['effectivedate'], self.name)
        return None, msg


class FeatureMakerTradeId(FeatureMaker):
    def __init__(self, input_file_name=None):
        assert input_file_name is None
        super(FeatureMakerTradeId, self).__init__('trade id')

    def make_features(self, ticker_index, ticker_record):
        'return Dict[feature_name, feature_value], err'
        return {
            'id_ticker_index': ticker_index,
            'id_cusip': ticker_record['cusip'],
            'id_cusip1': ticker_record['cusip1'],
            'id_effectivedatetime': ticker_record['effectivedatetime'],
            'id_effectivedate': ticker_record['effectivedate'],
            'id_effectivetime': ticker_record['effectivetime'],
        }, None


def adjust_date(dates_list, days_back):
    'return date if its in valid_dates, else None'
    try:
        return dates_list[-days_back]
    except:
        return None


def ratio_day(prices_spx, prices_ticker, start, stop):
    delta_ticker = prices_ticker[start] * 1.0 / prices_ticker[stop]
    delta_spx = prices_spx[start] * 1.0 / prices_spx[stop]
    ratio_day = delta_ticker * 1.0 / delta_spx
    return ratio_day


class FeatureMakerOhlc(FeatureMaker):
    'ratio_days of delta ticker / delta spx for closing prices'
    def __init__(self, df_ticker=None, df_spx=None, verbose=False):
        'precompute all results'
        super(FeatureMakerOhlc, self).__init__('ohlc')

        self.df_ticker = df_ticker
        self.df_spx = df_spx
        self.skipped_reasons = collections.Counter()

        self.days_back = [
            1 + days_back
            for days_back in xrange(30)
        ]
        self.ratio_day = self._make_ratio_day(df_ticker, df_spx)

    def make_features(self, ticker_index, ticker_record):
        'return Dict[feature_name, feature_value], err'
        date = ticker_record.effectivedatetime.date()
        result = {}
        feature_name_template = 'p_price_delta_ratio_back_%s_%02d'
        for days_back in self.days_back:
            key = (date, days_back)
            if key not in self.ratio_day:
                return False, 'missing key %s in self.ratio_date' % key
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
                stop_date = adjust_date(dates_list, 2)
                start_date = adjust_date(dates_list, 2 + days_back)
                if stop_date is None or start_date is None:
                    msg = 'no valid date for trade date %s days_back %d' % (date, days_back)
                    if verbose:
                        print msg
                    self.skipped_reasons[msg] += 1
                    continue
                ratio[(date, days_back)] = ratio_day(
                    closing_price_spx,
                    closing_price_ticker,
                    start_date,
                    stop_date,
                )
                if verbose:
                    print 'ratio', days_back, start_date, stop_date, date, ratio[(date, days_back)]
        return ratio


def months_from_until(a, b):
    'return months from date a to date b'
    delta_days = (b - a).days
    return delta_days / 30.0


class FeatureMakerSecurityMaster(FeatureMaker):
    def __init__(self, df):
        super(FeatureMakerSecurityMaster, self).__init__('securitymaster')
        self.df = df

    def make_features(self, ticker_index, ticker_record):
        'return Dict[feature_name, feature_value], err'
        cusip = ticker_record['cusip']
        if cusip not in self.df.index:
            return 'cusip %s not in security master' % cusip
        row = self.df.loc[cusip]

        # check the we have coded all the discrete values that will occur
        def make_coupon_types(value):
            expected_values = ('Fixed rate', 'Floating rate')
            result = {}
            if value not in expected_values:
                print 'error: unexpected value:', value
                print 'expected values:', expected_values
                pdb.set_trace()
            result = {}
            for expected_value in expected_values:
                feature_name = 'p_coupon_type_is_%s' % expected_value.lower().replace(' ', '_')
                result[feature_name] = 1 if value == expected_value else 0
            return result

        assert row.curr_cpn >= 0.0
        assert row.is_callable in (False, True)
        assert row.is_puttable in (False, True)
        if not isinstance(row.issue_amount, numbers.Number):
            print 'not a number', row.issue_amount
            pdb.set_trace()
        if not row.issue_amount > 0:
            print 'not positive', row.issue_amount
            pdb.set_trace()

        result = {
            'p_amount_issued_size': row.issue_amount,
            'p_coupon_current_size': row.curr_cpn,
            'p_is_callable': 1 if row.is_callable else 0,
            'p_is_puttable': 1 if row.is_puttable else 0,
            'p_months_to_maturity_size': months_from_until(ticker_record.effectivedate, row.maturity_date)
        }
        result.update(make_coupon_types(row.coupon_type))
        return result, None


class TraceTradetypeContext(object):
    'accumulate running info from trace prints for each trade type'
    def __init__(self, lookback=None, typical_bid_offer=None, proximity_cutoff=None):
        self.order_imbalance4_object = OrderImbalance4.OrderImbalance4(
            lookback=lookback,
            typical_bid_offer=typical_bid_offer,
            proximity_cutoff=proximity_cutoff,
        )
        self.order_imbalance4 = None

        # these data members are part of the API
        # NOTE: they are for a specific trade type
        self.prior_oasspread = {}
        self.prior_price = {}
        self.prior_quantity = {}

        self.all_trade_types = ('B', 'D', 'S')

    def update(self, trace_record, verbose=False):
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
            return 'oasspread is NaN'
        if price != price:
            return 'price is NaN'
        if quantity != quantity:
            return 'quantity is NaN'

        self.order_imbalance4 = self.order_imbalance4_object.imbalance(
            trade_type=trade_type,
            trade_quantity=quantity,
            trade_price=price,
        )
        # the updated values could be missing, in which case, they are np.nan values
        self.prior_oasspread[trade_type] = oasspread
        self.prior_price[trade_type] = price
        self.prior_quantity[trade_type] = quantity
        return None

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


class TestTimeVolumeWeightedAverage(unittest.TestCase):
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


class VolumeWeightedAverage(object):
    def __init__(self, k):
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


class TestVolumeWeightedAverage(unittest.TestCase):
    def test(self):
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


class FeatureMakerTrace(FeatureMaker):
    def __init__(self, order_imbalance4_hps=None):
        super(FeatureMakerTrace, self).__init__('trace')
        assert order_imbalance4_hps is not None

        self.contexts = {}  # Dict[cusip, TraceTradetypeContext]
        self.order_imbalance4_hps = order_imbalance4_hps

        self.ks = (1, 2, 5, 10)  # num trades of weighted average spreads
        self.volume_weighted_average = {}
        self.time_volume_weighted_average = {}
        for k in self.ks:
                self.volume_weighted_average[k] = VolumeWeightedAverage(k)
                self.time_volume_weighted_average[k] = TimeVolumeWeightedAverage(k)

        self.last_trace_print_effectivedatetime = {}  # Dict[cusip, TimeStamp]

    def make_features(self, ticker_index, ticker_record):
        'return Dict[feature_name, feature_value], err'
        cusip = ticker_record['cusip']
        # interarrival time
        effectivedatetime = ticker_record['effectivedatetime']
        if cusip not in self.last_trace_print_effectivedatetime:
            self.last_trace_print_effectivedatetime[cusip] = effectivedatetime
            return None, 'no prior trace print'
        else:
            interval = effectivedatetime - self.last_trace_print_effectivedatetime[cusip]
            self.last_trace_print_effectivedatetime[cusip] = effectivedatetime
            # interval: Timedelta, a subclass of datetime.timedelta
            # attributes of a datetime.timedelta are days, seconds, microseconds
            interarrival_seconds = (interval.days * 24.0 * 60.0 * 60.0) + (interval.seconds * 1.0)

        # other info from prior trades of this cusip
        if cusip not in self.contexts:
            self.contexts[cusip] = TraceTradetypeContext(
                lookback=self.order_imbalance4_hps['lookback'],
                typical_bid_offer=self.order_imbalance4_hps['typical_bid_offer'],
                proximity_cutoff=self.order_imbalance4_hps['proximity_cutoff'],
            )
        cusip_context = self.contexts[cusip]
        err = cusip_context.update(ticker_record)
        if err is not None:
            return None, err
        err = cusip_context.missing_any_prior_oasspread()
        if err is not None:
            return None, err

        # weighted average spreads
        oasspread = ticker_record['oasspread']
        quantity = ticker_record['quantity']
        effectivedatetime = ticker_record['effectivedatetime']
        volume_weighted_average_spread = {}
        time_volume_weighted_average_spread = {}
        for k in self.ks:
            volume_weighted_spread, err = self.volume_weighted_average[k].weighted_average(
                oasspread,
                quantity,
            )
            if err is not None:
                return None, 'volume weighted average: ' + err
            volume_weighted_average_spread[k] = volume_weighted_spread

            time_volume_weighted_spread, err = self.time_volume_weighted_average[k].weighted_average(
                oasspread,
                quantity,
                effectivedatetime,
            )
            if err is not None:
                return None, 'time volume weighted average:' + err
            time_volume_weighted_average_spread[k] = time_volume_weighted_spread

        weighted_spread_features = {}
        for k in self.ks:
            feature_name = 'p_volume_weighted_oasspread_%d_back' % k
            weighted_spread_features[feature_name] = volume_weighted_average_spread[k]

            feature_name = 'p_time_volume_weighted_oasspread_%d_back' % k
            weighted_spread_features[feature_name] = time_volume_weighted_average_spread[k]

        # "prior" is the wrong adjective, because we use the values from the current trade
        # Cannot return price, spread, or related values for the current ticker record
        # These are the targets we are predicting
        other_features = {
            'p_interarrival_seconds': interarrival_seconds,
            'p_order_imbalance4': cusip_context.order_imbalance4,
            'p_prior_oasspread_B': cusip_context.prior_oasspread['B'],
            'p_prior_oasspread_D': cusip_context.prior_oasspread['D'],
            'p_prior_oasspread_S': cusip_context.prior_oasspread['S'],
            'p_prior_quantity_B_size': cusip_context.prior_quantity['B'],
            'p_prior_quantity_D_size': cusip_context.prior_quantity['D'],
            'p_prior_quantity_S_size': cusip_context.prior_quantity['S'],
            'p_quantity_size': ticker_record['quantity'],
            'p_trade_type_is_B': 1 if ticker_record['trade_type'] == 'B' else 0,
            'p_trade_type_is_D': 1 if ticker_record['trade_type'] == 'D' else 0,
            'p_trade_type_is_S': 1 if ticker_record['trade_type'] == 'S' else 0,
        }

        all_features = weighted_spread_features
        all_features.update(other_features)
        return all_features, None


if __name__ == '__main__':
    unittest.main()


if False:
    # avoid errors from linter
    pdb
    pprint
