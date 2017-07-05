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

from abc import ABCMeta, abstractmethod
import collections
import copy
import datetime
import math
import pandas as pd
import pdb
from pprint import pprint
import unittest

# imports from seven/
import Fundamentals
import InterarrivalTime
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
    def __init__(self, issuer, cusips):
        'create a feature maker for each CUSIP'
        self.issuer = issuer
        self.cusipd = cusips

        self.last_effectivedatetime = None

        # Dict[cusip, OrderedDict[trace_index, Dict[feature, value]]  about 50 features
        self.primary_features = collections.defaultdict(lambda: collections.OrderedDict())
        # Dict[cusip, Dict[trace_index_of_cusip, (otr_cusip, index_in_otr_cusip)]]
        self.otr_index = collections.defaultdict(dict)

        self.features_ticker_cusip = {}
        for cusip in cusips:
            self.features_ticker_cusip[cusip] = FeaturesTickerCusip(issuer, cusip)

    def append_features(self, trace_index, trace_record):
        'return None or error_message:str; mutate self.features_ticker_cusip[cusip]'
        # assure that datetimes are in non-decreasing order
        effectivedatetime = trace_record['effectivedatetime']
        if self.last_effectivedatetime is not None:
            assert self.last_effectivedatetime <= effectivedatetime
        self.last_effectivedatetime = effectivedatetime

        cusip = trace_record['cusip']
        if not (cusip == self.primary_cusip or cusip in self.otr_cusips):
            return 'trace record cusip not the primary cusip and not in otr cusips'
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
        # NOTE: mis-named, as it returns all of the features, not just the primary features
        # deprecated: used only by fit_predict.py
        def append_features(data, feature_values, check_for_nans=True):
            'append primary and OTR features to the data'
            for feature_name, feature_value in features_values.iteritems():
                data[feature_name].append(feature_value)
            otr_index_dict = self.otr_index[self.primary_cusip]
            otr_cusip, index_in_otr_cusip = otr_index_dict[trace_index]
            otr_features_values = self.primary_features[otr_cusip][index_in_otr_cusip]
            for feature_name, feature_value in otr_features_values.iteritems():
                if feature_name.startswith('p_'):
                    new_feature_name = 'otr1_' + feature_name[2:]
                    if check_for_nans:
                        if feature_value != feature_value:
                            print 'found NaN feature value', feature_value, feature_name
                            pdb.set_trace()
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

    def get_dataframe(self):
        'return DataFrame with primary and OTR features'
        return self.get_primary_features_dataframe()


class FeaturesAccumulator(object):
    'accumulate rows in the DataFrame self.features contaning the features derived from trace prints'
    def __init__(self, issuer, cusip):
        def make_FeatureMakerEtfCusip(logical_name):
            'return a constructured FeatureMakerEtf'
            # this code works for the weight cusip * logical file names
            # It does not work for issue and sector ETFs weights, because the columns names are
            # different for them.
            # - weight issuer * has column "ticker" (containing either the ticker or corporate name), not "cusip"
            # - weight sector * has column "sector", not column "cusip" (as in weight cusip *)
            # Perhaps the way to handle this to either
            # - subcclass FeatureMakerEtf with 3 subclasses (this is cleanest)
            # - pass an additional construction parameter to FeatureMakerEtf (a hack)
            pdb.set_trace()
            print 'change me to accomdate new file names and new file columns'
            print 'upstream team to remove illrelevant CUSIP'
            print 'so the file wil be smaller'
            print 'maybe: change logic to read the file just once, not once for each cusip'
            print ''
            print 'file name eft_weight_of_cusip_pct_agg_recent.csv' # with last 120 days of info for cusips of interest
            df_raw = read_csv.input(issuer=None, logical_name=logical_name)  # ETF weights for the cusip
            # since the FeatureAccumulator is for a cusip, we can discard all non-cusip info from the input
            # that will speed up processing
            df = df_raw.reset_index()  # put the index columns into the actual data frame
            mask = df['cusip'] == cusip
            df_cusip = df.loc[mask]
            return FeatureMakerEtfCusip(
                df=df_cusip,
                name=logical_name,
            )

        self.issuer = issuer
        self.cusip = cusip
        self.all_feature_makers = (
            # make_FeatureMakerEtfCusip('weight cusip agg'),
            # make_FeatureMakerEtfCusip('weight cusip lqd'),
            # make_FeatureMakerEtf('weight issuer agg'),
            # make_FeatureMakerEtf('weight issuer lqd'),
            # make_FeatureMakerEtf('weight sector agg'),
            # make_FeatureMakerEtf('weight sector lqd'),
            FeatureMakerFundamentals(self.issuer),
            FeatureMakerTradeId(),
            # FeatureMakerOhlc(
            #     df_ticker=read_csv.input(self.issuer, 'ohlc ticker'),
            #     df_spx=read_csv.input(self.issuer, 'ohlc spx'),
            # ),
            FeatureMakerSecurityMaster(
                df=read_csv.input(issuer=None, logical_name='security master'),
            ),
            FeatureMakerTrace(
                order_imbalance4_hps={
                    'lookback': 10,
                    'typical_bid_offer': 2,
                    'proximity_cutoff': 20,
                },
            ),
            # TODO: add convexity feature (file to come) in trace print file
        )
        # NOTE: callers rely on the definitions of skipped and features
        self.skipped = collections.Counter()
        self.features = pd.DataFrame()  # The API guarantees this feature

    def __str__(self):
        return 'FeaturesTickerCusip(issuer=%s,cusip%s)' % (self.issuer, self.cusip)

    def __deepcopy__(self, memo_dict):
        'return new AllFeatures instance'
        # ref: http://stackoverflow.com/questions/15214404/how-can-i-copy-an-immutable-object-like-tuple-in-python
        other = AllFeatures(self.ticker, self.cusip)
        other.all_feature_makers = copy.deepcopy(self.all_feature_makers, memo_dict)
        other.skipped = copy.deepcopy(self.skipped, memo_dict)
        other.features = copy.deepcopy(self.features, memo_dict)
        return other

    def append_features(self, trace_index, trace_record, verbose=False):
        'Append a new row of features to self.features:DataFrame and return None, or return an iterable of errors'
        # assure all the trace_records are for the same CUSIP
        # if not, the caller has goofed
        assert self.cusip == trace_record['cusip']

        # accumulate features from the feature makers
        # stop on the first error from a feature maker
        all_features = {}  # Dict[feature_name, feature_value]
        all_errors = []
        for feature_maker in self.all_feature_makers:
            if verbose:
                print feature_maker.name
            features, errors = feature_maker.make_features(trace_index, trace_record)
            if errors is not None:
                if isinstance(errors, str):
                    all_errors.append(errors)
                else:
                    all_errors.extend(errors)
            else:
                # make sure that the features_makers have not created a duplicate feature
                for k, v in features.iteritems():
                    if k in all_features:
                        print 'internal error: duplicate feature', k, feature_maker.name
                        pdb.set_trace()
                    all_features[k] = v

        if len(all_errors) > 0:
            return all_errors

        df = pd.DataFrame(
            data=all_features,
            index=pd.Index(
                data=[trace_index],
                name='trace_index',
            )
        )
        self.features = self.features.append(df)  # the API guarantees this dataframe
        return None


class FeatureMaker(object):
    __metaclass__ = ABCMeta

    def __init__(self, name=None):
        print 'constructing FeatureMaker', name
        self.name = name  # used in error message; informal name of the feature maker

    @abstractmethod
    def make_features(trace_index, tickercusip, trace_record):
        'return error:str or List[errors:str] or Dict[feature_name:str, feature_value:nmber]'
        pass


class FeatureMakerEtf(FeatureMaker):
    def __init__(self, df=None, name=None):
        'construct'
        # name is "weight {security_kind} {etf_name}"
        assert df is not None
        assert name is not None
        super(FeatureMakerEtf, self).__init__('etf ' + name)  # sets self.name
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


class FeatureMakerEtfCusip(FeatureMakerEtf):
    def __init__(self, df=None, name=None):
        print 'construction FeatureMakerEtfCusip'
        super(FeatureMakerEtfCusip, self). __init__(df=df, name=name)
        self.second_index_name = 'cusip'


class TestFeatureMakerEtfCusip(unittest.TestCase):
    def test(self):
        Test = collections.namedtuple('Test', 'logical_name cusip date, expected_featurename, expected_weight')
        tests = (
            Test('weight cusip agg', '00184AAG0', datetime.date(2010, 1, 29), 'p_weight_etf_pct_cusip_agg_size', 0.152722039314371),
        )
        for test in tests:
            fm = FeatureMakerEtfCusip(
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


class FeatureMakerFundamentals(FeatureMaker):
    def __init__(self, issuer):
        super(FeatureMakerFundamentals, self).__init__('fundamentals ' + issuer)
        self.fundamentals = Fundamentals.Fundamentals(issuer=issuer)
        self.base_feature_names = self.fundamentals.logical_names()
        return

    def make_features(self, trace_index, trace_record):
        'return Dict[feature_name, feature_value], err'
        'return (True, Dict[feature_name, feature_value]) or (False, error_message)'
        def check_no_negatives(d):
            for k, v in d.iteritems():
                assert v >= 0.0

        debug = True
        date = trace_record['effectivedate'].date()

        # build features directly from the fundamentals file data
        result = {}
        errors = []
        for base_feature_name in self.base_feature_names:
            value, err = self.fundamentals.get(date, base_feature_name)
            if err is not None:
                errors.append(err)
            else:
                result['p_%s_size' % base_feature_name] = value
        # add in derived features

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

        maybe_add_ratio('p_debt_to_market_cap_size', 'p_total_debt_size', 'p_mkt_cap_size')
        maybe_add_ratio('p_debt_to_ltm_ebitda_size', 'p_total_debt_size', 'p_LTM_EBITDA_size')

        check_no_negatives(result)
        if len(errors) == 0:
            return result, None
        else:
            if debug:
                print 'errors found by FeatureMakerFundamentals %s' % self.name
                for error in errors:
                    print error
                pdb.set_trace()
                return None, errors


class FeatureMakerTradeId(FeatureMaker):
    def __init__(self, input_file_name=None):
        assert input_file_name is None
        super(FeatureMakerTradeId, self).__init__('trade id')

    def make_features(self, trace_index, trace_record):
        'return Dict[feature_name, feature_value], err'
        return {
            'id_trace_index': trace_index,
            'id_cusip': trace_record['cusip'],
            'id_effectivedatetime': trace_record['effectivedatetime'],
            'id_effectivedate': trace_record['effectivedate'],
            'id_effectivetime': trace_record['effectivetime'],
            'id_trade_type': trace_record['trade_type'],
            'id_issuepriceid': trace_record['issuepriceid'],  # unique identifier of the trace print
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

        self.df_ticker = df_ticker  # a DataFreame
        self.df_spx = df_spx        # a DataFrame
        self.skipped_reasons = collections.Counter()

        self.days_back = [
            1 + days_back
            for days_back in xrange(30)
        ]
        self.ratio_day, self.dates_list = self._make_ratio_day(df_ticker, df_spx)
        self.dates_set = set(self.dates_list)

    def make_features(self, trace_index, trace_record):
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
        return ratio, dates_list


def months_from_until(a, b):
    'return months from date a to date b'
    delta_days = (b - a).days
    return delta_days / 30.0


class FeatureMakerSecurityMaster(FeatureMaker):
    def __init__(self, df):
        super(FeatureMakerSecurityMaster, self).__init__('securitymaster')
        self.df = df  # the security master records

    def make_features(self, trace_index, trace_record):
        'return (Dict[feature_name, feature_value], None) or (None, err)'
        cusip = trace_record['cusip']
        if cusip not in self.df.index:
            return (None, 'cusip %s not in security master' % cusip)
        security = self.df.loc[cusip]

        result = {
            'p_coupon_type_is_fixed_rate': 1 if security['coupon_type'] == 'Fixed rate' else 0,
            'p_coupon_type_is_floating_rate': 1 if security['coupon_type'] == 'Floating rate' else 0,
            'p_original_amount_issued_size': security['original_amount_issued'],
            'p_months_to_maturity_size': months_from_until(trace_record['effectivedate'], security['maturity_date']),
            'p_months_of_life_size': months_from_until(security['issue_date'], security['maturity_date']),
            'p_is_callable': 1 if security['is_callable'] else 0,
            'p_is_puttable': 1 if security['is_puttable'] else 0,
        }
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

        self.interarrivaltimes = collections.defaultdict(lambda: InterarrivalTime.InterarrivalTime())  # key = cusip

    def make_features(self, trace_index, trace_record):
        'return Dict[feature_name, feature_value], err'
        cusip = trace_record['cusip']

        # interarrival time are the time difference since the last trace print for the same cusip
        interarrival_seconds, err = self.interarrivaltimes[cusip].interarrival_seconds(trace_index, trace_record)
        if err is not None:
            return (None, 'interarrivaltime: %s' % err)

        # other info from prior trades of this cusip
        if cusip not in self.contexts:
            self.contexts[cusip] = TraceTradetypeContext(
                lookback=self.order_imbalance4_hps['lookback'],
                typical_bid_offer=self.order_imbalance4_hps['typical_bid_offer'],
                proximity_cutoff=self.order_imbalance4_hps['proximity_cutoff'],
            )
        cusip_context = self.contexts[cusip]
        err = cusip_context.update(trace_record)
        if err is not None:
            return None, err
        err = cusip_context.missing_any_prior_oasspread()
        if err is not None:
            return None, 'TraceTradetypeContext: ' + err

        # weighted average spreads
        oasspread = trace_record['oasspread']
        quantity = trace_record['quantity']
        effectivedatetime = trace_record['effectivedatetime']
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
                return None, 'time volume weighted average: ' + err
            time_volume_weighted_average_spread[k] = time_volume_weighted_spread

        weighted_spread_features = {}
        for k in self.ks:
            feature_name = 'p_volume_weighted_oasspread_%02d_back' % k
            weighted_spread_features[feature_name] = volume_weighted_average_spread[k]

            feature_name = 'p_time_volume_weighted_oasspread_%02d_back' % k
            weighted_spread_features[feature_name] = time_volume_weighted_average_spread[k]

        other_features = {
            'p_interarrival_seconds': interarrival_seconds,
            'p_order_imbalance4': cusip_context.order_imbalance4,
            'p_prior_oasspread_B': cusip_context.prior_oasspread['B'],
            'p_prior_oasspread_D': cusip_context.prior_oasspread['D'],
            'p_prior_oasspread_S': cusip_context.prior_oasspread['S'],
            'p_prior_quantity_B_size': cusip_context.prior_quantity['B'],
            'p_prior_quantity_D_size': cusip_context.prior_quantity['D'],
            'p_prior_quantity_S_size': cusip_context.prior_quantity['S'],
            'p_quantity_size': trace_record['quantity'],
            'p_trade_type_is_B': 1 if trace_record['trade_type'] == 'B' else 0,
            'p_trade_type_is_D': 1 if trace_record['trade_type'] == 'D' else 0,
            'p_trade_type_is_S': 1 if trace_record['trade_type'] == 'S' else 0,
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
