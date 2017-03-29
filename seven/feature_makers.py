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
'''

from __future__ import division

import abc
import collections
import datetime
import numpy as np
import pdb
from pprint import pprint
from xlrd.xldate import xldate_as_tuple


from applied_data_science.timeseries import FeatureMaker

from seven.OrderImbalance4 import OrderImbalance4


class Skipped(object):
    def __init__(self):
        self.counter = collections.Counter()
        self.n_skipped = 0

    def skip(self, reason):
        self.counter[reason] += 1
        self.n_skipped += 1


class TickertickerContextCusip(object):
    'accumulate running info for trades for a specific CUSIP'
    def __init__(self, lookback=None, typical_bid_offer=None, proximity_cutoff=None):
        self.order_imbalance4_object = OrderImbalance4(
            lookback=lookback,
            typical_bid_offer=typical_bid_offer,
            proximity_cutoff=proximity_cutoff,
        )
        self.order_imbalance4 = None

        self.prior_oasspread_B = np.nan
        self.prior_oasspread_D = np.nan
        self.prior_oasspread_S = np.nan

        self.prior_price_B = np.nan
        self.prior_price_D = np.nan
        self.prior_price_S = np.nan

        self.prior_quantity_B = np.nan
        self.prior_quantity_D = np.nan
        self.prior_quantity_S = np.nan

    def update(self, trade):
        'update self.* using current trade'
        self.order_imbalance4 = self.order_imbalance4_object.imbalance(
            trade_type=trade.trade_type,
            trade_quantity=trade.quantity,
            trade_price=trade.price,
        )
        # the updated values could be missing, in which case, they are np.nan values
        if trade.trade_type == 'B':
            self.prior_oasspread_B = trade.oasspread
            self.prior_price_B = trade.price
            self.prior_quantity_B = trade.quantity
        elif trade.trade_type == 'D':
            self.prior_oasspread_D = trade.oasspread
            self.prior_price_D = trade.price
            self.prior_quantity_D = trade.quantity
        elif trade.trade_type == 'S':
            self.prior_oasspread_S = trade.oasspread
            self.prior_price_S = trade.price
            self.prior_quantity_S = trade.quantity
        else:
            print trade
            print 'unknown trade_type', trade.trade_type
            pdb.set_trace()

    def missing_any_historic_oasspread(self):
        return (
            np.isnan(self.prior_oasspread_B) or
            np.isnan(self.prior_oasspread_D) or
            np.isnan(self.prior_oasspread_S)
        )


class FeatureMakerFund(FeatureMaker):
    def __init__(self, df_fund=None):
        super(FeatureMakerFund, self).__init__('fund')
        # precompute all the features
        # convert the Excel date in column "Date" to a python.datetime
        features = {}  # Dict[python_date, feature_dict]
        excel_date_mode = 0  # 1900 based
        for i, excel_date in enumerate(df_fund['Date']):
            python_date_tuple = xldate_as_tuple(excel_date, excel_date_mode)
            # make sure time portion of date is zero
            assert python_date_tuple[3] == 0
            assert python_date_tuple[4] == 0
            assert python_date_tuple[5] == 0
            python_date = datetime.date(*python_date_tuple[:3])
            feature_dict = self._make_feature_dict(df_fund.iloc[i])
            features[python_date] = feature_dict
        self.features = features

    def make_features(self, ticker_index, ticker_record):
        'return error_msg:str or Dict[feature_name, feature_value]'
        ticker_date = ticker_record.effectivedatetime.date()
        return self.features.get(ticker_date, None)

    def _make_feature_dict(self, series):
        'return dictionary of features'
        return {
            'debt_to_market_cap': series['Total Debt BS'] / series['Market capitalization'],
            'debt_to_ltm_ebitda': series['Debt / LTM EBITDA'],
            'gross_leverage': series['Debt / Mkt Cap'],
            'interest_coverage': series['INTEREST_COVERAGE_RATIO'],
            'ltm_ebitda_size': series['LTM EBITDA'] 
        }


class FeatureMakerTradeId(FeatureMaker):
    def __init__(self, input_file_name=None):
        assert input_file_name is None
        super(FeatureMakerTradeId, self).__init__('trade id')

    def make_features(self, ticker_index, ticker_record):
        'return error_msg:str or Dict[feature_name, feature_value]'
        return {
            'id_ticker_index': ticker_index,
            'id_cusip': ticker_record.cusip,
            'id_effectivedatetime': ticker_record.effectivedatetime
        }


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
        'return Dict[feature_name: str, feature_value: number] or error_msg:str'
        date = ticker_record.effectivedatetime.date()
        result = {}
        feature_name_template = 'price_delta_ratio_back_%s_%02d'
        for days_back in self.days_back:
            key = (date, days_back)
            if key not in self.ratio_day:
                return 'missing key %s in self.ratio_date' % key
            feature_name = feature_name_template % ('days', days_back)
            result[feature_name] = self.ratio_day[key]
        return result

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
        'return Dict[feature_name: str, feature_value: number] or error_msg:str'
        cusip = ticker_record.cusip
        if cusip not in self.df.index:
            return 'error: cusip %s not in security master file' % cusip
        row = self.df.loc[cusip]
        # check the we have coded all the discrete values that will occur
        assert row.COLLAT_TYP in ('SR UNSECURED',)
        assert row.CPN_TYP in ('FIXED', 'FLOATING',)
        return {
            'amount_issued_size': row.issue_amount,
            'collateral_type_is_sr_unsecured': row.COLLAT_TYP == 'SR UNSECURED',
            'coupon_is_fixed': row.CPN_TYP == 'FIXED',
            'coupon_is_floating': row.CPN_TYP == 'FLOATING',
            'coupon_current_size': row.curr_cpn,
            'is_callable': row.is_callable == 'TRUE',
            'months_to_maturity_size': months_from_until(ticker_record.effectivedate, row.maturity_date)
        }


class FeatureMakerTicker(FeatureMaker):
    def __init__(self, order_imbalance4_hps=None):
        super(FeatureMakerTicker, self).__init__('ticker')
        assert order_imbalance4_hps is not None

        self.contexts = {}  # Dict[cusip, TickertickerContextCusip]
        self.order_imbalance4_hps = order_imbalance4_hps

    def make_features(self, ticker_index, ticker_record):
        'return Dict[feature_name: string, feature_value: number] or error:str'
        cusip = ticker_record.cusip
        if cusip not in self.contexts:
            self.contexts[cusip] = TickertickerContextCusip(
                lookback=self.order_imbalance4_hps['lookback'],
                typical_bid_offer=self.order_imbalance4_hps['typical_bid_offer'],
                proximity_cutoff=self.order_imbalance4_hps['proximity_cutoff'],
            )
        cusip_context = self.contexts[cusip]
        cusip_context.update(ticker_record)
        if cusip_context.missing_any_historic_oasspread():
            return 'missing any historic oasspread'
        elif np.isnan(ticker_record.oasspread):
            return 'ticker_record.oasspread is NaN (missing)'
        else:
            return {
                'oasspread_size': ticker_record.oasspread,
                'order_imbalance4': cusip_context.order_imbalance4,
                'oasspread_B_size': cusip_context.prior_oasspread_B,
                'oasspread_D_size': cusip_context.prior_oasspread_D,
                'oasspread_S_size': cusip_context.prior_oasspread_S,
                'quantity_B_size': cusip_context.prior_quantity_B,
                'quantity_D_size': cusip_context.prior_quantity_D,
                'quantity_S_size': cusip_context.prior_quantity_S,
                'quantity_size': ticker_record.quantity,
                'trade_type_is_B': 1 if ticker_record.trade_type == 'B' else 0,
                'trade_type_is_D': 1 if ticker_record.trade_type == 'D' else 0,
                'trade_type_is_S': 1 if ticker_record.trade_type == 'S' else 0,
            }


if False:
    # avoid errors from linter
    pdb
    pprint
