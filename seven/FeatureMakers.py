'''class to create features from the input files
'''

from __future__ import division

import abc
import collections
import datetime
import numpy as np
import pdb

from seven.OrderImbalance4 import OrderImbalance4


class Skipped(object):
    def __init__(self):
        self.counter = collections.Counter()
        self.n_skipped = 0

    def skip(self, reason):
        self.counter[reason] += 1
        self.n_skipped += 1


class TickerContextCusip(object):
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


class FeatureMaker(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def make_features(cusip, input_record):
        'return None or Dict[feature_name: string, feature_value: number]'
        pass


def adjust_date(valid_dates, date):
    'return the date in valid_dates at date or just before it'
    earliest_date = min(valid_dates)
    for days_before in xrange(len(valid_dates)):
        candidate = date - datetime.timedelta(days_before)
        if candidate < earliest_date:
            return None
        if candidate in valid_dates:
            return candidate
    return None


def ratio(prices_spx, prices_ticker, start, stop):
    delta_ticker = prices_ticker[start] * 1.0 / prices_ticker[stop]
    delta_spx = prices_spx[start] * 1.0 / prices_spx[stop]
    ratio = delta_ticker * 1.0 / delta_spx
    return ratio


class FeatureMakerOhlc(FeatureMaker):
    'ratios of delta ticker / delta spx for closing prices'
    def __init__(self, df_ticker=None, df_spx=None, verbose=False):
        'precompute all results'
        self.skipped_reasons = collections.Counter()
        self.days_back = (1, 2, 3, 5, 7, 20, 28)
        closing_price_spx = {}     # Dict[date, closing_price]
        closing_price_ticker = {}
        self.ratio = {}            # Dict[(date, days_back), float]
        dates_seen = set()
        for timestamp in sorted(df_ticker.index):
            ticker = df_ticker.loc[timestamp]
            if timestamp.date() not in df_spx.index:
                msg = 'FeatureMakerOhlc: skipping index %s is in ticker but not spx' % timestamp
                self.skipped_reasons[msg] += 1
                continue
            spx = df_spx.loc[timestamp]

            date = timestamp.date()
            dates_seen.add(date)
            closing_price_spx[date] = spx.Close
            closing_price_ticker[date] = ticker.Close
            if verbose:
                print 'trades', date, 'ticker', ticker.Close, 'spx', spx.Close
            for days_back in self.days_back:
                # detemine for calendar days (which might fail)
                # assume that the calendar dates are a subset of the market dates
                stop_calendar_date = adjust_date(
                    dates_seen,
                    date - datetime.timedelta(1),
                )
                start_calendar_date = adjust_date(
                    dates_seen,
                    date - datetime.timedelta(1 + days_back),
                )
                if stop_calendar_date is None or start_calendar_date is None:
                    msg = 'no valid prior calendar date for trade date %s days_back %d' % (date, days_back)
                    self.skipped_reasons[msg] += 1
                    continue
                # ratio for calendar date
                self.ratio[(date, days_back)] = ratio(
                    closing_price_spx,
                    closing_price_ticker,
                    start_calendar_date,
                    stop_calendar_date,
                )
                if verbose:
                    print 'feature', days_back, start_calendar_date, stop_calendar_date, self.ratio[(date, days_back)]

    def make_features(self, cusip, ticker_record):
        'return Dict[feature_name: str, feature_value: number] or None'
        date = ticker_record.effectivedatetime.date()
        result = {}
        for days_back in self.days_back:
            key = (date, days_back)
            if key not in self.ratio:
                return None
            feature_name = 'price_delta_ratio_back_%02d' % days_back
            result[feature_name] = self.ratio[key]
        return result


class FeatureMakerTicker(FeatureMaker):
    def __init__(self, order_imbalance4_hps=None):
        assert order_imbalance4_hps is not None
        self.order_imbalance4_hps = order_imbalance4_hps

        self.input_file = 'ticker'
        self.contexts = {}
        self.skipped_reasons = collections.Counter()
        self.count_created = 0

    def make_features(self, cusip, ticker):
        'return Dict[feature_name: string, feature_value: number] or None'
        if cusip not in self.contexts:
            self.contexts[cusip] = TickerContextCusip(
                lookback=self.order_imbalance4_hps['lookback'],
                typical_bid_offer=self.order_imbalance4_hps['typical_bid_offer'],
                proximity_cutoff=self.order_imbalance4_hps['proximity_cutoff'],
            )
        cusip_context = self.contexts[cusip]
        cusip_context.update(ticker)
        if cusip_context.missing_any_historic_oasspread():
            self.skipped_reasons['missing any historic oasspread'] += 1
            return None
        elif np.isnan(ticker.oasspread):
            self.skipped_reasons['missing ticker.oasspread'] += 1
            return None
        else:
            self.count_created += 1
            return {
                'oasspread': ticker.oasspread,
                'order_imbalance4': cusip_context.order_imbalance4,
                'prior_oasspread_B': cusip_context.prior_oasspread_B,
                'prior_oasspread_D': cusip_context.prior_oasspread_D,
                'prior_oasspread_s': cusip_context.prior_oasspread_S,
                'prior_quantity_B': cusip_context.prior_quantity_B,
                'prior_quantity_D': cusip_context.prior_quantity_D,
                'prior_quantity_s': cusip_context.prior_quantity_S,
                'quantity': ticker.quantity,
                'trade_type_is_B': 1 if ticker.trade_type == 'B' else 0,
                'trade_type_is_D': 1 if ticker.trade_type == 'D' else 0,
                'trade_type_is_S': 1 if ticker.trade_type == 'S' else 0,
            }