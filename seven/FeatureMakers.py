'''class to create features from the input files
'''

from __future__ import division

import abc
import collections
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


class DeltaPrice(object):
    'ratios of delta ticker / delta spx for closing prices'
    def __init__(self, df_ticker=None, df_spx=None):
        'precompute all results'
        test = False
        self.days_back = (1, 2, 3, 5, 20)
        self.available_dates = set()
        history_len = 1 + 20  # 1 plus max lookback period (1 month == 20 trade days, ignoring holidays)
        closes_ticker = collections.deque([], history_len)
        closes_spx = collections.deque([], history_len)
        self.ratios = {}  # Dict[(date, days_back), ratio:float]
        for timestamp in sorted(df_ticker.index):
            ticker = df_ticker.loc[timestamp]
            if timestamp.date() not in df_spx.index:
                print 'skipping index %s is in ticker but not spx' % timestamp
                continue
            spx = df_spx.loc[timestamp]
            if test:
                print timestamp
                print ticker
                print spx
            closes_ticker.append(ticker.Close)
            closes_spx.append(spx.Close)
            if len(closes_ticker) >= history_len:
                # constraint is to be able to determine equity_price_delta_close_prior_month
                # CHECK: OK to define week at 5 days ago and month as 20 days ago
                date = timestamp.date()
                assert date not in self.available_dates
                self.available_dates.add(date)
                for days_back in self.days_back:
                    day_index = 1 + days_back
                    delta_ticker = closes_ticker[-day_index] * 1.0 / ticker.Close
                    delta_spx = closes_spx[-day_index] * 1.0 / spx.Close
                    ratio = delta_ticker / delta_spx
                    self.ratios[(date, days_back)] = ratio
        if test:
            count = 0
            for date in self.available_dates:
                for days_back in self.days_back:
                    print date, days_back, self.ratio(date, days_back)
                count += 1
                if count > 2:
                    break

    def is_available(self, date):
        return date in self.available_dates

    def ratio(self, date, days_back):
        assert date in self.available_dates
        assert days_back in self.days_back
        return self.ratios[(date, days_back)]


class FeatureMaker(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def make_features(cusip, input_record):
        'return None or Dict[feature_name: string, feature_value: float]'
        pass


class FeatureMakerTicker(FeatureMaker):
    def __init__(self, order_imbalance4_hps=None):
        assert order_imbalance4_hps is not None
        self.order_imbalance4_hps = order_imbalance4_hps

        self.input_file = 'ticker'
        self.contexts = {}
        self.skipped_reasons = collections.Counter()
        self.count_created = 0

    def make_features(self, cusip, ticker):
        'return Dict[feature_name: string, feature_value: float] or None'
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
