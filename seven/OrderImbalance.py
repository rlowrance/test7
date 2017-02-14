from __future__ import division

import pandas as pd
import pdb
import unittest

from MaybeNumber import MaybeNumber
import path
from Windowed import Window


class OrderImbalance(object):
    def __init__(self, lookback, typical_bid_offer_spread):
        self.lookback = lookback  # in number of trades
        self.typical_bid_offer_spread = typical_bid_offer_spread
        self.prior_bid_price = MaybeNumber(None)
        self.prior_offer_price = MaybeNumber(None)
        self.synthetic_mid_price = MaybeNumber(None)
        self.buy_volume_rolling = Windowed()
        self.sell_volumn_rolling = Windowed()


    def create_order_imbalance(self, lookback_value, typical_bid_offer_value):
        '''return ? where
        lookback_value: int -- number of trades of specified type (dealer_buy, dealer_dealer, dealer_sell)
        typical_bid_offer_value: float -- typical spread
        '''

        lookback = MaybeNumber(lookback_value)
        typical_bid_offer = MaybeNumber(typical_bid_offer_value)
        typical_bid_offer_half = typical_bid_offer / MaybeNumber(2.0)

        def make_prior_bid_price(prior_offer_price, synthetic_mid_price):
            return (
                synthetic_mid_price - typical_bid_offer_half if prior_offer_price.value is None else
                prior_offer_price - typical_bid_offer
            )

        def make_prior_offer_price(prior_bid_price, synthetic_mid_price):
            return (
                synthetic_mid_price + typical_bid_offer_half if prior_bid_price.value is None else
                prior_offer_price + typical_bid_offer
            )

        pdb.set_trace()
        result = pd.DataFrame()
        pdb.set_trace()
        prior_bid_price = MaybeNumber(None)
        prior_offer_price = MaybeNumber(None)
        synthetic_mid_price = MaybeNumber(None)
        open_interest = MaybeNumber(None)
        rolling_sum_bid = RollingSummer(lookback_value)
        rolling_sum_offer = RollingSummer(lookback_value)
        buy_volume = 0
        sell_volumn = 0
        for index, trade in self.trace_history.iterrows():
            print index, trade
            quantity = trade['quantity']
            # only one of these 3 X_price variables has a value
            buy_price = MaybeNumber(trade['buy_price'])
            dlr_price = MaybeNumber(trade['dlr_price'])
            sell_price = MaybeNumber(trade['sell_price'])
            print quantity, buy_price, dlr_price, sell_price
            # TODO: what are missing values
            if prior_bid_price.value is None:
                prior_bid_price = make_prior_bid_price(prior_offer_price, synthetic_mid_price)
            if prior_offer_price.value is None:
                prior_offer_price = make_prior_offer_price(prior_bid_price, synthetic_mid_price)
            synthetic_mid_price = prior_bid_price.mean(prior_offer_price)
            # update rolling sum
            rolling_sum_bid.add(buy_price)
            rolling_sum_offer.add(sell_price)
            open_interest = rolling_sum_bid.volume() - rolling_sum_offer.volume()
            if dlr_price.value is not None:
                # its a dealer-to-dealer trade
                if dlr_price > synthetic_mid_price or abs(dlr_price - prior_offer_price ) < abs(dlr_price - prior_sell_price):
                    offer_volume += quantity
                else:
                    buy_volume += quantity

            # setup next iteration
            prior_bid_price = prior_bid_price if buy_price.value is None else buy_price
            prior_offer_price = prior_offer_price if sell_price.value is None else sell_price
        pdb.set_trace()
        return result


class TestOrderImbalance(unittest.TestCase):
    def test_create_orer_imbalance(self):
        return
        pdb.set_trace()
        df = pd.read_csv(path.orcl_sample1_csv())
        oi = OrderImbalance(df)
        x = oi.create_order_imbalance(1, 2)
        print x


if __name__ == '__main__':
    unittest.main()
