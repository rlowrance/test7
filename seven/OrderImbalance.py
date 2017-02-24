'OrderImbalance abstract class definition'

from __future__ import division

from abc import ABCMeta, abstractmethod


class OrderImbalance(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def imbalance(self, trade_type=None, trade_quantity=None, trade_price=None, verbose=False):
        'return the order imbalance after the trade, a number or None'
        pass
