'''define accumulators

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a license.
'''
from abc import ABCMeta, abstractmethod
import collections
import numbers
import numpy as np
import pandas as pd
import pdb
import pprint

import feature_makers
import read_csv

pp = pprint.pprint

class Accumulator(object):
    __metaclass__ = ABCMeta

    def __init__(self, name=None):
        assert name is not None
        # these attributes are part of the API
        self.name = name  # used in error message; informal name of the accumulator
        self.accumulated = pd.DataFrame()  # this attribute is always available

    @abstractmethod
    def accumulate(self, trace_index, trace_record):
        'return None (the info was accumulated) or List[str] (the errors occured and info was not accumulated)'
        pass


class FeaturesAccumulator(Accumulator):
    'accumulate features for a specific cusip'
    def __init__(self, issuer, cusip):
        super(FeaturesAccumulator, self).__init__('FeatureAccumulator(issuer=%s, cusip=%s)' % (
            issuer,
            cusip))

        self.issuer = issuer
        self.cusip = cusip
        history_length = 10
        self.all_feature_makers = (
            # TODO: re-implement these (the file format changed)
            # They were commented out when the file format changed
            # EtfCusip('weight cusip agg'),
            # EtfCusip('weight cusip lqd'),
            # Etf('weight issuer agg'),
            # Etf('weight issuer lqd'),
            # Etf('weight sector agg'),
            # Etf('weight sector lqd'),
            # Ohlc(
            #     df_ticker=read_csv.input(self.issuer, 'ohlc ticker'),
            #     df_spx=read_csv.input(self.issuer, 'ohlc spx'),
            # ),
            feature_makers.OrderImbalance(  # this one must be first (it creates the reclassified trade type)
                lookback=10,
                typical_bid_offer=2,
                proximity_cutoff=20,
            ),
            feature_makers.Fundamentals(self.issuer),
            feature_makers.InterarrivalTime(),
            feature_makers.SecurityMaster(
                df=read_csv.input(issuer=None, logical_name='security master'),
            ),
            feature_makers.HistoryOasspread(history_length=history_length),
            feature_makers.HistoryPrice(history_length=history_length),
            feature_makers.HistoryQuantity(history_length=history_length),
            feature_makers.HistoryQuantityWeightedAverageSpread(history_length=2),
            feature_makers.HistoryQuantityWeightedAverageSpread(history_length=3),
            feature_makers.HistoryQuantityWeightedAverageSpread(history_length=4),
            feature_makers.HistoryQuantityWeightedAverageSpread(history_length=5),
            feature_makers.HistoryQuantityWeightedAverageSpread(history_length=6),
            feature_makers.HistoryQuantityWeightedAverageSpread(history_length=7),
            feature_makers.HistoryQuantityWeightedAverageSpread(history_length=8),
            feature_makers.HistoryQuantityWeightedAverageSpread(history_length=9),
            feature_makers.HistoryQuantityWeightedAverageSpread(history_length=10),
            feature_makers.PriorTraceRecord(),
            feature_makers.TraceRecord(),
        )

    def accumulate(self, trace_index, trace_record, verbose=True, debug=False):
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
                print 'FeatureAccumulator.accumulate:',feature_maker.name
            features, errors = feature_maker.make_features(trace_index, trace_record, all_features)
            if errors is not None:
                if isinstance(errors, str):
                    all_errors.append(errors)
                else:
                    all_errors.extend(errors)
            else:
                # make sure that the features_makers have not created a duplicate feature
                # make sure that every feature is numeric
                # accumulate features into all_features
                for k, v in features.iteritems():
                    if k in all_features:
                        print 'internal error: duplicate feature', k, feature_maker.name
                        pdb.set_trace()
                    if k.startswith('p_'):
                        if not isinstance(v, numbers.Number):
                            print 'internal error: feature %s=%s is not numeric, but must be' % (k, v)
                            assert isinstance(v, numbers.Number)
                    all_features[k] = v

        if len(all_errors) > 0:
            return all_errors

        if debug:
            pp(all_features)
            pdb.set_trace()

        df = pd.DataFrame(
            data=all_features,
            index=pd.Index(
                data=[trace_index],
                name='trace_index',
            )
        )
        self.accumulated = self.accumulated.append(df)  # the API guarantees this dataframe
        return None


class TargetsAccumulator(Accumulator):
    def __init__(self):
        super(TargetsAccumulator, self).__init__('TargetMaker')
        self.interarrival_time = feature_makers.InterarrivalTime()

    def accumulate(self, trace_index, trace_record):
        'return (features, err) and append features to self.targets'
        pdb.set_trace()
        interarrival_time_features, err = self.interarrival_time(trace_index, trace_record)
        if err is not None:
            return ['%s: %s' % (interarrival_time_features.name, err)]

        oasspread = trace_record['oasspread']
        if np.isnan(oasspread):
            return ['oasspread is NaN']

        trade_type = trace_record['trade_type']
        assert trade_type in ('B', 'D', 'S')

        row_dict = {
            'id_trace_index': trace_index,
            'id_trade_type': trade_type,
            'id_oasspread': oasspread,
            'id_effectivedatetime': trace_record['effectivedatetime'],
            'id_effectivedate': trace_record['effectivedate'].date(),
            'id_effectivetime': trace_record['effectivetime'],
            'id_quantity': trace_record['quantity'],
            'target_oasspread': oasspread,
            'target_interarrival_seconds': interarrival_time_features['interrarival_seconds_size'],
        }
        row = pd.DataFrame(
            data=row_dict,
            index=pd.Index(
                data=[trace_index],
                name='trace_index',
            )
        )
        self.accumlated = self.accumulated.append(row)
        return None  # no error, data accumulated
