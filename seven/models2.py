'''hold all info about the models and features

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''

import abc
import numbers
import numpy as np
import pandas as pd
import pdb
import sklearn.linear_model
import sklearn.ensemble
import sys
import unittest

import applied_data_science.timeseries as timeseries

from . import logging


def synthetic_query(query_features, trade_type):
    'return query_features, but with the trade_type reset'
    result = pd.DataFrame()
    for index, row in query_features.iterrows():
        # there is only 1 row
        row['p_trade_type_is_B'] = 0
        row['p_trade_type_is_D'] = 0
        row['p_trade_type_is_S'] = 0
        row['p_trade_type_is_%s' % trade_type] = 1
        if row['id_otr1_cusip'] == row['id_p_cusip'] and row['id_otr1_effectivedatetime'] == row['id_p_effectivedatetime']:
            pdb.set_trace()
            row['otr1_trade_type_is_B'] = 0
            row['otr1_trade_type_is_D'] = 0
            row['otr1_trade_type_is_S'] = 0
            row['otr1_trade_type_is_%s' % trade_type] = 1
        result = result.append(row)
    return result


class ExceptionFit(Exception):
    def __init__(self, message):
        # call base class constructor
        super(ExceptionFit, self).__init__(message)
        self.message = message  # the base class may also save the message

    def __str__(self):
        return 'ExceptionFit(%s)' % self.message


class ExceptionPredict(Exception):
    def __init__(self, parameter):
        self.parameter = parameter

    def __str__(self):
        return 'ExceptionPredict(%s)' % self.parameter


trade_types = ('B', 'D', 'S')  # trade_types


class Model(timeseries.Model, metaclass=abc.ABCMeta):
    def __init__(self, model_spec, random_state):
        self.model_spec = model_spec
        self.random_state = random_state

        # ElasticNet and RandomForests subclasses set these values
        self.model = None
        self.importances = None
        self.feature_names = None   # list of names used, in same order as importances

    # helper methods for subclasses to use

    def _fit(self, training_features, training_targets, trace=False):
        'common fitting procedure for scikit-learn models'
        assert isinstance(training_features, list)  # a list of dict[feature_name, feature_value]
        assert isinstance(training_targets, list)   # a list of numbers
        if trace:
            pdb.set_trace()

        # select the number of trades back
        n_trades_back = self.model_spec.n_trades_back
        # Note: the indexing just below succeeds even if there is one training feature
        # and n_traces_back is greater than 1: it returns all of the trades in that case
        relevant_training_features = training_features[-n_trades_back:]
        relevant_training_targets = training_targets[-n_trades_back:]
        if len(relevant_training_features) == 0:
            raise ExceptionFit('no training data after looking back n trades (%d)' % n_trades_back)

        assert isinstance(relevant_training_features[0], dict)
        assert isinstance(relevant_training_targets[0], numbers.Number)

        feature_names, x = self._make_featurenames_x(relevant_training_features)
        self.feature_names = feature_names
        y = self._make_y(relevant_training_targets)
        try:
            self.model.fit(x, y)
        except Exception as e:
            print('exception in _fit:', e)
            raise ExceptionFit(e)

    def _make_featurenames_x(self, feature_vectors, trace=False):
        'return list of feature_names and np.array 2D with one column for each possibly-transformed feature'
        'return np.array 2D containing the features possibly transformed'
        if trace:
            pdb.set_trace()
        feature_names = [
            feature_name
            for feature_name in sorted(feature_vectors[0].keys())
            if not feature_name.startswith('id_')
        ]
        if trace:
            pdb.set_trace()
        shape = (len(feature_vectors), len(feature_names))
        result = np.zeros(shape)
        for row_index, feature_vector in enumerate(feature_vectors):
            for column_index, feature_name in enumerate(feature_names):
                raw_value = feature_vector[feature_name]
                if raw_value != raw_value:
                    raise timeseries.ExceptionFit('nan at (%d,%d) (%s,%s)' % (
                        row_index,
                        column_index,
                        feature_vector,
                        feature_name,
                    ))
                value = self._transform_raw_value(
                    self.model_spec.transform_x,
                    feature_name,
                    raw_value)
                result[row_index, column_index] = value
        return feature_names, result

    def _make_y(self, targets, trace=False):
        'return 1D np.array'
        assert isinstance(targets, list)
        result = np.zeros(len(targets))
        for row_index, raw_value in enumerate(targets):
            if raw_value != raw_value:
                    raise timeseries.ExceptionFit('nan at (%d) (%s)' % (
                        row_index,
                        targets,
                    ))
            value = self._transform_raw_value(
                self.model_spec.transform_y,
                'target_value',
                raw_value,
            )
            result[row_index] = value
        return result

    def _predict(self, feature_vectors, trace=False):
        'common prediction procedure for scikit-learn models'
        assert isinstance(feature_vectors, list)
        if trace:
            pdb.set_trace()
        feature_names, x = self._make_featurenames_x(feature_vectors)
        result = self._untransform(self.model.predict(x))
        if trace:
            pdb.set_trace()
        return result

    def _transform_raw_value(self, transformation, feature_name, raw_value):
        if feature_name.endswith('_size'):
            if transformation is None:
                return raw_value
            elif transformation == 'log':
                assert raw_value > 0.0
                return np.log(raw_value)
            elif transformation == 'log1p':
                assert raw_value + 1 > 0.0
                return np.log1p(raw_value)
            else:
                logging.critical('unexpected transformation: %s' % transformation)
                sys.exit(1)
        else:
            return raw_value

    def _untransform(self, raw_y_value):
        if self.model_spec.transform_y is None:
            return raw_y_value
        elif self.model_spec.transform_y == 'log':
            return np.exp(raw_y_value)
        else:
            print('error: unexpected transform_y value: %s' % self.model_spec.transform_y)


class ModelNaive(Model):
    def __init__(self, model_spec, random_state):
        assert model_spec.name == 'n'
        super(ModelNaive, self).__init__(model_spec, random_state)

    def fit(self, training_features, training_targets):
        'predict the last trade of the type '
        'simple save the last value of the feature we want to predict'
        self.predictions = [training_targets[-1]]  # predictions must be a list
        self.importances = {}   # no features, no importances

    def predict(self, query_features):
        'predict the most recent historic trade for the trade_type'
        return self.predictions


class ModelElasticNet(Model):
    def __init__(self, model_spec, random_state):
        assert model_spec.name == 'en'
        super(ModelElasticNet, self).__init__(model_spec, random_state)
        self.model = sklearn.linear_model.ElasticNet(
            alpha=model_spec.alpha,
            l1_ratio=model_spec.l1_ratio,
            random_state=self.random_state,
            # explicilly set other hyperparameters to default values
            fit_intercept=True,
            normalize=False,
            max_iter=1000,
            copy_X=False,
            tol=0.0001,
            warm_start=False,
            selection='cyclic',
        )

    def fit(self, training_features, training_targets):
        'set self.fitted_model'
        self._fit(training_features, training_targets, trace=False)
        self.importances = {}
        for i, coef in enumerate(self.model.coef_):
            self.importances[self.feature_names[i]] = coef

    def predict(self, query_features):
        return self._predict(query_features)


class ModelRandomForests(Model):
    def __init__(self, model_spec, random_state):
        assert model_spec.name == 'rf'
        super(ModelRandomForests, self).__init__(model_spec, random_state)
        self.model = sklearn.ensemble.RandomForestRegressor(
            n_estimators=model_spec.n_estimators,
            max_features=model_spec.max_features,
            max_depth=model_spec.max_depth,
            random_state=self.random_state,
            # explicitly set other hyperparameters to default values
            criterion='mse',
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0,
            max_leaf_nodes=None,
            min_impurity_split=1e-7,
            bootstrap=True,
            oob_score=False,
            n_jobs=1,
            verbose=0,
            warm_start=False,
        )

    def fit(self, training_features, training_targets):
        self._fit(training_features, training_targets, trace=False)
        self.importances = {}
        for i, importance in enumerate(self.model.feature_importances_):
            self.importances[self.feature_names[i]] = importance

    def predict(self, query_features):
        return self._predict(query_features, trace=False)


if __name__ == '__main__':
    unittest.main()
