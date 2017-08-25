'''hold all info about the models and features

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''

import abc
import numpy as np
import pandas as pd
import pdb
import sklearn.linear_model
import sklearn.ensemble
import unittest

import applied_data_science.timeseries as timeseries


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

        # names of columns on which we rely
        self.column_effectivedatetime = 'id_p_effectivedatetime'

    # helper methods for subclasses to use
    def _sorting_indices(self, df):
        'return indices that reorder the df by self.column_effectivedatetime'
        sorted = df.sort_values(self.column_effectivedatetime)
        return sorted.index

    def _untransform(self, raw_y_value):
        if self.model_spec.transform_y is None:
            return raw_y_value
        elif self.model_spec.transform_y == 'log':
            return np.exp(raw_y_value)
        else:
            print('error: unexpected transform_y value: %s' % self.model_spec.transform_y)

    def _fit(self, training_features, training_targets, trace=False):
        'common fitting procedure for scikit-learn models'
        if trace:
            pdb.set_trace()

        # select the number of trades back
        # that requires sorting the training data
        sorted_features, sorted_targets = self._sort_by_effectivedatetime(
            training_features,
            training_targets,
        )
        n_trades_back = self.model_spec.n_trades_back
        relevant_training_features = sorted_features.iloc[-n_trades_back:]
        relevant_training_targets = sorted_targets.iloc[-n_trades_back:]
        if len(relevant_training_features) == 0:
            raise ExceptionFit('no training data after looking back n trades (%d)' % n_trades_back)

        feature_names, x = self._make_featurenames_x(relevant_training_features)
        self.feature_names = feature_names
        y = self._make_y(relevant_training_targets)
        try:
            self.model.fit(x, y)
        except Exception as e:
            print('exception in _fit:', e)
            pdb.set_trace()
            raise ExceptionFit(e)

    def _predict(self, query_features, trace=False):
        'common prediction procedure for scikit-learn models'
        if trace:
            pdb.set_trace()
        assert len(query_features) == 1
        feature_names, x = self._make_featurenames_x(query_features)
        result = self._untransform(self.model.predict(x))
        if trace:
            pdb.set_trace()
        return result

    def _sort_by_effectivedatetime(self, features, targets):
        assert len(features) == len(targets)
        sorted_features = features.sort_values('id_p_effectivedatetime')
        sorted_targets = targets.loc[sorted_features.index]
        return sorted_features, sorted_targets

    def _transform(self, vector, transform):
        'return possibly-transformed vector (which has no NaN values)'
        if transform is None:
            return vector
        elif transform == 'log':
            return np.log(vector)
        elif transform == 'log1p':
            return np.log1p(vector)
        else:
            print('error: unexpected transform: %s' % transform)
            pdb.set_trace()

    def _has_no_nans(self, vector):
        if np.isnan(vector).any():
            return False
        else:
            return True

    def _make_featurenames_x(self, df, trace=False):
        'return list of feature_names and np.array 2D with one column for each possibly-transformed feature'
        'return np.array 2D containing the features possibly transformed'
        if trace:
            pdb.set_trace()
        feature_names = [
            feature_name
            for feature_name in df.columns
            if not feature_name.startswith('id_')
        ]
        if trace:
            pdb.set_trace()
        shape_transposed = (len(feature_names), len(df))
        result = np.empty(shape_transposed)
        for i, feature in enumerate(feature_names):
            raw_column = df[feature].values
            if not self._has_no_nans(raw_column):
                # print 'about to raise timeseries.ExceptionFit(', i, feature
                # pdb.set_trace()
                raise timeseries.ExceptionFit('models:_make_featurenames_x: raw column for feature %s has NaN values; 1st index %d' % (
                    feature,
                    df.index[0],
                ))
            transformed_column = (
                self._transform(raw_column, self.model_spec.transform_x) if feature.endswith('_size') else
                raw_column
            )
            if not self._has_no_nans(transformed_column):
                raise timeseries.ExceptionFit('models:_make_featurenames_x: transformed column for feature %s has NaN values: 1st index %d' % (
                    feature,
                    df.index[0],
                ))
            result[i] = transformed_column
        if trace:
            pdb.set_trace()
        return feature_names, result.transpose()

    def _make_y(self, df, trace=False):
        if trace:
            pdb.set_trace()
        shape_transposed = (len(df.columns), len(df))
        result = np.empty(shape_transposed)
        for i, column_name in enumerate(df.columns):
            raw_column = df[column_name]
            transformed_column = (
                self._transform(raw_column, self.model_spec.transform_y) if column_name.endswith('_size') else
                raw_column
            )
            result[i] = transformed_column
        if trace:
            pdb.set_trace()
        return result.transpose()


class ModelNaive(Model):
    def __init__(self, model_spec, random_state):
        assert model_spec.name == 'n'
        super(ModelNaive, self).__init__(model_spec, random_state)

    def fit(self, training_features, training_targets):
        'predict the last trade of the type '
        'simple save the last value of the feature we want to predict'
        sorted_training_features, sorted_training_targets = self._sort_by_effectivedatetime(
            training_features,
            training_targets,
        )
        self.predictions = sorted_training_targets.iloc[-1].values  # : np.ndarray
        self.importances = None

    def predict(self, query_features):
        'predict the most recent historic trade for the trade_type'
        return self.predictions  # must return an array-like object


class ModelElasticNet(Model):
    def __init__(self, model_spec, random_state):
        assert model_spec.name == 'en'
        super(ModelElasticNet, self).__init__(model_spec, random_state)
        self.model = sklearn.linear_model.ElasticNet(
            alpha=model_spec.alpha,
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
