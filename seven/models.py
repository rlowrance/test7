'''hold all info about the models and features'''

import abc
import numpy as np
import pdb
import sklearn.linear_model
import sklearn.ensemble
import unittest

import applied_data_science.timeseries as timeseries


class ExceptionFit(Exception):
    def __init__(self, parameter):
        self.parameter = parameter

    def __str__(self):
        return 'ExceptionFit(%s)' % str(self.parameter)


trade_types = ('B', 'D', 'S')  # trade_types


def just_last_trades(features, targets, n_trades):
    'return DataFrame with up to n_trades before and including the datetime'
    # NOTE: it's possible that the query_index is not in this set
    # That can happend if more than n_trades occur at the exact time of the query trade
    sorted = features.sort_values('id_effectivedatetime')
    n = 1 if n_trades is None else n_trades
    if len(sorted) < n:
        return sorted, targets.loc[sorted.index]
    else:
        reduced = sorted[-n:]
        return reduced, targets.loc[reduced.index]


class NaiveOLD(object):
    'the naive model returns the just prior price'
    def __init__(self, trade_type):
        self.price_column_name = 'prior_oasspread_' + trade_type
        self.predicted = None

    def fit(self, df_samples):
        pdb.set_trace()
        sorted = df_samples.sort_values('id_effectivedatetime')
        price_column = sorted[self.price_column_name]
        self.predicted = price_column.iloc[-1]  # the last prior price
        if np.isnan(self.predicted):
            print 'Naive::fit: predictes is NaN'
            pdb.set_trace()
        return self

    def predict(self):
        ' always predict the prior price; ignore the query'
        pdb.set_trace()
        assert self.predicted is not None, 'call fit() before you call predict()'
        return np.array([self.predicted])


class Model(timeseries.Model):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model_spec, predicted_feature, random_state):
        self.model_spec = model_spec
        self.predicted_feature = predicted_feature
        self.random_state = random_state

        # ElasticNet and RandomForests subclasses set these values
        self.model = None
        self.importances = None
        self.feature_names = None   # list of names used, in same order as importances

        # names of columns on which we rely
        self.column_effectivedatetime = 'id_effectivedatetime'

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
            print 'error: unexpected transform_y value: %s' % self.model_spec.transform_y

    def _fit(self, training_features, training_targets):
        'common fitting procedure for scikit-learn models'
        sorted_indices = self._sorting_indices(training_features)
        relevant_sorted_indices = sorted_indices[-self.model_spec.n_trades_back:]
        relevant_training_features = training_features.loc[relevant_sorted_indices]
        relevant_training_targets = training_targets.loc[relevant_sorted_indices]
        feature_names, x = self._make_featurenames_x(relevant_training_features)
        self.feature_names = feature_names
        y = self._make_y(relevant_training_targets)
        try:
            self.model.fit(x, y)
        except Exception as e:
            raise ExceptionFit(e)

    def _predict(self, query_features):
        'common prediction procedure for scikit-learn models'
        assert len(query_features) == 1
        feature_names, x = self._make_featurenames_x(query_features)
        result = self._untransform(self.model.predict(x))
        return result

    def _transform(self, vector, transform):
        'return possibly-transformed vector (which has no NaN values)'
        if transform is None:
            return vector
        elif transform == 'log':
            return np.log(vector)
        elif transform is 'log1p':
            return np.log1p(vector)
        else:
            print 'error: unexpected transform: %s' % transform
            pdb.set_trace()

    def _has_no_nans(self, vector):
        if np.isnan(vector).any():
            return False
        else:
            return True

    def _make_featurenames_x(self, df):
        'return list of feature_names and np.array 2D with one column for each possibly-transformed feature'
        'return np.array 2D containing the features possibly transformed'
        feature_names = [
            feature_name
            for feature_name in df.columns
            if not feature_name.startswith('id_')
        ]
        shape_transposed = (len(feature_names), len(df))
        result = np.empty(shape_transposed)
        for i, feature in enumerate(feature_names):
            raw_column = df[feature].values
            if not self._has_no_nans(raw_column):
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
        return feature_names, result.transpose()

    def _make_y(self, df):
        raw_column = df[self.predicted_feature]
        transformed_column = self._transform(raw_column, self.model_spec.transform_y)
        if not self._has_no_nans(transformed_column):
            print 'unexpected NaNs'
            print raw_column
            print transformed_column
            print self.predicted_feature
            pdb.set_trace()
        assert self._has_no_nans(transformed_column)
        return transformed_column


class ModelNaive(Model):
    # Given these actual trades in increasing time order:
    #    B 100
    #    D 101
    #    B 102
    #    S 103
    # the predictions for the next trade by trade type are:
    #    next_B 102
    #    next_D 101
    #    next_S 103
    def __init__(self, model_spec, predicted_feature, random_state):
        assert model_spec.name == 'n'
        super(ModelNaive, self).__init__(model_spec, predicted_feature, random_state)

    def fit(self, training_features, training_targets):
        'simple save the last value of the feature we want to predict'
        query = training_targets.iloc[-1]  # the query is the last target row
        self.model = query['info_this_oasspread']  # always predict the current spread
        self.importances = {self.predicted_feature: 1.0}

    def predict(self, query_features):
        'predict the most recent historic trade (which was determined by the fit method'
        return [self.model]  # must return an array-like object


class ModelElasticNet(Model):
    def __init__(self, model_spec, predicted_feature, random_state):
        assert model_spec.name == 'en'
        super(ModelElasticNet, self).__init__(model_spec, predicted_feature, random_state)
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
        self._fit(training_features, training_targets)
        self.importances = {}
        for i, coef in enumerate(self.model.coef_):
            self.importances[self.feature_names[i]] = coef

    def predict(self, query_features):
        return self._predict(query_features)


class ModelRandomForests(Model):
    def __init__(self, model_spec, predicted_features, random_state):
        assert model_spec.name == 'rf'
        super(ModelRandomForests, self).__init__(model_spec, predicted_features, random_state)
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
        self._fit(training_features, training_targets)
        self.importances = {}
        for i, importance in enumerate(self.model.feature_importances_):
            self.importances[self.feature_names[i]] = importance

    def predict(self, query_features):
        return self._predict(query_features)


if __name__ == '__main__':
    unittest.main()
