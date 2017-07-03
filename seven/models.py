'''hold all info about the models and features'''

import abc
import copy
import numpy as np
import pdb
import sklearn.linear_model
import sklearn.ensemble
import unittest

import applied_data_science.timeseries as timeseries


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


class Model(timeseries.Model):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model_spec, target_name, random_state):
        self.model_spec = model_spec
        self.target_name = target_name
        self.random_state = random_state

        # ElasticNet and RandomForests subclasses set these values
        self.model = None
        self.importances = None
        self.feature_names = None   # list of names used, in same order as importances

        # names of columns on which we rely
        self.column_effectivedatetime = 'id_effectivedatetime'

        # for now, allow only one target. Later allow others (such as inter-trade time)
        if self.target_name != 'target_oasspread':
            print 'unexpected target_name', self.target_name
            pdb.set_trace()

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

    def _fit(self, training_features, training_targets, trace=False):
        'common fitting procedure for scikit-learn models'
        # this code implements the decision in file targets2.txt

        # MAYBE: In account for n_trades_back, we view the sorted trade print streams as ordered elswhere
        # however, we know the time stamps for each trace print.
        # An alternative approach is to use weighted average feature values for each time period.
        # We would apply this alternative approach to trades at the edge of the lookback window
        if trace:
            pdb.set_trace()

        # remove trades that are not for the self.target_name
        if False:
            # the original version of the fitting code fitted only on the transactions of the same trade type
            # as the target
            # It's better to train on as much data as possible
            mask = np.isnan(training_targets[self.target_name])
            relevant_targets = training_targets.loc[~mask]
            relevant_features = training_features.loc[~mask]
            if len(relevant_targets) == 0:
                raise ExceptionFit('no training data after remove non-target samples')
        else:
            relevant_features = training_features
            relevant_targets = training_targets

        # select the number of trades back
        # that requires sorting the training data
        sorted_features = relevant_features.sort_values('id_effectivedatetime')
        sorted_targets = relevant_targets.loc[sorted_features.index]
        n_trades_back = self.model_spec.n_trades_back
        relevant_training_features = sorted_features.iloc[-n_trades_back:]
        relevant_training_targets = sorted_targets.iloc[-n_trades_back:]
        if len(relevant_training_features) == 0:
            raise ExceptionFit('no training data after looking back n trades (%d)' % n_trades_back)

        # # only use the n_trade_back remaining trades

        # n_trades_back = self.model_spec.n_trades_back
        # in_window_features = sorted_features.iloc[-n_trades_back:]
        # in_window_targets = sorted_targets.iloc[-n_trades_back:]
        # if len(in_window_features) == 0:
        #     raise ExceptionFit('no training data within lookback window')

        # # Drop rows that dont' have the predicted variable in their targets
        # # Train only samples that are for the requested target variable (which is in self.target_name)
        # # # NOTE: this selection interacts with the n_trades_back model spec to use often fewer trades
        # # than n_trades_back specifies
        # bad_indices = []
        # assert (in_window_features.index == in_window_targets.index).all()
        # for index, row in in_window_targets.iterrows():
        #     if np.isnan(row[self.target_name]):
        #         bad_indices.append(index)
        # relevant_training_features = in_window_features.drop(bad_indices)
        # relevant_training_targets = in_window_targets.drop(bad_indices)
        # if len(relevant_training_features) == 0:
        #     raise ExceptionFit('no training data with %s' % self.target_name)

        feature_names, x = self._make_featurenames_x(relevant_training_features)
        self.feature_names = feature_names
        y = self._make_y(relevant_training_targets)
        try:
            self.model.fit(x, y)
        except Exception as e:
            print 'exception in _fit:', e
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

    def _transform(self, vector, transform):
        'return possibly-transformed vector (which has no NaN values)'
        if transform is None:
            return vector
        elif transform == 'log':
            return np.log(vector)
        elif transform == 'log1p':
            return np.log1p(vector)
        else:
            print 'error: unexpected transform: %s' % transform
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
        raw_column = df[self.target_name]
        transformed_column = self._transform(raw_column, self.model_spec.transform_y)
        if not self._has_no_nans(transformed_column):
            # print 'unexpected NaNs'
            # print raw_column
            # print transformed_column
            # print self.target_name
            raise ExceptionFit('_make_y: NaN in transformed column %s transformation %s' % (
                self.target_name,
                self.model_spec.transform_y,
            ))
        if trace:
            pdb.set_trace()
        return transformed_column


class ModelNaive(Model):
    def __init__(self, model_spec, target_name, random_state):
        assert model_spec.name == 'n'
        super(ModelNaive, self).__init__(model_spec, target_name, random_state)

    def fit(self, training_features, training_targets):
        'predict the last trade of the type '
        'simple save the last value of the feature we want to predict'
        assert len(training_targets) > 0
        sorted_training_targets = training_targets.sort_values('id_effectivedatetime')
        self.prediction = sorted_training_targets.iloc[-1][self.target_name]
        self.importances = None

    def predict(self, query_features):
        'predict the most recent historic trade for the trade_type'
        return [self.prediction]  # must return an array-like object


class ModelElasticNet(Model):
    def __init__(self, model_spec, target_name, random_state):
        assert model_spec.name == 'en'
        super(ModelElasticNet, self).__init__(model_spec, target_name, random_state)
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
    def __init__(self, model_spec, target_name, random_state):
        assert model_spec.name == 'rf'
        super(ModelRandomForests, self).__init__(model_spec, target_name, random_state)
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
        return self._predict(query_features, trace=False)


if __name__ == '__main__':
    unittest.main()
