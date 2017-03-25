'''hold all info about the models and features'''

import abc
import datetime
import numpy as np
import pandas as pd
import pdb
import sklearn.linear_model
import sklearn.ensemble
import sys
import unittest


trade_types = ('B', 'D', 'S')  # trade_types

featuresOLD = (
    'coupon', 'days_to_maturity', 'order_imbalance4',
    'prior_price_B', 'prior_price_D', 'prior_price_S',
    'prior_quantity_B', 'prior_quantity_D', 'prior_quantity_S',
    'trade_price', 'trade_quantity',
    'trade_type_is_B', 'trade_type_is_D', 'trade_type_is_S',
)

size_featuresOLD = (
    'coupon', 'days_to_maturity',
    'prior_quantity_B', 'prior_quantity_D', 'prior_quantity_S',
    'trade_quantity',
)


def make_features_dictOLD(
    id_cusip=None,              # not a feature, but in the features file
    id_effectivedatetime=None,  # not a feature, but in the features file
    amount_issued=None,
    collateral_type_is_sr_unsecured=None,
    coupon_current=None,
    coupon_is_fixed=None,
    coupon_is_floating=None,
    is_callable=None,
    months_to_maturity=None,
    order_imbalance4=None,
    price_delta_ratio_1_day=None,
    price_delta_ratio_2_days=None,
    price_delta_ratio_3_days=None,
    price_delta_ratio_week=None,
    price_delta_ratio_month=None,
    prior_price_B=None,
    prior_price_D=None,
    prior_price_S=None,
    prior_quantity_B=None,
    prior_quantity_D=None,
    prior_quantity_S=None,
    trade_price=None,
    trade_quantity=None,
    trade_type_is_B=None,
    trade_type_is_D=None,
    trade_type_is_S=None,
):
    args_dict = locals().copy()  # NOTE that locals() returns all the locals, at this point, just the args
    result = {}  # Now locals() also contains result; without the copy, so would args_dict
    for k, v in args_dict.iteritems():
        if v is None:
            print 'missing feature %s', k
            assert False, k
        else:
            result[k] = v
    return result


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


class Naive(object):
    'the naive model returns the just prior price'
    def __init__(self, trade_type):
        self.price_column_name = 'prior_oasspread_' + trade_type
        self.predicted = None

    def fit(self, df_samples):
        pdb.set_trace()
        sorted = df_samples.sort_values('id_effectivedatetime')
        price_column = sorted[self.price_column_name]
        self.predicted = price_column.iloc[-1]  # the last prior price
        return self

    def predict(self):
        ' always predict the prior price; ignore the query'
        pdb.set_trace()
        assert self.predicted is not None, 'call fit() before you call predict()'
        return np.array([self.predicted])


def fitOLD(
    model_spec=None,
    training_features=None,
    training_targets=None,
    trade_type=None,
    random_state=None,
    test=False,
):
    'return fitted model and importance of features and possibly an error message'
    # print 'fit', model_spec.name, len(training_samples), trade_type
    if test:
        print 'test fit'
        pdb.set_trace()
    pdb.set_trace()
    last_training_features, last_training_targets = just_last_trades(
        training_features,
        training_targets,
        model_spec.n_trades_back,
    )
    if model_spec.name == 'n':
        m = Naive(trade_type)
        fitted = m.fit(last_training_features)
        return (fitted, None, None)  # no importances
    # NOTE: allow for no training feaatures and training_targets
    # What to the scikit-learn fitting functions return? Presumedly an exception
    make_x, make_y = None
    x = make_x(last_training_features, model_spec.transform_x)
    y = make_y(last_training_targets, model_spec.transform_y, trade_type)
    if model_spec.name == 'en':
        m = sklearn.linear_model.ElasticNet(
            alpha=model_spec.alpha,
            random_state=random_state,
            # explicilly set other hyperparameters to default values
            fit_intercept=True,
            normalize=False,
            max_iter=1000,
            copy_X=False,
            tol=0.0001,
            warm_start=False,
            selection='cyclic',
        )
        try:
            fitted = m.fit(x, y)
        except:
            e = sys.exc_info()
            print 'exception for en m.fit(x,y)', e
            print model_spec
            pdb.set_trace()
            return (fitted, fitted.coef_, e)
        return (fitted, fitted.coef_, None)
    elif model_spec.name == 'rf':
        m = sklearn.ensemble.RandomForestRegressor(
            n_estimators=model_spec.n_estimators,
            max_features=model_spec.max_features,
            max_depth=model_spec.max_depth,
            random_state=random_state,
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
        try:
            fitted = m.fit(x, y)
        except:
            e = sys.exc_info()
            print 'exception for rf m.fit(x,y)', e
            print model_spec
            pdb.set_trace()
            return (fitted, fitted.coef_, e)
        return (fitted, fitted.feature_importances_, None)
    else:
        print 'fit bad model_spec.name: %s' % model_spec.name
        pdb.set_trace()
        return None


def predictOLD(fitted_model=None, model_spec=None, query_sample=None, trade_type=None):
    'return the prediciton'
    assert len(query_sample) == 1
    if isinstance(fitted_model, Naive):
        return fitted_model.predict()
    make_x = None
    x = make_x(query_sample, model_spec.transform_x)
    result_raw = fitted_model.predict(x)
    result_transformed = (
        result_raw if model_spec.transform_y is None else
        np.exp(result_raw) if model_spec.transform_y == 'log' else
        None
    )
    if result_transformed is None:
        raise ValueError('unkndown model_spec.transform_y: %s' % model_spec.transform_y)
    return result_transformed


class Model(object):
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

    @abc.abstractmethod
    def fit(self, training_features, training_targets):
        'set self.model, self.importances, self.feature_names, if the model can be fit'
        pass

    @abc.abstractmethod
    def predict(self, query_sample):
        'return the predicted_feature value in an array of length 1'
        pass

    # helper methods for subclasses to use
    def _sorting_indices(self, df):
        'return indices that reorder the df by self.column_effectivedatetime'
        sorted = df.sort_values(self.column_effectivedatetime)
        return sorted.index

    def _untransform(self, raw_y_value):
        pdb.set_trace()
        if self.model_spec.transform_y is None:
            return raw_y_value
        elif self.model_spec.transform_y == 'log':
            return np.exp(raw_y_value)
        else:
            print 'error: unexpected transform_y value: %s' % self.model_spec.transform_y

    def _fit(self, training_features, training_targets):
        'set self.fitted_model'
        pdb.set_trace()
        sorted_indices = self._sorting_indices(training_features)
        relevant_sorted_indices = sorted_indices[-self.model_spec.n_trades_back]
        relevant_training_features = training_features.loc[relevant_sorted_indices]
        relevant_training_targets = training_features.loc[relevant_sorted_indices]
        try:
            feature_names, x = self._make_featurenames_x(relevant_training_features)
            y = self._make_y(relevant_training_target)
            self.model.fit(x, y)
            self.feature_names = feature_names
        except:
            e = sys.exc_info()
            print 'exception for rf m.fit(x,y)', e
            print self.model_spec
            pdb.set_trace()

    def _predict(self, query_features):
        pdb.set_trace()
        assert len(query_features) == 1
        feature_names, x = self._make_feature_names_x(query_features)
        result = self._untransform_y(self.model.predict(x))
        return result

    def _transform(self, vector, transform):
        'return possibly-transformed vector (which has no NaN values)'
        pdb.set_trace()
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
        pdb.set_trace()
        if np.isnan(vector).any():
            print 'problem: transformed column contains at least one nan'
            pdb.set_trace()
            return False
        else:
            return True

    def _make_featurenames_x(self, df):
        'return list of feature_names and np.array 2D with one column for each possibly-transformed feature'
        'return np.array 2D containing the features possibly transformed'
        pdb.set_trace()
        feature_names = [
            feature_name
            for feature_name in df.columns
            if not feature_name.startswith('id_')
        ]
        pdb.set_trace()
        shape_transposed = (len(feature_names), len(df))
        result = np.empty(shape_transposed)
        for i, feature in enumerate(feature_names):
            raw_column = df[feature].values
            transformed_column = (
                self._transform(raw_column, self.model_spec.transform_x) if feature.endswith('_size') else
                raw_column
            )
            assert self._has_no_nans(transformed_column)
            result[i] = transformed_column
        return feature_names, result.transpose()

    def _make_y(self, df):
        pdb.set_trace()
        raw_column = df[self.predicted_feature]
        transformed_column = self._transform(raw_column, self.model_spec.transform_y)
        assert self._has_no_nans(transformed_column)
        return transformed_column


class ModelNaive(Model):
    def __init__(self, model_spec, predicted_feature, random_state):
        assert model_spec.name == 'n'
        super(ModelNaive, self).__init__(model_spec, predicted_feature, random_state)

    def fit(self, training_features, training_targets):
        'simple save the last value of the feature we want to predict'
        if self.predicted_feature not in training_targets.columns:
            print 'error: predicted feature (%s) not in training targets' % self.predicted_feature
            print training_targets.columns
            pdb.set_trace()
        sorted_indices = self._sorting_indices(training_features)
        self.model = training_features.loc[sorted_indices].iloc[-1][self.predicted_feature]
        self.importances = {
            self.predicted_feature: 1.0,
        }
        return

    def predict(self, query_sample):
        ' always predict the prior feature; ignore the query'
        return [self.model]  # must return an array-like object
        return


class ModelElasticNet(Model):
    def __init__(self, model_spec, predicted_feature, random_state):
        pdb.set_trace()
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
        pdb.set_trace()
        self._fit(training_features, training_targets)
        self.importances = {}
        for i, coef in enumerate(self.coef_):
            self.importances[self.features[i]] = coef

    def predict(self, query_features):
        pdb.set_trace()
        return self._predict(query_features)


class ModelRandomForests(Model):
    def __init__(self, model_spec, predicted_features, random_state):
        pdb.set_trace()
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
        pdb.set_trace()
        self._fit(training_features, training_targets)
        self.importances = {}
        for i, importance in enumerate3(self.importances_):
            self.importances[self.features[i]] = importance

    def predict(self, query_features):
        pdb.set_trace()
        return self._predict(query_features)


def read_csv(path, date_columns=None, usecols=None, index_col=0, nrows=None, parse_dates=None, verbose=False):
    if index_col is not None and usecols is not None:
        print 'cannot read both the index column and specific columns'
        print 'possibly a bug in scikit-learn'
        pdb.set_trace()
    df = pd.read_csv(
        path,
        index_col=index_col,
        nrows=nrows,
        usecols=usecols,
        low_memory=False,
        parse_dates=parse_dates,
    )
    if verbose:
        print 'read %d rows from file %s' % (len(df), path)
        print df.columns
    return df


def make_effectivedatetime(df, effectivedate_column='effectivedate', effectivetime_column='effectivetime'):
    '''create new column that combines the effectivedate and effective time

    example:
    df['effectivedatetime'] = make_effectivedatetime(df)
    '''
    values = []
    for the_date, the_time in zip(df[effectivedate_column], df[effectivetime_column]):
        values.append(datetime.datetime(
            the_date.year,
            the_date.month,
            the_date.day,
            the_time.hour,
            the_time.minute,
            the_time.second,
        ))
    return pd.Series(values, index=df.index)


if __name__ == '__main__':
    unittest.main()
