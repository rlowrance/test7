'''hold all info about the models and features'''

import datetime
import numpy as np
import pandas as pd
import pdb
import sklearn.linear_model
import sklearn.ensemble
import unittest

from ModelSpec import ModelSpec

trade_types = ('B', 'D', 'S')  # trade_types

features = (
    'coupon', 'days_to_maturity', 'order_imbalance4',
    'prior_price_B', 'prior_price_D', 'prior_price_S',
    'prior_quantity_B', 'prior_quantity_D', 'prior_quantity_S',
    'trade_price', 'trade_quantity',
    'trade_type_is_B', 'trade_type_is_D', 'trade_type_is_S',
)

size_features = (
    'coupon', 'days_to_maturity', 'order_imbalance4',
    'prior_quantity_B', 'prior_quantity_D', 'prior_quantity_S',
    'trade_quantity',
)


def make_features_dict(
    coupon=None,
    cusip=None,              # not a feature, but in the features file
    days_to_maturity=None,
    effectivedatetime=None,  # not a feature, but in the features file
    order_imbalance4=None,
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
    sorted = features.sort_values('effectivedatetime')
    if len(sorted) < n_trades:
        return sorted, targets.loc[sorted.index]
    else:
        reduced = sorted[-n_trades:]
        return reduced, targets.loc[reduced.index]


class Naive(object):
    'the naive model returns the just prior price'
    def __init__(self, trade_type):
        self.price_column_name = 'prior_price_' + trade_type
        self.predicted = None

    def fit(self, df_samples):
        sorted = df_samples.sort_values('effectivedatetime')
        price_column = sorted[self.price_column_name]
        self.predicted = price_column.iloc[-1]  # the last prior price
        return self

    def predict(self):
        ' always predict the prior price; ignore the query'
        assert self.predicted is not None, 'call fit() before you call predict()'
        return np.array([self.predicted])


def make_x(df, transform_x):
    'return np.array 2D containing the features possibly transformed'
    shape_transposed = (len(features), len(df))
    result = np.empty(shape_transposed)
    for i, feature in enumerate(features):
        raw_column = df[feature].values
        transformed_column = (
            raw_column if feature not in size_features else
            np.log1p(raw_column) if transform_x == 'log1p' else  # inverse is np.expm1()
            raw_column if transform_x is None else
            None  # this is an error
        )
        if transformed_column is None:
            print 'make_x bad transformation', transform_x
            pdb.set_trace()
        result[i] = transformed_column
    return result.transpose()


def make_y(df, transform_y, trade_type):
    column_name = 'next_price_' + trade_type
    column = df[column_name]
    transformed_column = (
        column if transform_y is None else
        np.log(column) if transform_y is 'log' else
        None  # this is an error
    )
    if transformed_column is None:
        print 'make_y bad transformation', transform_y
        pdb.set_trace()
    return transformed_column


def fit(model_spec, training_features, training_targets, trade_type, random_state):
    'return fitted model and importance of features'
    # print 'fit', model_spec.name, len(training_samples), trade_type
    last_training_features, last_training_targets = just_last_trades(
        training_features,
        training_targets,
        model_spec.n_trades_back,
    )
    if model_spec.name == 'n':
        m = Naive(trade_type)
        fitted = m.fit(last_training_features)
        return (fitted, None)  # no importances
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
        fitted = m.fit(x, y)
        return (fitted, fitted.coef_)
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
        fitted = m.fit(x, y)
        return (fitted, fitted.feature_importances_)
    else:
        print 'fit bad model_spec.name: %s' % model_spec.name
        pdb.set_trace()
        return None


def predict(fitted_model, model_spec, query_sample, trade_type):
    'return the prediciton'
    if isinstance(fitted_model, Naive):
        return fitted_model.predict()
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


def read_csv(path, date_columns=None, usecols=None, index_col=0, nrows=None, parse_dates=None, verbose=False):
    if index_col is not None and usecols is not None:
        print 'cannot read both the index column and specific columns'
        print 'possibly a bug in scikit-learn'
        pdb.set_trace()
    df = pd.read_csv(
        path,
        index_col=0,
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
