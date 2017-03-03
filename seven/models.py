'''hold all info about the models and features'''

import datetime
import pdb
import unittest


import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.ensemble


class ModelSpec(object):
    'name of model + its hyperparameters'
    model_names = ('naive', 'rf', 'en')

    # hyperparamater settings grid
    # For now, just enought to exercise fit-predict.py
    transform_xs = (None, 'log1p')
    transform_ys = (None, 'log')
    alphas = (0.01, 1.00)  # weight on penalty term
    l1_ratios = (0.01, 0.99)  # 0 ==> only L2 penalty, 1 ==> only L1 penalty
    n_estimatess = (1, 10)    # number of trees in the forest
    max_featuress = (1, 3)     # number of features to consider when looking for best split
    max_depths = (2, None)     # max depth of a tree

    # cross-model parameters
    random_state = 123

    def __init__(
        self,
        name=None,
        # feature transformation just before fitting and predicting
        transform_x=None,
        transform_y=None,
        # hyperparameters for ElasticNet models in scikit-learn
        alpha=None,
        l1_ratio=None,
        # hyperameters for GradientBoostingRegressor models in scikit-learn
        n_estimates=None,
        max_depth=None,
        max_features=None,
    ):
        def is_2_digit_float(value):
            assert isinstance(value, float), 'not a float: %s' % value
            if round(value, 2) == value:
                return True
            else:
                print 'does not have 2 decimal digits: %s' % value
                pdb.set_trace()

        def is_int(value):
            return isinstance(value, int)

        assert isinstance(name, str) and name in ModelSpec.model_names
        if name == 'naive':
            assert transform_x is None or transform_x in ModelSpec.transform_xs
            assert transform_y is None or transform_y in ModelSpec.transform_ys
            assert alpha is None
            assert l1_ratio is None
            assert n_estimates is None
            assert max_depth is None
            assert max_features is None
        elif name == 'en':
            assert transform_x in ModelSpec.transform_xs
            assert transform_y in ModelSpec.transform_ys
            assert is_2_digit_float(alpha)
            assert is_2_digit_float(l1_ratio)
            assert 0.0 <= alpha <= 1.0
            assert 0.0 <= l1_ratio <= 1.0
            assert n_estimates is None
            assert max_depth is None
            assert max_features is None
        elif name == 'rf':
            assert transform_x is None
            assert transform_y is None
            assert alpha is None
            assert l1_ratio is None
            assert n_estimates is not None
            assert n_estimates is None or is_int(n_estimates)
            assert max_depth is None or is_int(max_depth)
            assert max_features is None or is_int(max_features)
        else:
            print 'internal error __init__ name:', name
            pdb.set_trace()
        self.name = name
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_estimates = n_estimates
        self.max_depth = max_depth
        self.max_features = max_features

    def __str__(self):
        'return parsable string version'
        def to_str(value):
            if value is None:
                return '-'
            else:
                return '-%s' % value

        return (
            self.name +
            to_str(self.transform_x) +
            to_str(self.transform_y) +
            to_str(self.alpha) +
            to_str(self.l1_ratio) +
            to_str(self.n_estimates) +
            to_str(self.max_depth) +
            to_str(self.max_features)
        )

    @staticmethod
    def make_from_str(s):
        'return the ModelSpec that printed as the str s'
        values = s.split('-')
        assert len(values) == 8
        name, transform_x, transform_y, alpha, l1_ratio, n_estimates, max_depth, max_features = values
        assert name in ModelSpec.model_names
        return ModelSpec(
            name=name,
            transform_x=None if transform_x == '' else transform_x,
            transform_y=None if transform_y == '' else transform_y,
            alpha=None if alpha == '' else float(alpha),
            l1_ratio=None if l1_ratio == '' else float(l1_ratio),
            n_estimates=None if n_estimates == '' else int(n_estimates),
            max_depth=None if max_depth == '' else int(max_depth),
            max_features=None if max_features == '' else int(max_features),
        )

    def __eq__(self, other):
        if not isinstance(other, ModelSpec):
            return False
        return (
            self.name == other.name and
            self.transform_x == other.transform_x and
            self.transform_y == other.transform_y and
            self.alpha == other.alpha and
            self.l1_ratio == other.l1_ratio and
            self.n_estimates == other.n_estimates and
            self.max_depth == other.max_depth and
            self.max_features == other.max_features
        )

    def __hash__(self):
        return hash((
            self.name,
            self.transform_x,
            self.transform_y,
            self.alpha,
            self.l1_ratio,
            self.n_estimates,
            self.max_depth,
            self.max_features,
        ))


def make_all_model_specs():
    'return List[ModelSpec]'
    result = []
    for name in ModelSpec.model_names:
        if name == 'naive':
            result.append(
                ModelSpec(
                    name=name,
                    )
            )
        elif name == 'en':
            for transform_x in ModelSpec.transform_xs:
                for transform_y in ModelSpec.transform_ys:
                    for alpha in ModelSpec.alphas:
                        for l1_ratio in ModelSpec.l1_ratios:
                            result.append(
                                ModelSpec(
                                    name=name,
                                    alpha=alpha,
                                    l1_ratio=l1_ratio,
                                )
                            )
        elif name == 'rf':
            for n_estimates in ModelSpec.n_estimatess:
                for max_depth in ModelSpec.max_depths:
                    for max_features in ModelSpec.max_featuress:
                        result.append(
                            ModelSpec(
                                name=name,
                                n_estimates=n_estimates,
                                max_depth=max_depth,
                                max_features=max_features,
                            )
                        )
        else:
            print 'internal error: make_all_model_specs name:', name
            pdb.set_trace()
    return result


all_model_specs = make_all_model_specs()

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
    assert transform_x in ModelSpec.transform_xs
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
    assert transform_y in ModelSpec.transform_ys
    column_name = 'next_price_' + trade_type
    column = df[column_name]
    transformed_column = (
        column if transform_y is None else
        column.log() if transform_y is None else
        None  # this is an error
    )
    if transformed_column is None:
        print 'make_y bad transformation', transform_y
        pdb.set_trace()
    return transformed_column


def fit(model_spec, training_features, training_targets, trade_type):
    'return fitted model and importance of features'
    # print 'fit', model_spec.name, len(training_samples), trade_type
    if model_spec.name == 'naive':
        m = Naive(trade_type)
        fitted = m.fit(training_features)
        return (fitted, None)  # no importance
    x = make_x(training_features, model_spec.transform_x)
    y = make_y(training_targets, model_spec.transform_y, trade_type)
    if model_spec.name == 'en':
        m = sklearn.linear_model.ElasticNet(
            alpha=model_spec.alpha,
            random_state=ModelSpec.random_state,
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
            n_estimators=model_spec.n_estimates,
            max_features=model_spec.max_features,
            max_depth=model_spec.max_depth,
            random_state=ModelSpec.random_state,
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
        result_raw.exp() if model_spec.transform_y == 'log' else
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


class TestModelSpec(unittest.TestCase):
    def test_construction_good(self):
        verbose = False
        naive = ModelSpec(
            name='naive'
        )
        if verbose:
            print naive
        en = ModelSpec(
            name='en',
            transform_x='log1p',
            alpha=0.20,
            l1_ratio=0.50,
        )
        if verbose:
            print en
        rf = ModelSpec(
            name='rf',
            n_estimates=10,
            max_depth=10,
            max_features=3,
        )
        if verbose:
            print rf
        self.assertTrue(True)  # for now, just test run to completion

    def test_to_from_str(self):
        'check for each ModelSpec in the grid'
        for model_spec in all_model_specs:
            s = str(model_spec)
            model_spec2 = ModelSpec.make_from_str(s)
            self.assertEqual(model_spec, model_spec2)
            self.assertEquals(s, str(model_spec2))

if __name__ == '__main__':
    unittest.main()
