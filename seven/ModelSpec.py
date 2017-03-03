'''model specificiation (=model + hyperparameters)'''

import pdb
import unittest


class ModelSpec(object):
    'name of model + its hyperparameters'
    allowed_values_name = ('n', 'rf', 'en')
    allowed_values_max_features_str = ('auto', 'sqrt', 'log2')
    allowed_values_transform_x = ('log1p',)
    allowed_values_transform_y = ('log',)

    def __init__(
        self,
        name=None,
        n_trades_back=None,
        # feature transformation just before fitting and predicting
        transform_x=None,
        transform_y=None,
        # hyperparameters for ElasticNet models in scikit-learn
        alpha=None,
        l1_ratio=None,
        # hyperameters for GradientBoostingRegressor models in scikit-learn
        n_estimators=None,
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

        def is_float(value):
            return isinstance(value, float)

        def is_int(value):
            return isinstance(value, int)

        assert name in ModelSpec.allowed_values_name
        assert isinstance(n_trades_back, int) and n_trades_back > 0
        if name == 'n':
            assert transform_x is None
            assert transform_y is None
            assert alpha is None
            assert l1_ratio is None
            assert n_estimators is None
            assert max_depth is None
            assert max_features is None
        elif name == 'en':
            assert transform_x is None or transform_x in ModelSpec.allowed_values_transform_x
            assert transform_y is None or transform_y in ModelSpec.allowed_values_transform_y
            # test for what sklearn.linear_model.ElasticNet allows
            assert is_float(alpha)
            assert 0.0 <= l1_ratio <= 1.0
            assert n_estimators is None
            assert max_depth is None
            assert max_features is None
        elif name == 'rf':
            assert transform_x is None
            assert transform_y is None
            assert alpha is None
            assert l1_ratio is None
            # test for what sklearn.ensemble.RandomForestRegressor allows
            assert is_int(n_estimators)
            assert max_depth is None or is_int(max_depth)
            assert (
                max_features is None or
                is_int(max_features) or
                is_float(max_features) or
                max_features in ModelSpec.allowed_values_max_features_str
            )
        else:
            print 'internal error __init__ name:', name
            pdb.set_trace()
        self.name = name
        self.n_trades_back = n_trades_back
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features

    def __str__(self):
        'return parsable string representation'
        def to_str(value):
            if value is None:
                return ''
            elif isinstance(value, float):
                return ('%f' % value).replace('.', '_')
            elif isinstance(value, int):
                return '%d' % value
            else:
                return str(value)

        return '%s-%s-%s-%s-%s-%s-%s-%s-%s' % (
            self.name,
            to_str(self.n_trades_back),
            to_str(self.transform_x),
            to_str(self.transform_y),
            to_str(self.alpha),
            to_str(self.l1_ratio),
            to_str(self.n_estimators),
            to_str(self.max_depth),
            to_str(self.max_features),
        )

    @staticmethod
    def make_from_str(s):
        'return the ModelSpec that printed as the str s'
        values = s.split('-')
        assert len(values) == 9
        (
            name,
            n_trades_back,
            transform_x,
            transform_y,
            alpha,
            l1_ratio,
            n_estimators,
            max_depth,
            max_features,
         ) = values
        assert name in ModelSpec.allowed_values_name
        return ModelSpec(
            name=name,
            n_trades_back=int(n_trades_back),
            transform_x=None if transform_x == '' else transform_x,
            transform_y=None if transform_y == '' else transform_y,
            alpha=None if alpha == '' else float(alpha.replace('_', '.')),
            l1_ratio=None if l1_ratio == '' else float(l1_ratio.replace('_', '.')),
            n_estimators=None if n_estimators == '' else int(n_estimators),
            max_depth=None if max_depth == '' else int(max_depth),
            max_features=(
                None if max_features == '' else
                max_features if max_features in ModelSpec.allowed_values_max_features_str else
                int(max_features) if '_' not in max_features else
                float(max_features.replace('_', '.'))
            )
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
            self.n_estimators == other.n_estimators and
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
            self.n_estimators,
            self.max_depth,
            self.max_features,
        ))


class TestModelSpec(unittest.TestCase):
    def test_construction_naive(self):
        tests = (
            ('n', 1),
        )
        for test in tests:
            name, n_trades_back = test
            model_spec = ModelSpec(
                name=name,
                n_trades_back=n_trades_back,
            )
            self.assertTrue(isinstance(model_spec, ModelSpec))
            self.assertEqual(model_spec.name, name)
            s = str(model_spec)
            model_spec2 = ModelSpec.make_from_str(s)
            self.assertEqual(model_spec, model_spec2)

    def test_construction_elasticnet_good(self):
        tests = (
            ('en', 1, None, None, 1.23, 0.5),
            ('en', 2, 'log1p', 'log', 0.00001, 0.9999),
        )
        for test in tests:
            name, n_trades_back, transform_x, transform_y, alpha, l1_ratio = test
            print 'test_construction_elasticnet_good', transform_y
            model_spec = ModelSpec(
                name=name,
                n_trades_back=n_trades_back,
                transform_x=transform_x,
                transform_y=transform_y,
                alpha=alpha,
                l1_ratio=l1_ratio,
            )
            self.assertTrue(isinstance(model_spec, ModelSpec))
            self.assertEqual(model_spec.name, name)
            self.assertEqual(model_spec.transform_x, transform_x)
            self.assertEqual(model_spec.transform_y, transform_y)
            self.assertEqual(model_spec.alpha, alpha)
            self.assertEqual(model_spec.l1_ratio, l1_ratio)
            s = str(model_spec)
            model_spec2 = ModelSpec.make_from_str(s)
            self.assertEqual(model_spec, model_spec2)

    def test_construction_elasticnet_bad(self):
        tests = (
            ('en', 1, None, None, 1.23, -1.0),
            ('en', 2, None, None, None, 0.1),
            ('en', 3, None, 'log1p', 1.23, 0.1),
            ('en', 4, 'log', None, 1.23, 0.1),
            ('en', 0, None, None, 1.23, 0.1),
        )
        for test in tests:
            name, n_trades_back, transform_x, transform_y, alpha, l1_ratio = test
            args = {
                'name': name,
                'n_trades_back': n_trades_back,
                'transform_x': transform_x,
                'transform_y': transform_y,
                'alpha': alpha,
                'la1_ratio': l1_ratio,
            }
            self.assertRaises(Exception, ModelSpec, [], args)

    def test_construction_randomforests_good(self):
            tests = (
                ('rf', 1, 123, 10, 20),
                ('rf', 2, 123, None, 20.1),
                ('rf', 3, 123, None, 'auto'),
                ('rf', 4, 123, None, 'sqrt'),
                ('rf', 5, 123, None, 'log2'),
                ('rf', 6, 123, None, None),
                ('rf', 6, 0, None, None),
            )
            for test in tests:
                name, n_trades_back, n_estimators, max_depth, max_features = test
                model_spec = ModelSpec(
                    name=name,
                    n_trades_back=n_trades_back,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                )
                self.assertTrue(isinstance(model_spec, ModelSpec))
                self.assertEqual(model_spec.name, name)
                self.assertEqual(model_spec.n_estimators, n_estimators)
                self.assertEqual(model_spec.max_depth, max_depth)
                self.assertEqual(model_spec.max_features, max_features)
                s = str(model_spec)
                model_spec2 = ModelSpec.make_from_str(s)
                self.assertEqual(model_spec, model_spec2)

    def test_construction_randomforests_bad(self):
        tests = (
            ('rf', 1, 10, None, 'bad'),
            ('rf', 2, 10, 1.0, None),
            ('rf', 3, 1.0, 1, None),
            ('rf', 0, 10, 1, None),
        )
        for test in tests:
            name, n_trades_back, n_estimators, max_features, max_depth = test
            args = {
                'name': name,
                'n_trades_back': n_trades_back,
                'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
            }
            self.assertRaises(Exception, ModelSpec, [], args)


if __name__ == '__main__':
    unittest.main()
