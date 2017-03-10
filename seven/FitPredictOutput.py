import pdb
import unittest


import applied_data_science.framework.FitPredictOutput


class FitPredictOutput(applied_data_science.framework.FitPredictOutput.FitPredictOutput):
    def __init__(
        self,
        query_index=None,
        model_spec=None,
        trade_type=None,
        predicted_value=None,
        actual_value=None,
        importances=None,
        n_training_samples=None,
    ):
        args_passed = locals().copy()

        def test(feature_name):
            value = args_passed[feature_name]
            if value is None:
                raise ValueError('%s cannot be None' % feature_name)
            else:
                return value

        self.query_index = test('query_index')
        self.model_spec = test('model_spec')
        self.trade_type = test('trade_type')
        self.predicted_value = test('predicted_value')
        self.actual_value = test('actual_value')
        self.importances = importances  # will be None, when the method doesn't provide importances
        self.n_training_samples = test('n_training_samples')

    def as_dict(self):
        return {
            'query_index': self.query_index,
            'model_spec': self.model_spec,
            'trade_type': self.trade_type,
            'predicted_value': self.predicted_value,
            'actual_value': self.actual_value,
            'importances': self.importances,
            'n_training_samples': self.n_training_samples,
        }


class Test(unittest.TestCase):
    def test_construction(self):
        x = FitPredictOutput(
            query_index=1,
            model_spec=2,
            trade_type=3,
            predicted_value=4,
            actual_value=5,
            importances=6,
            n_training_samples=7,
        )
        self.assertTrue(isinstance(x, FitPredictOutput))
        d = x.as_dict()
        self.assertEqual(4, d['predicted_value'])


if __name__ == '__main__':
    unittest.main()
    if False:
        # avoid pyflakes warnings
        pdb
