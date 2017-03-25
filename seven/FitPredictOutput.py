import pdb
import numbers
import unittest


import applied_data_science.framework.FitPredictOutput

from ModelSpec import ModelSpec


class Id(object):
    def __init__(self, target_index=None, model_spec=None, predicted_feature=None):
        args_passed = locals().copy()

        assert isinstance(target_index, numbers.Number)
        assert isinstance(model_spec, ModelSpec)
        assert isinstance(predicted_feature, str)

        self.target_index = target_index
        self.model_spec = model_spec
        self.predicted_feature = predicted_feature

    def __str__(self):
        return 'target_index %s model_spec %s predicted_feature %s' % (
            self.target_index,
            self.model_spec,
            self.predicted_feature,
        )

    def as_dict(self):
        return {
            'target_index': self.target_index,
            'model_spec': self.model_spec,
            'predicted_feature': self.predicted_feature,
        }


class Payload(object):
    def __init__(
        self,
        predicted_value=None,
        actual_value=None,
        importances=None,
        n_training_samples=None,
    ):
        assert isinstance(predicted_value, numbers.Number)
        assert isinstance(actual_value, numbers.Number)
        assert isinstance(importances, dict)
        assert isinstance(n_training_samples, numbers.Number)

        self.predicted_value = predicted_value
        self.actual_value = actual_value
        self.importances = importances
        self.n_training_samples = n_training_samples

    def __str__(self):
        return 'predicted %s actual %s' % (
            self.predicted_value,
            self.actual_value,
        )

    def as_dict(self):
        return {
            'predicted_value': self.predicted_value,
            'actual_value': self.actual_value,
            'importances': self.importances,
            'n_training_samples': self.n_training_samples,
        }


class Record(applied_data_science.framework.FitPredictOutput.FitPredictOutput):
    'record in pickle file'
    def __init__(
        self,
        id=None,
        payload=None,
    ):
        assert isinstance(id, Id)
        assert isinstance(payload, Payload)

        self.id = id
        self.payload = payload

    def __str__(self):
        return '%s %s' % (
            self.id,
            self.payload,
        )

    def as_dict(self):
        result = self.id.as_dict()
        for k, v in self.payload.as_dict():
            result[k] = v
        return result


class Test(unittest.TestCase):
    def test_construction(self):
        x = Record(
            id=Id(
                target_index=1,
                model_spec=ModelSpec(name='n'),
                predicted_feature='feature1',
            ),
            payload=Payload(
                predicted_value=3,
                actual_value=5,
                importances={
                    'feature1': 1.0,
                },
                n_training_samples=6,
            )
        )
        self.assertTrue(isinstance(x, Record))
        self.assertEqual(x.id.target_index, 1)
        self.assertEqual(x.payload.n_training_samples, 6)


if __name__ == '__main__':
    unittest.main()
    if False:
        # avoid pyflakes warnings
        pdb
