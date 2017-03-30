import pdb
import numbers
import unittest


import applied_data_science.timeseries as timeseries


class Id(object):
    def __init__(self, query_index=None, model_spec=None, predicted_feature_name=None):
        args_passed = locals().copy()

        assert isinstance(query_index, numbers.Number)
        assert isinstance(model_spec, timeseries.ModelSpec)
        assert isinstance(predicted_feature_name, str)

        self.query_index = query_index
        self.model_spec = model_spec
        self.predicted_feature_name = predicted_feature_name

    def __str__(self):
        return 'query_index %s model_spec %s predicted_feature_name %s' % (
            self.query_index,
            self.model_spec,
            self.predicted_feature_name,
        )

    def __hash__(self):
        return hash((self.query_index, self.model_spec, self.predicted_feature_name))

    def __eq__(self, other):
        return (
            self.query_index == other.query_index and
            self.model_spec == other.model_spec and
            self.predicted_feature_name == other.predicted_feature_name
        )

    def as_dict(self):
        return {
            'query_index': self.query_index,
            'model_spec': self.model_spec,
            'predicted_feature_name': self.predicted_feature_name,
        }


class Payload(object):
    def __init__(
        self,
        predicted_feature_value=None,
        actual_value=None,
        importances=None,
        n_training_samples=None,
    ):
        assert isinstance(predicted_feature_value, numbers.Number)
        assert isinstance(actual_value, numbers.Number)
        assert isinstance(importances, dict)
        assert isinstance(n_training_samples, numbers.Number)

        self.predicted_feature_value = predicted_feature_value
        self.actual_value = actual_value
        self.importances = importances
        self.n_training_samples = n_training_samples

    def __str__(self):
        return 'predicted %s actual %s' % (
            self.predicted_feature_value,
            self.actual_value,
        )

    def as_dict(self):
        return {
            'predicted_feature_value': self.predicted_feature_value,
            'actual_value': self.actual_value,
            'importances': self.importances,
            'n_training_samples': self.n_training_samples,
        }


class Record(timeseries.FitPredictOutput):
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
        result.update(self.payload.as_dict)
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
