'''messages published and subscribed to only within the machine learning subsystem'''
import copy
import json
import pdb
import typing
import unittest

import features
import machine_learning
import shared_message


class Test(shared_message.Message):
    def __init__(self,
                 source: str,
                 identifier: str,
                 feature_vector: features.FeatureVector,
                 ):
        self._super = super(Test, self)
        self._super.__init__('Test', source, identifier)
        
        self.feature_vector = copy.copy(feature_vector)

    def __repr__(self):
        return self._super.__repr__(
            message_name='Test',
            other_fields='feature_vector=%s' % (
                str(self.feature_vector),
                ),
            )

    def __str__(self):
        return json.dumps(self.as_dict())

    def as_dict(self):
        result = self._super.as_dict()
        result['feature_vector'] = self.feature_vector
        return result

    @staticmethod
    def from_dict(d: dict):
        return Test(
            source=d['source'],
            identifier=d['identifier'],
            feature_vector=d['feature_vector'],
            )

    
class Train(shared_message.Message):
    def __init__(self,
                 source: str,
                 identifier: str,
                 feature_vectors: typing.List[features.FeatureVector],
                 ):
        self._super = super(Train, self)
        self._super.__init__('Train', source, identifier)
        
        self.feature_vectors = copy.copy(feature_vectors)

    def __repr__(self):
        return self._super.__repr__(
            message_name='Train',
            other_fields='|feature_vector|=%d' % (
                len(self.feature_vectors),
                ),
            )

    def __str__(self):
        return json.dumps(self.as_dict())

    def as_dict(self):
        result = self._super.as_dict()
        result['feature_vectors'] = self.feature_vectors
        return result

    @staticmethod
    def from_dict(d: dict):
        return Train(
            source=d['source'],
            identifier=d['identifier'],
            feature_vectors=d['feature_vectors'],
            )
        
    
###################################################
# construct from string
###################################################
def from_string(s: str):
    obj = json.loads(s)
    assert isinstance(obj, dict)
    message_type = obj['message_type']
    if message_type == 'Test':
        return Test.from_dict(obj)
    elif message_type == 'Train':
        return Train.from_dict(obj)
    else:
        return shared_message.from_string(s)

    
#####################################################
# Unit Tests
#####################################################
class Tester(unittest.TestCase):
    def setUp(self):
        self.feature_vector1 = features.FeatureVector({
            'id_1': 1,
            'value_1': 1.0,
            })
        self.feature_vector2 = features.FeatureVector({
            'id_1': 2,
            'value_1': 2.0,
            })
        self.feature_vectors = [
            self.feature_vector1,
            self.feature_vector2,
            ]

    def test_Test(self):
        debug = False
        set_trace = machine_learning.make_set_trace(debug)
        vp = machine_learning.make_verbose_print(debug)

        set_trace()

        source = 'testing'
        identifier = '123'

        x = Test(
            source=source,
            identifier=identifier,
            feature_vector=self.feature_vector1,
            )
        vp(x)
        xx = from_string(str(x))
        vp(xx)
        self.assertTrue(isinstance(x, Test))
        self.assertTrue(isinstance(xx, Test))
        self.assertEqual(xx.source, source)
        self.assertEqual(xx.identifier, identifier)
        vp(xx.feature_vector)
        self.assertEqual(xx.feature_vector, self.feature_vector1)

    def test_Train(self):
        debug = False
        set_trace = machine_learning.make_set_trace(debug)
        vp = machine_learning.make_verbose_print(debug)

        set_trace()

        source = 'testing'
        identifier = '123'

        x = Train(
            source=source,
            identifier=identifier,
            feature_vectors=self.feature_vectors,
            )
        vp(x)
        xx = from_string(str(x))
        vp(xx)
        self.assertTrue(isinstance(x, Train))
        self.assertTrue(isinstance(xx, Train))
        self.assertEqual(xx.source, source)
        self.assertEqual(xx.identifier, identifier)
        vp(xx.feature_vectors)
        self.assertEqual(xx.feature_vectors, self.feature_vectors)

        
if __name__ == '__main__':
    unittest.main()
