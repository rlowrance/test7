'''time series prediction

Copyright 2017 Roy E. Lowrance

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on as "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing premission and
limitation under the license.
'''
from abc import ABCMeta, abstractmethod
import collections
import pdb
import numbers
import numpy as np
import unittest


class CreateFeatures(object):
    'create features file from a master file and possibly many associated information files'
    def __init__(self):
        self.n_input_records = None
        self.n_output_records = None
        self.skipped = None  # collections.Counter for reasons input records were skipped
        pass

    def create(
        self,
        feature_makers=None,
        master_file_records=None,
        selected=None,   # lambda index, master_record -> (use_record: Bool, maybe_error_message)
        report_skipped_master_record=None,  # lambda index, master_record, [None|feature_maker], msg -> None
        # start optional arguments
        verbose=False,
    ):
        'yield sequence (features:Dict, index, master_record) of output records with features derived from the master file'
        def error(msg):
            print('error in feature maker %s feature_name %s feature_value %s' % (
                feature_maker.name,
                feature_name,
                feature_value,
            ))
            print(msg)
            print('entering pdb')
            pdb.set_trace()

        assert feature_makers is not None
        assert master_file_records is not None
        assert selected is not None
        assert report_skipped_master_record is not None
        self.n_input_records = 0
        self.n_output_records = 0
        self.skipped = collections.Counter()
        for index, master_record in master_file_records:
            self.n_input_records += 1
            if self.n_input_records % 10000 == 1:
                print('creating features from master record %d index %s' % (self.n_input_records, index))
            (use_record, msg) = selected(index, master_record)
            if not use_record:
                report_skipped_master_record(index, master_record, None, msg)
                continue
            # create features from the master_record
            # the feature makers may incorporate data from other records
            features_made = {}
            stopped_early = False
            # accumulate all the features from the feature makers
            # check for errors on the way
            for feature_maker in feature_makers:
                maybe_features = feature_maker.make_features(index, master_record)
                if isinstance(maybe_features, str):
                    report_skipped_master_record(index, master_record, feature_maker, maybe_features)
                    stopped_early = True
                    break
                elif isinstance(maybe_features, dict):
                    for feature_name, feature_value in maybe_features.items():
                        if feature_name in features_made:
                            error('duplicate feature name')
                        elif not feature_name.startswith('id_') and not isinstance(feature_value, numbers.Number):
                            error('feature value is not numeric')
                        elif feature_name.endswith('_size') and feature_value < 0.0:
                            error('size feature is negative')
                        else:
                            features_made[feature_name] = feature_value
                else:
                    print(feature_maker.name)
                    print(feature_maker)
                    print(maybe_features)
                    print(type(maybe_features))
                    error('unexpected return type from a feature_maker')
            if stopped_early:
                continue
            self.n_output_records += 1
            yield features_made, index, master_record


class FeatureMaker(object, metaclass=ABCMeta):
    def __init__(self, name=None):
        self.name = name  # used in error message; informal name of the feature maker

    @abstractmethod
    def make_features(ticker_index, tickercusip, ticker_record):
        'return errors:str or Dict[feature_name:str, feature_value:nmber]'
        pass


FitPredictResult = collections.namedtuple(
    'FitPredictResult',
    (
        'query_index',
        'query_features',
        'predicted_feature_name',
        'predicted_feature_value',
        'model_spec',
        'prediction', 
        'fitted_model',
        'n_training_samples',
    )
)


class ExceptionFit(Exception):
    def __init__(self, parameter):
        self.parameter = parameter

    def __str__(self):
        return 'ExceptionFit(%s)' % str(self.parameter)


class FitPredict(object):
    'fit models and predict targets'

    def fit_predict(
        self,
        df_features=None,
        df_targets=None,
        make_model=None,
        model_specs=None,
        timestamp_feature_name=None,
        already_seen_lambda=None,    # lambda query_index, model_spec, predicted_feature_name: Bool
    ):
        'yield either (True, result:FitPredictResult) or (False, error_msg:str)'
        # df_targets: a sorted sequence of targets, sorted in timestamp order (y values)
        sorted_targets = df_targets.sort_values(by=timestamp_feature_name)
        for query_index in sorted_targets.index:
            if query_index not in df_features.index:
                yield False, 'no query feature for query index %s' % query_index
                continue
            query_features = df_features.loc[[query_index]]  # must be a DataFrame
            assert len(query_features) == 1
            timestamp = query_features.iloc[0][timestamp_feature_name]
            mask = sorted_targets[timestamp_feature_name] < timestamp
            training_targets = sorted_targets.loc[mask]
            if len(training_targets) == 0:
                yield False, 'no training_targets for query index %s timestamp %s' % (query_index, timestamp)
                continue
            training_features = df_features.loc[training_targets.index]
            if len(training_features) == 0:
                yield False, 'no training_features for query index %s timestamp %s' % (query_index, timestamp)
                continue
            for predicted_feature_name, predicted_feature_value in sorted_targets.loc[query_index].items():
                if predicted_feature_name.startswith('id_'):
                    continue  # skip identifiers, as these are not features
                if predicted_feature_name.endswith('_decreased') or predicted_feature_name.endswith('_increased'):
                    yield False, 'classification not yet implemented; target feature name %s' % predicted_feature_name
                    continue
                for model_spec in model_specs:
                    if already_seen_lambda(query_index, model_spec, predicted_feature_name):
                        yield False, 'already seen: %s %s %s' % (query_index, model_spec, predicted_feature_name)
                        continue
                    m = make_model(model_spec, predicted_feature_name)
                    try:
                        # TODO: turn into keywords
                        m.fit(training_features, training_targets)
                    except ExceptionFit as e:
                        yield False, 'exception raised during fitting: %s' % str(e)
                        continue
                    predictions = m.predict(query_features)
                    assert len(predictions) == 1
                    prediction = predictions[0]
                    if np.isnan(prediction):
                        print('prediction is NaN', prediction)
                        print(model_spec)
                        pdb.set_trace()
                    if prediction is None:
                        yield False, 'predicted value was None: %s %s %s' % (query_index, model_spec, predicted_feature_name)
                    else:
                        yield (
                            True,
                            FitPredictResult(
                                query_index=query_index,
                                query_features=query_features,
                                predicted_feature_name=predicted_feature_name,
                                predicted_feature_value=predicted_feature_value,
                                model_spec=model_spec,
                                prediction=predictions[0],
                                fitted_model=m,
                                n_training_samples=len(training_features),
                            )
                        )


class FitPredictOutput(object, metaclass=ABCMeta):
    'content of output file for program fit_predict.py'

    @abstractmethod
    def as_dict(self):
        'return a dict with all the fields'
        pass


class HpChoices(object, metaclass=ABCMeta):
    'iterated over HpSpec instances'

    @abstractmethod
    def __iter__(self):
        'yield sequence of HpSpec objects'
        pass


class Model(object, metaclass=ABCMeta):
    @abstractmethod
    def fit(self, df_training_samples_features, df_training_samples_targets):
        'mutate self; set attribute importances: Dict[feature_name:str, feature_importance:Number]'
        pass

    @abstractmethod
    def predict(self, df_query_samples_features):
        'return predictions'
        pass


class ModelSpec(object, metaclass=ABCMeta):
    'specification of a model name and its associated hyperparamters'

    @abstractmethod
    def __str__(self):
        'return parsable string representation'
        # Hint: Use method self._to_str(value) to convert individual values to strings
        # That will make all the string representations use the same encoding of values to strings
        pass

    @staticmethod
    @abstractmethod
    def make_from_str(s):
        'parse the representation returned by str(self) to create an instance'
        pass

    @abstractmethod
    def iteritems(self):
        'yield each (hyparameter name:str, hyperparameter value)'
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __lt__(self, other):
        pass

    def _to_str(self, value):
        'internal method. Convert value to a string. Use me in your __str__ method'
        def remove_trailing_zeroes(s):
            return (
                s if s[-1] != '0' else
                remove_trailing_zeroes(s[:-1])
            )
        if value is None:
            return ''
        elif isinstance(value, float):
            return remove_trailing_zeroes(('%f' % value).replace('.', '_'))
        elif isinstance(value, int):
            return '%d' % value
        else:
            return str(value)


class TestHpChoices(unittest.TestCase):
    def test_construction(self):
        self.assertRaises(Exception, HpChoices, None)


class TestHpSpeC(unittest.TestCase):
    def test_construction(self):
        self.assertRaises(Exception, HpSpec, None)


if __name__ == '__main__':
    unittest.main()
