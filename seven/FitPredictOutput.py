'''fit and predict all models on one CUSIP feature file

INVOCATION
  python fit-predict.py {ticker} {cusip} {--test} {--trace}

EXAMPLES OF INVOCATION
 python fit-predict.py orcl 68389XAS4

INPUTS
 WORKING/features/{cusip}.csv
 WORKING/features/{ticker}-{cusip}.csv  read to implement checkpoint restart # TODO: make this happen

OUTPUTS
 WORKING/fit-predict/{ticker}-{cusip}.pickle  file containing predictions for each fitted model
  # TODO: change from current file name to file name written above
  The file is a sequence of records, each record a tuple:
  (model_spec, original_print_file_index,
   actual_B, predicted_B, actual_D, predicted_D, actual_S, predicted_S,
  )
 WORKING/fit-predict/{ticker}-{cusip}-importances.pickle
where
  model_spec is a string specifying both the model family (naive, en, rf) and its hyperparameters
'''

from __future__ import division

import pdb
from pprint import pprint


class FitPredictOutput(object):
    def __init__(
        self,
        query_index=None,
        model_spec=None,
        trade_type=None,
        predicted_value=None,
        actual_value=None,
        importances=None
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


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
