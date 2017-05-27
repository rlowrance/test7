'''type written to output files by program fit_predict.property'''
import collections


OutputKey = collections.namedtuple(
    'OutputKey',
    'trace_index model_spec',
)


Prediction = collections.namedtuple(
    'Prediction',
    'effectivedatetime trade_type quantity actual prediction',

)

Importance = collections.namedtuple(
    'Importance',
    'effectivedatetime trade_type importance'
)
