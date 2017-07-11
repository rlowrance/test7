'''type written to output files by program fit_predict.property

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''
import collections


OutputKey = collections.namedtuple(
    'OutputKey',
    'trace_index model_spec',
)


Prediction = collections.namedtuple(
    'Prediction',
    'effectivedatetime trade_type quantity interarrival_seconds actual prediction',

)

Importance = collections.namedtuple(
    'Importance',
    'effectivedatetime trade_type importance'
)
