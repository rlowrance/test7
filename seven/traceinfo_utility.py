import collections
import cPickle as pickle
import os
import pdb

import path  # seven.path

TraceInfo = collections.namedtuple(
    'TraceInfo',
    'issuer cusip issuepriceid effective_date effective_datetime'
)


def read_by_trace_index(issuer):
    path_to_file = os.path.join(path.working(), 'traceinfo', issuer, 'by_trace_index.pickle')
    with open(path_to_file, 'rb') as f:
        obj = pickle.load(f)
    return obj


def read_summary(issuer):
    path_to_file = os.path.join(path.working(), 'traceinfo', issuer, 'summary.pickle')
    with open(path_to_file, 'rb') as f:
        obj = pickle.load(f)
    return obj
