'''create information ("side info") needed to control the build process

Read each ticker file and update the files association information with CUSIPs.

NOTE: This program looks at all trades for the CUSIP and related OTR CUSIPs from the
beginning of time. Thus trades 10 years ago are used to predict next oasspreads today.
That is way too much history. We should fix this when we have a streaming infrastructure.
Most likely, only the last 1000 or so trades are relevant.

INVOCATION
  python features_targets.py {cusip} {effective_date} {--test} {--trace}
where
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 effective_date: YYYY-MM-DD is the date of the trade
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATIONS
  python buildinfo.py ORCL

See build.py for input and output files.

IDEA FOR FUTURE:
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import datetime
import os
import pdb
from pprint import pprint
import random
import sys

import applied_data_science.debug
import applied_data_science.dirutility
import applied_data_science.lower_priority
import applied_data_science.pickle_utilities

from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.arg_type
import seven.build
import seven.feature_makers
import seven.fit_predict_output
import seven.read_csv
import seven.target_maker

pp = pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('issuer', type=seven.arg_type.issuer)
    # parser.add_argument('cusip', type=seven.arg_type.cusip)
    # parser.add_argument('effective_date', type=seven.arg_type.date)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')

    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    path = seven.build.buildinfo(arg.issuer, test=arg.test)
    applied_data_science.dirutility.assure_exists(path['dir_out'])

    return Bunch(
        arg=arg,
        path=path,
        random_seed=random_seed,
        timer=Timer(),
    )


def sort_by_effectivedatetime(df):
    'return new DataFrame in sorted order'
    # Note: the following statement works but generates a SettingWithCopyWarning
    df['effectivedatetime'] = seven.feature_makers.make_effectivedatetime(df)
    result = df.sort_values('effectivedatetime')
    return result


def read_and_transform_trace_prints(issuer, test):
    'return sorted DataFrame containing trace prints for the cusip and all related OTR cusips'
    trace_prints = seven.read_csv.input(
        issuer,
        'trace',
        nrows=1000 if test else None,
    )
    # make sure the trace prints are in non-decreasing datetime order
    sorted_trace_prints = sort_by_effectivedatetime(trace_prints)
    sorted_trace_prints['issuepriceid'] = sorted_trace_prints.index  # make the index one of the values
    return sorted_trace_prints


def read_pickle(path):
    'read pickle file, if it exists; return (obj_in_it, err)'
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj, False
    else:
        return None, 'file does not exist'


def write_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print 'wrote pickled object of type %s to path %s' % (type(obj), path)


def do_work(control):
    'write predictions from fitted models to file system'
    def lap():
        'return ellapsed wall clock time:float since previous call to lap()'
        return control.timer.lap('lap', verbose=False)[1]

    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    # input files are for a specific ticker
    trace_prints = read_and_transform_trace_prints(
        control.arg.issuer,
        control.arg.test,
    )
    print 'read and transformed %d trace prints in %.3f wall clock seconds' % (len(trace_prints), lap())

    issuers, err = read_pickle(control.path['out_issuers'])
    if err is not None:
        issuers = collections.defaultdict(set)

    counter = collections.Counter()

    def count(s):
        counter[s] += 1

    def skip(reason):
        # print 'skipped', reason
        count('skipped: ' + reason)

    # accumulate info for cusips
    for trace_index, trace_record in trace_prints.iterrows():
        count('n trace records read')
        cusip = trace_record['cusip']
        issuers[cusip].add(control.arg.issuer)  # instead could parse nasdsymbol
        continue

    print 'end loop on input in %.3f wall clock seconds' % lap()
    print 'counts'
    for k in sorted(counter.keys()):
        print '%71s: %6d' % (k, counter[k])

    print 'issuers by cusip'
    for k, v in issuers.iteritems():
        if len(v) != 1:
            print 'cusip %s has multiple issuers: %s' % (k, v)
        else:
            for v_value in v:
                print k, v_value

    write_pickle(issuers, control.path['out_issuers'])
    return None


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    # print control
    print control.arg
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
        datetime

    main(sys.argv)
