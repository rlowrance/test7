'''
Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''
# invocations:
#   scons -f sconstruct.py /
#   scons -n -f sconstruct.py /
#   scons --debug=explain -f sconstruct.py /
#   scons -j <nprocesses> -f sconstruct.py /
#   scons -j <nprocesses> -f sconstruct.py / what=build
#   scons -j <nprocesses> -f sconstruct.py / what=features
#   scons -j <nprocesses> -f sconstruct.py / what=predictions

# where / means to build everything (not just stuff in the current working directory .)

# examples:
# scons -n -f sconstruct-test-train.py /  prediction_date=2017-08-18 -j 16 -n

import collections
import csv
import os
import pdb
import pprint

import seven.build
import seven.exception
import seven.path

pp = pprint.pprint
pdb

env = Environment(
    ENV=os.environ,
)

env.Decider('MD5-timestamp')  # if timestamp out of date, examine MD5 checksum


def command(*args, **kwargs):
    make_paths = args[0]
    other_args = args[1:]
    scons = seven.build.make_scons(make_paths(*other_args, **kwargs))
    if False:
        print 'command targets', scons['targets']
        print 'command sources', scons['sources']
        print 'command commands', scons['commands']
    env.Command(
        scons['targets'],
        scons['sources'],
        scons['commands'],
    )


# deteremine issuer cusips of interest
# these are in the secmaster file
path_secmaster = seven.path.input(
    issuer=None,
    logical_name='security master',
)
issuer_cusip = collections.defaultdict(list)
n_cusips = 0
with open(path_secmaster) as f:
    dict_reader = csv.DictReader(f)
    for row in dict_reader:
        issuer_cusip[row['ticker']].append(row['CUSIP'])
        n_cusips += 1
print 'found %d issuers which all together had %d cusips' % (
    len(issuer_cusip),
    n_cusips,
)

prediction_date = ARGUMENTS.get('prediction_date', None)
assert prediction_date is not None

for issuer in issuer_cusip.keys():
    for cusip in issuer_cusip[issuer]:
        target = 'oasspread'
        hpset = 'grid5'
        start_events = '2017-04-01'
        start_predictions = prediction_date
        stop_predictions = prediction_date
        print 'evaluate test_train.py %s %s %s %s %s %s %s' % (
            issuer,
            cusip,
            target,
            hpset,
            start_events,
            start_predictions,
            stop_predictions,
        )
        command(
            seven.build.test_train,
            issuer,
            cusip,
            target,
            hpset,
            start_events,
            start_predictions,
            stop_predictions,
            debug=True,
        )
