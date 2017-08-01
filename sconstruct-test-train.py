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
# cons -n -f sconstruct-features_targets.py /
# cons -f sconstruct-master.py / run_id=[alien|dell|mac|testt]
# cons -f sconstruct-master.py / cusip=68389XAC9
# cons -f sconstruct-master.py / just_reports

import collections
import cPickle as pickle
import datetime
import os
import pdb
import pprint

import seven.build
import seven.exception
import seven.EventId
import seven.EventInfo
import seven.path

pp = pprint.pprint
pdb

dir_home = os.path.join('C:', r'\Users', 'roylo')
dir_dropbox = os.path.join(dir_home, 'Dropbox')
dir_working = os.path.join(dir_dropbox, 'data', '7chord', '7chord-01', 'working')
dir_midpredictor_data = os.path.join(dir_dropbox, 'MidPredictor', 'data')

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


def as_datetime_date(s):
    year, month, day = s.split('-')
    return datetime.date(int(year), int(month), int(day))

# Katrina:
# We are suggesting the following CUSIPs for Go-Live: (Roy: these are in order)
# ORCL 68389XAU9
# ORCL 68389XAS4
# IBM 459200HU8
# your AAPL Cusip
# AMZN 023135AN6


Dates = collections.namedtuple('Dates', 'first_features first_ensemble last_ensemble')


def make_cusips(prefix, suffixes):
    return [
        prefix + suffix
        for suffix in suffixes
    ]


issuer_cusips_1 = {
    'AAPL': make_cusips('037833A', ['J9'])
}
issuer_cusips_all = {  # the tickers and cusips are identified in the file secmaster.csv
    'AAPL':
        make_cusips('037833A', [
            'G5', 'H5', 'J9', 'K6', 'L4', 'M2', 'N0', 'P5', 'Q3', 'R1',
            'S9', 'T7', 'W0', 'X8', 'Y6', 'Z3',
        ]) +
        make_cusips('037833B', [
            'A7', 'B5', 'C3', 'D1', 'E9', 'F6', 'G4', 'H2', 'N9', 'Q2', 
            'R0', 'S8', 'T6', 'U3', 'W9', 'X7', 'Y5', 'Z2',
        ]) +
        make_cusips('037833C', [
            'A6', 'B4', 'C2', 'D0', 'E8', 'F5', 'G3', 'H1', 'J7', 'K4',
            'L2', 'M0', 'N8', 'P3', 'Q1', 'R9', 'S7', 'T5', 'U2', 'X6',
        ]),
    'AMZN': 
        make_cusips('023135', [
            'AH5', 'AJ5', 'AK2', 'AL0', 'AM8', 'AN6', 'AP1', 'AQ9',
        ]),
    'CSCO':
        make_cusips('17275RA', [
            'C6', 'D4', 'E3', 'F9', 'G7', 'H5', 'J1', 'K8', 'N2', 'P7',
            'Q5', 'R3', 'S1', 'T9', 'U6', 'V4', 'W2', 'X0', 'Y8', 'Z5', 
        ]) +
        make_cusips('17275RB', [
            'A9', 'B7', 'C5', 'D3', 'E1', 'G6', 'H4', 'J0', 'K7', 'L5', 
        ]),
    'GOOGL': 
        make_cusips('02079KA', [
            'A5', 'B3', 'C1',
        ]) +
        make_cusips('38259PA', [
            'B8', 'D4',
        ]),
    'IBM':
        make_cusips('459200A', [
            'G6', 'L5', 'M3', 'N1', 'R2', 'S0',
        ]) +
        make_cusips('459200G', [
            'J4', 'L9', 'M7', 'N5', 'R6', 'S4', 'T2', 'U9', 'W5', 'X3',
            'Z8',
        ]) +
        make_cusips('459200H', [
            'A2', 'B0', 'C8', 'D6', 'E4', 'F1', 'G9', 'K0', 'L8', 'M6',
            'P9', 'T1', 'U8', 'W4', 'X2', 'Z7',
        ]) + 
        make_cusips('459200J', [
            'A0', 'C6', 'D4', 'E2', 'F9', 'G7', 'H5', 'N2', 'P7', 'Q5', 
            'R3',
        ]) +
        make_cusips('459200Q', [
            'DY7',
        ]),
    'MSFT': 
        make_cusips('594918A', [
            'B0', 'C8', 'D6', 'F1', 'G9', 'H7', 'J3', 'K0', 'L8', 'M6',
            'P9', 'Q7', 'R5', 'S3', 'T1', 'U8', 'V6', 'W4', 'X2', 'Y0',
        ]) +
        make_cusips('594918B', [
            'A1', 'B9', 'C7', 'D5', 'E3', 'F0', 'G8', 'H6', 'J2', 'K9',
            'L7', 'M5', 'N3', 'P8', 'Q6', 'R4', 'S2', 'T0', 'U7', 'V5',
            'W3', 'X1', 'Y9', 'Z6', 
        ]) +
        make_cusips('594918C', [
            'A0', 'B8',
        ]),
    'ORCL': 
        make_cusips('68389XA', [
            'C9', 'E5', 'F2', 'G0', 'H8', 'J4', 'K1', 'L9', 'M7', 'N5', 'P0', 'Q8', 'R6', 'S4', 'T2',
            'U9', 'V7', 'W5', 'X3', 'Y1',
        ]) +
        make_cusips('68389XB', [
            'A2', 'B0', 'C8', 'D6', 'E4', 'F1', 'G9', 'H7', 'J3', 'K0', 'L8', 'M6', 
        ]) +
        make_cusips('68402LA', [
            'C8'
        ]),
}
issuer_cusips = issuer_cusips_1
# dates = {}
# for issuer in issuer_cusips.keys():
#     dates[issuer] = Dates(
#         first_features='2017-07-14',
#         first_ensemble='2017-07-21',
#         last_ensemble='2017-07-21',
#     )


Control = collections.namedtuple(
    'Control',
    'first_feature_date fit_dates predict_dates trading_date_before ensemble_dates',
)

# NOTE: fit and every date there is a prediction
control = Control(
    first_feature_date=datetime.date(2017, 6, 30),  # needed for AAPL 037833AJ9 fitted on 07-19  
    fit_dates=[
        # datetime.date(2017, 7, 13),
        datetime.date(2017, 7, 18),  # 18 ==> Tue
        datetime.date(2017, 7, 19),  # 19 ==> Wed
        # datetime.date(2017, 7, 20),
        # datetime.date(2017, 7, 21),
        ],      # 14 ==> Friday
    predict_dates=[
        # datetime.date(2017, 7, 14),
        # datetime.date(2017, 7, 18),  # Tue
        datetime.date(2017, 7, 19),  # Wed
        # datetime.date(2017, 7, 20),  # Thu
        # datetime.date(2017, 7, 21),
        ],
    trading_date_before={
        datetime.date(2017, 7, 19): datetime.date(2017, 7, 18),
        datetime.date(2017, 7, 20): datetime.date(2017, 7, 19),
    },
    ensemble_dates=[
        datetime.date(2017, 7, 20),  # Thu
    ],
)


hpset = 'grid4'


def get_issuers(maybe_specific_issuer):
    'yield sequence of issuer of interest'
    if maybe_specific_issuer is None:
        for issuer in issuer_cusips.iterkeys():
            yield issuer
    else:
        assert maybe_specific_issuer in issuer_cusips
        yield maybe_specific_issuer


def date_range(first, last):
    'yield consecutive dates in [first, last]'
    assert first <= last
    current = first
    while current <= last:
        yield current
        current += datetime.timedelta(1)  # 1 day


def ensemble_dates(issuer):
    'yield each date in [dates[issuer].first_ensemble, dates[issuer].last_ensemble]'
    first_ensemble_date = as_datetime_date(dates[issuer].first_ensemble)
    last_ensemble_date = as_datetime_date(dates[issuer].last_ensemble)
    for date in date_range(first_ensemble_date, last_ensemble_date):
        yield date


def predict_dates(issuer):
    'yield each date that the experts should make predictions on'
    # one day preceeding what the ensemble model needs
    first_predict_date = as_datetime_date(dates[issuer].first_ensemble) - datetime.timedelta(1)
    last_predict_date = as_datetime_date(dates[issuer].last_ensemble) - datetime.timedelta(1)
    for date in date_range(first_predict_date, last_predict_date):
        yield date


def last_date(dates):
    result = None
    for date in dates:
        if result is None:
            result = date
        elif date > result:
            result = date
    return result


def commands_for_sort_trace_file(maybe_specific_issuer, invoke_with_debug):
    for issuer in get_issuers(maybe_specific_issuer):
        print 'evaluate sort_trace_file.py %s' % issuer
        command(
            seven.build.sort_trace_file,
            issuer,
            debug=invoke_with_debug,
        )

##############################################################################################
# main program
##############################################################################################

def invocation_error(msg=None):
    if msg is not None:
        print 'ERROR: %s' % msg
    print 'ERROR: must specify what=[build | features | fit | predict | accuracy | ensemble | predictions] on invocation'
    print 'predictions implies running sequentially with fit > predict > accuracy > ensemble'
    Exit(2)


what = ARGUMENTS.get('what', None)
maybe_specific_issuer = ARGUMENTS.get('issuer', None)
invoke_with_debug = ARGUMENTS.get('debug', True)
# TODO: devise a way to not invoke with debug

if what == 'None':
    invocation_error()
elif what == 'sort_trace_file':
    commands_for_sort_trace_file(maybe_specific_issuer, invoke_with_debug)
else:
    invocation_error('what=%s is not a recognized invocation option' % what)    
