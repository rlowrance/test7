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
    first_feature_date=datetime.date(2017, 7, 1),  # needed for AAPL 037833AJ9 fitted on 07-19  
    fit_dates=[
        # datetime.date(2017, 7, 13),
        datetime.date(2017, 7, 19),  # 19 ==> Wed
        datetime.date(2017, 7, 20),
        # datetime.date(2017, 7, 21),
        ],      # 14 ==> Friday
    predict_dates=[
        # datetime.date(2017, 7, 14),
        # datetime.date(2017, 7, 18),  # Tue
        # datetime.date(2017, 7, 19),  # Wed
        datetime.date(2017, 7, 20),  # Thu
        # datetime.date(2017, 7, 21),
        ],
    trading_date_before={
        datetime.date(2017, 7, 19): datetime.date(2017, 7, 18),
        datetime.date(2017, 7, 20): datetime.date(2017, 7, 19),
    },
    ensemble_dates=[
        datetime.date(2017, 7, 21),
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


class TraceInfo(object):
    def __init__(self):
        self.infos_by_issuer = {}
        pass

    def _initialize(self, issuer):
        'mutate self.infos_by_issuer to hold all of the trace info for the issuer'
        scons = seven.build.traceinfo(issuer)
        for k, v in scons.iteritems():
            # we use just one of the files created by traceinfo.py
            # NOTE: build.py may use other output files created by traceinfo.py
            if k == 'out_by_issuer_cusip':
                with open(v, 'rb') as f:
                    obj = pickle.load(f)
                if issuer not in self.infos_by_issuer:
                    self.infos_by_issuer[issuer] = {}
                self.infos_by_issuer[issuer][k] = obj

    def infos_for_trades_on(self, issuer, cusip, the_date):
        'return list of infos on the date for the issuer-cusip'
        self._initialize(issuer)
        by_issuer = self.infos_by_issuer[issuer]
        by_issuer_cusip = by_issuer['out_by_issuer_cusip']
        infos = by_issuer_cusip[(issuer, cusip)]  # : List[info]
        result = [
            info
            for info in infos
            if info['effective_date'] == the_date
        ]
        return result

    def infos_for_trades_before(self, issuer, cusip, the_datetime):
        self._initialize(issuer)
        by_issuer = self.infos_by_issuer[issuer]
        by_issuer_cusip = by_issuer['out_by_issuer_cusip']
        infos = by_issuer_cusip[(issuer, cusip)]  # : List[info]
        result = [
            info
            for info in infos
            if info['effective_datetime'] < the_datetime
        ]
        return result

    def n_trades_on(self, issuer, cusip, the_date):
        return len(self.infos_for_trades_on(issuer, cusip, the_date))

    def n_trades_at(self, issuer, cusip, the_datetime):
        'yield the info on the date'
        self._initialize(issuer)
        by_issuer = self.infos_by_issuer[issuer]
        by_issuer_cusip = by_issuer['out_by_issuer_cusip']
        infos = by_issuer_cusip[(issuer, cusip)]  # : List[info]
        count = 0
        for info in infos:
            if info['effective_datetime'] == the_datetime:
                count += 1
        return count


trace_info = TraceInfo()


def commands_for_build():
    'issue command to build the build information'
    # traceinfo.py
    for issuer in issuer_cusips:
        print 'scons traceinfo.py', issuer
        command(seven.build.traceinfo, issuer)

    # buildinfo.py
    print 'scons buildinfo.py'
    command(seven.build.buildinfo)


def commands_for_features(maybe_specific_issuer, invoke_with_debug):
    'issue commands to build the feature sets'
    for issuer in get_issuers(maybe_specific_issuer):
        for cusip in issuer_cusips[issuer]:
            # build feature sets from the first feature date through the last predict date
            last_feature_date = max(
                max(control.fit_dates),
                max(control.predict_dates),
                max(control.ensemble_dates),
            )
            for effective_date in date_range(control.first_feature_date, last_feature_date):
                effective_date_str = '%s' % effective_date
                print 'evalute features_targets.py', issuer, cusip, effective_date_str
                command(
                    seven.build.features_targets,
                    issuer,
                    cusip,
                    effective_date_str,
                    debug=invoke_with_debug,
                    )


def commands_for_fit(maybe_specific_issuer, invoke_with_debug):
    'issue commands to fit the models'
    for issuer in get_issuers(maybe_specific_issuer):
        for cusip in issuer_cusips[issuer]:
            target = 'oasspread'
            for current_date in control.fit_dates:
                for event_id in EventInfo.features_on_date(issuer, cusip, current_date):
                    hpset = 'grid4'
                    print 'evaluate fit.py', issuer, cusip, target, event_id, hpset
                    command(
                        seven.build.fit,
                        issuer,
                        cusip,
                        target,
                        event_id,
                        hpset,
                        debug=invoke_with_debug,
                        )


def commands_for_predict(maybe_specific_issuer, invoke_with_debug):
    'issue commands to predict queries using the fitted models'
    # given a prediction event, fit it to an earlier event that occured at a distinct time
    # both must have the same reclassified trade type
    for issuer in get_issuers(maybe_specific_issuer):
        for cusip in issuer_cusips[issuer]:
            event_info = seven.EventInfo.EventInfo(issuer, cusip)
            for prediction_date in control.predict_dates:
                for prediction_event_id in event_info.on_date(prediction_date):
                    prediction_reclassified_trade_type = event_info.reclassified_trade_type(prediction_event_id)
                    # find the oldest prior event_id such that
                    #  there is only 1 trade and that time
                    #  the reclassified trade type is the same as the prediction event trade type
                    fitted_event_id = None
                    for prior_datetime in event_info.prior_datetimes(prediction_event_id.datetime()):
                        # the prior_datetimes are oldest, the second oldest, ...
                        prior_event_ids = event_info.at_datetime(prior_datetime)
                        if len(prior_event_ids) == 1:
                            prior_event_id = prior_event_ids[0]
                            if event_info.reclassified_trade_type(prior_event_id) == prediction_reclassified_trade_type:
                                fitted_event_id = prior_event_id
                                break
                    if fitted_event_id is None:
                        print 'unable to find a suitable fitted_event_id'
                        pdb.set_trace()

                    target = 'oasspread'
                    print 'evaluate predict.py', issuer, cusip, target, prediction_event_id, fitted_event_id
                    command(
                        seven.build.predict,
                        issuer,
                        cusip,
                        target,
                        str(prediction_event_id),
                        str(fitted_event_id),
                        debug=invoke_with_debug,
                    )                    

                    # This is the code for the ensemble model
                    # predict using the model fitted on the last trade of the prior day
                    # unless there were multiple events at the same time
                    # prior_date = control.tradeing_date_before[prediction_event_id.datetime()]
                    # events_on_prior_date = EventId.features_on_date(issuer, cusip, prior_date)
                    # count_at_datetime = collections.Counter()
                    # event_at_datetime = {}
                    # for event in events_on_prior_date:
                    #     count_at_datetime[event.datetime()] += 1
                    #     event_at_datetime[event.datetime()] = event
                    # pdb.set_trace()
                    # sorted_datetimes = sorted(count.keys(), reverse=True)
                    # fitted_event_id = None
                    # for dt in sorted_datetimes:
                    #     if count_at_datetime[dt] == 1:
                    #         fitted_event_id = event_at_datetime[dt]
                    #         break
                    # if fitted_event_id is None:
                    #     print 'no transactions to fit'
                    #     print 'try prior day'
                    #     pdb.set_trace()
   
                    # target = 'oasspread'
                    # print 'evaluate predict.py', issuer, cusip, target, prediction_event_id, fitted_event_id
                    # command(
                    #     seven.build.predict,
                    #     issuer,
                    #     cusip,
                    #     target,
                    #     prediction_event_id,
                    #     fitted_event_id,
                    #     debug=invoke_with_debug,
                    # )
                    

def commands_for_accuracy(maybe_specific_issuer):
    'issue commands to determine accuracy of the predictions'
    for issuer in get_issuers(maybe_specific_issuer):
        for cusip in issuer_cusips[issuer]:
            for prediction_date in control.predict_dates:
            # for prediction_date in predict_dates(issuer):
                if trace_info.n_trades_on(issuer, cusip, prediction_date) > 0:
                    print 'evaluatee accuracy.py', issuer, cusip, prediction_date
                    command(seven.build.accuracy, issuer, cusip, str(prediction_date))


def commands_for_ensemble_predictions(maybe_specific_issuer):
    'issue commands to predict using the predictions of the experts'
    for issuer in get_issuers(maybe_specific_issuer):
        for cusip in issuer_cusips[issuer]:
            for ensemble_date in control.ensemble_dates:
            # for ensemble_date in ensemble_dates(issuer):
                print 'scons ensemble_predictions', issuer, cusip, ensemble_date
                command(seven.build.ensemble_predictions, issuer, cusip, str(ensemble_date))


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
elif what == 'build':
    commands_for_build()
elif what == 'features':
    commands_for_features(maybe_specific_issuer, invoke_with_debug)
elif what == 'fit':
    commands_for_fit(maybe_specific_issuer, invoke_with_debug)
elif what == 'predict':
    commands_for_predict(maybe_specific_issuer, invoke_with_debug)
elif what == 'accuracy':
    commands_for_accuracy(maybe_specific_issuer, invoke_with_debug)
elif what == 'ensemble':
    commands_for_ensemble_predictions(maybe_specific_issuer, invoke_with_debug)
elif what == 'predictions':
    functions = (
        commands_for_fit,
        commands_for_predict,
        commands_for_accuracy,
        commands_for_ensemble_predictions,
    )
    for f in functions:
        f(maybe_specific_issuer)
else:
    invocation_error('what=%s is not a recognized invocation option' % what)    
