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


def command(*args):
    make_paths = args[0]
    other_args = args[1:]
    scons = seven.build.make_scons(make_paths(*other_args))
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
issuer_cusips = {
    # 'AAPL': ['037833AG5'],
    'AMZN': ['023135AN6'],
    'CSCO': [],
    'GOOGL': [],
    'MSFT': [],
    'ORCL': [
        '68389XAU9',
        '68389XAS4',
        ],
}
dates = {}
for issuer in issuer_cusips.keys():
    dates[issuer] = Dates(
        first_features='2017-06-12',
        first_ensemble='2017-07-06',
        last_ensemble='2017-07-12',
    )
# dates['APPL'] = Dates(
#         first_features='2017-06-09',
#         first_ensemble='2017-06-26',
#         last_ensemble='2017-07-07',
# )


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


def commands_for_features(maybe_specific_issuer):
    'issue commands to build the feature sets'
    for issuer in get_issuers(maybe_specific_issuer):
        for cusip in issuer_cusips[issuer]:
            # build feature sets from first date to the last ensemble date
            first_feature_date = as_datetime_date(dates[issuer].first_features)
            last_ensemble_date = as_datetime_date(dates[issuer].last_ensemble)
            for current_date in date_range(first_feature_date, last_ensemble_date):
                current_date_str = '%s' % current_date
                print 'scons features_targets.py', issuer, cusip, current_date_str
                command(seven.build.features_targets, issuer, cusip, current_date_str)


def commands_for_fit(maybe_specific_issuer):
    'issue commands to fit the models'
    for issuer in get_issuers(maybe_specific_issuer):
        for cusip in issuer_cusips[issuer]:
            for current_date in predict_dates(issuer):
                for info in trace_info.infos_for_trades_on(issuer, cusip, current_date):
                    issuepriceid = info['issuepriceid']
                    print 'scons fit.py', issuer, cusip, issuepriceid, hpset, ' # on date: %s' % current_date
                    command(seven.build.fit, issuer, cusip, issuepriceid, hpset)


def commands_for_predict(maybe_specific_issuer):
    'issue command to predict queries using the fitted models'
    verbose = False
    for issuer in get_issuers(maybe_specific_issuer):
        for cusip in issuer_cusips[issuer]:
            for prediction_date in predict_dates(issuer):
                predict_trades = trace_info.infos_for_trades_on(issuer, cusip, prediction_date)
                if verbose:
                    print 'scons predict.py: num trades for %s %s on %s: %d' % (
                        issuer,
                        cusip,
                        prediction_date,
                        len(predict_trades),
                    )
                for predict_trade in predict_trades:
                    previous_trades = filter(
                        lambda info: info['effective_datetime'] < predict_trade['effective_datetime'],
                        predict_trades
                    )
                    if verbose:
                        print '  num previous trades for issuepriceid %s date %s: %d' % (
                            predict_trade['issuepriceid'],
                            predict_trade['effective_date'],
                            len(previous_trades),
                        )
                    sorted_previous_trades = sorted(
                        previous_trades,
                        key=lambda info: info['effective_datetime'],
                        reverse=True,
                    )
                    for previous_trade in sorted_previous_trades:
                        n_trades = trace_info.n_trades_at(issuer, cusip, previous_trade['effective_datetime'])
                        if verbose:
                            print '    previous trade issueprice id %s at datetime %s; %d trades at that datetime' % (
                                previous_trade['issuepriceid'],
                                previous_trade['effective_datetime'],
                                n_trades,
                            )
                        if n_trades == 1:
                            print 'scons predict.py %s %s %s # on date: %s' % (
                                issuer,
                                str(predict_trade['issuepriceid']),
                                str(previous_trade['issuepriceid']),
                                prediction_date,
                            )
                            command(
                                seven.build.predict,
                                issuer,
                                str(predict_trade['issuepriceid']),
                                str(previous_trade['issuepriceid']),
                            )
                            break  # stop the search
                        # continue to search backwards in time


def commands_for_accuracy(maybe_specific_issuer):
    'issue commands to determine accuracy of the predictions'
    for issuer in get_issuers(maybe_specific_issuer):
        for cusip in issuer_cusips[issuer]:
            for prediction_date in predict_dates(issuer):
                if trace_info.n_trades_on(issuer, cusip, prediction_date) > 0:
                    print 'scons accuracy.py', issuer, cusip, prediction_date
                    command(seven.build.accuracy, issuer, cusip, str(prediction_date))


def commands_for_ensemble_predictions(maybe_specific_issuer):
    'issue commands to predict using the predictions of the experts'
    for issuer in get_issuers(maybe_specific_issuer):
        for cusip in issuer_cusips[issuer]:
            for ensemble_date in ensemble_dates(issuer):
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

if what == 'None':
    invocation_error()
elif what == 'build':
    commands_for_build()
elif what == 'features':
    commands_for_features(maybe_specific_issuer)
elif what == 'fit':
    commands_for_fit(maybe_specific_issuer)
elif what == 'predict':
    commands_for_predict(maybe_specific_issuer)
elif what == 'accuracy':
    commands_for_accuracy(maybe_specific_issuer)
elif what == 'ensemble':
    commands_for_ensemble_predictions(maybe_specific_issuer)
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
