# invocations:
#   scons -f sconstruct.py /
#   scons -n -f sconstruct.py /
#   scons --debug=explain -f sconstruct.py /
#   scons -j <nprocesses> -f sconstruct.py /

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
# 459200HU8
# your AAPL Cusip
# 023135AN6

def commands_for():
    'call command(*args) for appropriate targets'
    # for now, just work with one CUSIP and hence one issuer
    CD = collections.namedtuple('CD', 'first_features fit_dates')
    issuer_cusips = {
        # 'AAPL': ['037833AG5'],
        'AMZN': [],
        'CSCO': [],
        'GOOGL': [],
        'MSFT': [],
        'ORCL': ['68389XAU9'],
    }
    cusip_data = {
        '037833AG5': CD('2017-06-09', ('2017-06-26',)),
        '68389XAU9': CD('2017-06-15', ('2017-07-05',)),
    }
    first_features = {
        '037833AG5': '2017-06-09',  # for fitting on 2017-06-26
        '68389XAU9': '2017-06-15',  # a guess, for fitting on 2017-07-05
    }
    prediction_dates = {  # NOTE: must be in increasing date order!
        '037833AG5': ['2017-06-26'],
        '68389XAU9': ['2017-07-05'],
    }
    hpset = 'grid4'
    # date for which to create features and targets
    # the oldest such date we need for the select CUSIP is 2017-06-09, to fit trades on 2017-06-26
    # For AAPL/037833AG5 fitted on 2017006-26, the earliest features needed are on 2017-06-
    # feature_dates = [  # dates for which to create features and targets
    #     '2017-06-%02d' % d
    #     for d in range(9, 28)  # days 09 .. 27
    # ]
    # fit_dates = ('2017-06-26',)  # fit  on this Monday, predict on Tuesday
    hpset = 'grid4'

    # traceinfo.py
    for issuer, cusips in issuer_cusips.iteritems():
        command(seven.build.traceinfo, issuer)

    if False:
        print 'truncated after traceinfo.py'
        return

    # NOTE: read in the previous version of the traceinfo.
    # FIXME: read the version of the traceinfo created by executing the above command
    traceinfo_all = {}  # Dict[cusip, traceinfo]
    traceinfo_by_issuer_cusip = {}  # Dict[cusip, Dict[(issuer, cusip)]]
    for issuer in issuer_cusips.keys():
        with open(seven.build.traceinfo(issuer)['out_summary'], 'rb') as f:
            traceinfo_all[issuer] = pickle.load(f)  # a list of dict

        with open(seven.build.traceinfo(issuer)['out_by_issuer_cusip'], 'rb') as f:
            traceinfo_by_issuer_cusip[issuer] = pickle.load(f)  # Dict[(issuer,cusip), List[Dict]]

    # features_targets.py
    for issuer, cusips in issuer_cusips.iteritems():
        for cusip in cusips:
            cd = cusip_data[cusip]
            first_feature_date = as_datetime_date(cd.first_features)
            last_feature_date = as_datetime_date(cd.fit_dates[-1])
            current_feature_date = first_feature_date
            while current_feature_date <= last_feature_date:
                command(seven.build.features_targets, issuer, cusip, current_feature_date)
                current_feature_date += datetime.timedelta(1)  # 1 day

    if False:
        print 'truncated after features_targets.py'
        return

    def infos_for_issuer_cusip(issuer, cusip):
        return traceinfo_by_issuer_cusip[issuer][(issuer, cusip)]

    def infos_for_trades_on(issuer, cusip, date_arg):
        date = as_datetime_date(date_arg) if isinstance(date_arg, str) else date_arg
        return filter(
            lambda info: info['effective_date'] == date,
            infos_for_issuer_cusip(issuer, cusip),
        )

    def infos_for_trades_at(issuer, cusip, dt):
        return filter(
            lambda info: info['effective_datetime'] == dt,
            infos_for_issuer_cusip(issuer, cusip),
        )

    def n_trades_at(issuer, cusip, dt):
        return len(infos_for_trades_at(issuer, cusip, dt))

    for issuer in issuer_cusips.keys():
        for cusip in issuer_cusips[issuer]:
            cd = cusip_data[cusip]
            for fit_date in cd.fit_dates:
                infos_on_fit_date = infos_for_trades_on(issuer, cusip, fit_date)
                print 'scons fit.py: # trades for %s %s on %s: %d' % (issuer, cusip, fit_date, len(infos_on_fit_date))
                for info in infos_on_fit_date:

                    command(seven.build.fit, issuer, cusip, str(info['issuepriceid']), hpset)

    if False:
        print 'truncated after fit.py'
        return

    for issuer in issuer_cusips.keys():
        for cusip in issuer_cusips[issuer]:
            for prediction_date in prediction_dates[cusip]:
                predict_trades = infos_for_trades_on(issuer, cusip, prediction_date)
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
                        n_trades = n_trades_at(issuer, cusip, previous_trade['effective_datetime'])
                        print '    previous trade issueprice id %s at datetime %s; %d trades at that datetime' % (
                            previous_trade['issuepriceid'],
                            previous_trade['effective_datetime'],
                            n_trades,
                        )
                        if n_trades == 1:
                            command(
                                seven.build.predict,
                                issuer,
                                str(predict_trade['issuepriceid']),
                                str(previous_trade['issuepriceid']),
                            )
                            break  # stop the search
                        # continue to search backwards in time

    if True:
        print 'truncated after predict.py'
        return

    # accuracy.py
    accuracy_issuer = 'AAPL'
    accuracy_cusip = '037833AG5'
    accuracy_dates = ('2017-06-26',)
    for accuracy_date in accuracy_dates:
        command(seven.build.accuracy, accuracy_issuer, accuracy_cusip, accuracy_date)

    if False:
        print 'truncated after accuracy.py'
        return

    # ensemble_predictions.py
    ensemble_issuer = 'AAPL'
    ensemble_cusip = '037833AG5'
    ensemble_dates = ('2017-06-27',)
    for ensemble_date in ensemble_dates:
        command(seven.build.ensemble_predictions, ensemble_issuer, ensemble_cusip, ensemble_date)

# main program
commands_for()
