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


def commands_for():
    'call command(*args) for appropriate targets'
    # for now, just work with one CUSIP and hence one issuer
    issuers = ('AAPL',)
    issuer_cusips = {
        'AAPL': ('037833AG5',),
    }
    feature_dates = ('2017-06-23', '2017-06-26', '2017-06-27')
    fit_dates = feature_dates[:-1]  # don't fit the last date on which we create features
    hpset = 'grid4'

    # buildinfo.py
    # traceinfo.py
    for issuer in ('AAPL',):
        command(seven.build.traceinfo, issuer) 

    with open(seven.build.traceinfo(issuer)['out_summary'], 'rb') as f:
        traceinfos = pickle.load(f)  # a list of dict

    with open(seven.build.traceinfo(issuer)['out_by_issuer_cusip'], 'rb') as f:
        by_issuer_cusip = pickle.load(f)  # Dict[(issuer,cusip), List[Dict]]

    # features_targets.py
    for issuer, cusips in issuer_cusips.iteritems():
        for cusip in cusips:
            for feature_date in feature_dates:
                command(seven.build.features_targets, issuer, cusip, feature_date)

    # fit.py
    for issuer in issuers:
        for cusip in issuer_cusips[issuer]:
            infos = by_issuer_cusip[(issuer, cusip)]
            for fit_date_str in fit_dates:
                fit_date = as_datetime_date(fit_date_str)

                def select_issuepriceids(info):
                    return (
                        info['effective_date'] == fit_date
                    )

                issuepriceids = filter(select_issuepriceids, infos)
                for issuepriceid in issuepriceids:
                    # print 'fit', issuer, cusip, issuepriceid.issuepriceid, hpset
                    command(seven.build.fit, issuer, cusip, str(issuepriceid['issuepriceid']), hpset)

    # predict.py for back-testing purposes
    prediction_issuer = 'AAPL'
    prediction_cusip = '037833AG5'
    infos = by_issuer_cusip[(prediction_issuer, prediction_cusip)]

    prediction_date = datetime.date(2017, 6, 26)
    predict_trades = filter(
        lambda info: info['effective_date'] == prediction_date,
        infos
    )

    for predict_trade in predict_trades:
        previous_trades = filter(
            lambda info: info['effective_datetime'] < predict_trade['effective_datetime'],
            predict_trades
        )
        sorted_previous_trades = sorted(previous_trades, key=lambda info: info['effective_datetime'], reverse=True)
        for previous_trade in sorted_previous_trades:
            def one_trade_at_effectivedatetime(info):
                at_effectivedatetime = filter(
                    lambda x: x['effective_datetime'] == info['effective_datetime'],
                    traceinfos,
                )
                return len(at_effectivedatetime) == 1

            if one_trade_at_effectivedatetime(previous_trade):
                command(
                    seven.build.predict,
                    'AAPL',
                    str(predict_trade['issuepriceid']),
                    str(previous_trade['issuepriceid']),
                )
                break

    # accuracy.py
    accuracy_issuer = 'AAPL'
    accuracy_cusip = '037833AG5'
    accuracy_dates = ('2017-06-26',)
    for accuracy_date in accuracy_dates:
        command(seven.build.accuracy, accuracy_issuer, accuracy_cusip, accuracy_date)


# main program
commands_for()
