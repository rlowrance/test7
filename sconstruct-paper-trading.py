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
import datetime
import os
import pdb
import pprint

import seven.build
import seven.GetBuildInfo

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


# dt = datetime.date

# Info = collections.namedtuple('Info', 'issuer first_date last_date')


# cusip_infos = {
#     '037833AG5': Info('AAPL', dt(2017, 5, 1), dt(2017, 6, 26)),
#     # '037833AG5': Info('AAPL', dt(2017, 6, 1), dt(2017, 6, 1)),
#     # '68389XAS4': Info('ORCL', dt(2016, 11, 1), dt(2016, 11, 2), "test"),  # test this scons file
#     # '68389XAC9': Info('ORCL', dt(2016, 9, 1), dt(2016, 12, 31), "dell"),   # requests from Katrina in email
#     # '68389XBJ3': Info('ORCL', dt(2016, 6, 29), dt(2017, 3, 16), "alien"),
#     # '68389XBC8': Info('ORCL', dt(2015, 4, 28), dt(2017, 3, 16), "mac"),
#     # '68389XBM6': Info('ORCL', dt(2016, 6, 29), dt(2017, 3, 16), "dell"),
#     # '68389XAG0': Info('ORCL', dt(2012, 1, 4), dt(2017, 3, 16), "done"),
#     # '68389XAK1': Info('ORCL', dt(2012, 1, 3), dt(2017, 3, 16), "mac"),
#     # '68389XAM7': Info('ORCL', dt(2012, 1, 3), dt(2017, 3, 16), "mac"),
#     # '68389XBA2': Info('ORCL', dt(2014, 6, 30), dt(2017, 3, 16), "dell"),
#     # '68389XBD6': Info('ORCL', dt(2015, 4, 28), dt(2017, 3, 16), "alien"),
#     # '68389XBE4': Info('ORCL', dt(2015, 4, 28), dt(2017, 3, 16), "alien"),
#     # '68389XBK0': Info('ORCL', dt(2016, 6, 29), dt(2017, 3, 16), "dell"),
# }


# def select_cusip_from_info_where_run_id_equals(run_id):
#     'return set of cusips'
#     result = {
#         cusip
#         for cusip, info in info.iteritems()
#         if info.run_id == run_id
#     }
#     return result


# def select_ticker_from_info_where_run_id_equals(run_id):
#     'return set of tickers'
#     result = {
#         info.ticker
#         for cusip, info in info.iteritems()
#         if info.run_id == run_id
#     }
#     return result


# def select_ticker_from_info_where_cusip_equals(selected_cusip):
#     'return set of tickers'
#     result = {
#         info.ticker
#         for cusip, info in info.iteritems()
#         if cusip == selected_cusip
#     }
#     return result


# def to_str(dt):
#     'convert datetime.date to str'
#     result = '%4d-%02d-%02d' % (dt.year, dt.month, dt.day)
#     return result


# def effective_dates_for_cusip(cusip, info):
#     'return list of dates (:str) to be fed to features_targets'
#     next_date = info.first_date
#     last_date = info.last_date
#     result = []
#     while next_date <= last_date:
#         result.append(to_str(next_date))
#         next_date += datetime.timedelta(days=1)
#     return result


# def hpset_for(cusip):
#     return 'grid3'


def commands_for():
    'call command(*args) for appropriate targets'
    # for now, just work with one CUSIP and hence one issuer
    issuers = ('AAPL',)
    issuer_cusips = {
        'AAPL': ('037833AG5',),
    }
    fit_dates = ('2017-06-26',)
    hpset = 'grid4'

    # buildinfo.py
    for issuer in ('AAPL',):
        command(seven.build.buildinfo, issuer)

    # features_targets.py
    for issuer, cusips in issuer_cusips.iteritems():
        for cusip in cusips:
            for fit_date in fit_dates:
                command(seven.build.features_targets, issuer, cusip, fit_date)

    # fit.py
    for issuer in issuers:
        gbi = seven.GetBuildInfo.GetBuildInfo(issuer)
        cusips_for_build = set(issuer_cusips[issuer])
        for fit_date in fit_dates:
            # determine issuepriceids for the cusip on the fit_date
            issuepriceids = gbi.get_issuepriceids(fit_date)
            cusips = map(lambda x: gbi.get_cusip(x), issuepriceids)
            relevant_issuepriceids_cusips = filter(lambda x: x[1] in cusips_for_build, zip(issuepriceids, cusips))
            for issuepriceid_cusip in relevant_issuepriceids_cusips:
                issuepriceid, cusip = issuepriceid_cusip
                command(seven.build.fit, issuer, cusip, str(issuepriceid), hpset)


# main program
commands_for()
