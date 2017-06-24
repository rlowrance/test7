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


dt = datetime.date

Info = collections.namedtuple('Info', 'ticker first_date last_date run_id')


infos = {
    # '68389XAS4': Info('ORCL', dt(2016, 11, 1), dt(2016, 11, 2), "test"),  # test this scons file
    '68389XAC9': Info('ORCL', dt(2016, 9, 1), dt(2016, 12, 31), "dell"),   # requests from Katrina in email
    # '68389XBJ3': Info('ORCL', dt(2016, 6, 29), dt(2017, 3, 16), "alien"),
    # '68389XBC8': Info('ORCL', dt(2015, 4, 28), dt(2017, 3, 16), "mac"),
    # '68389XBM6': Info('ORCL', dt(2016, 6, 29), dt(2017, 3, 16), "dell"),
    # '68389XAG0': Info('ORCL', dt(2012, 1, 4), dt(2017, 3, 16), "done"),
    # '68389XAK1': Info('ORCL', dt(2012, 1, 3), dt(2017, 3, 16), "mac"),
    # '68389XAM7': Info('ORCL', dt(2012, 1, 3), dt(2017, 3, 16), "mac"),
    # '68389XBA2': Info('ORCL', dt(2014, 6, 30), dt(2017, 3, 16), "dell"),
    # '68389XBD6': Info('ORCL', dt(2015, 4, 28), dt(2017, 3, 16), "alien"),
    # '68389XBE4': Info('ORCL', dt(2015, 4, 28), dt(2017, 3, 16), "alien"),
    # '68389XBK0': Info('ORCL', dt(2016, 6, 29), dt(2017, 3, 16), "dell"),
}


def select_cusip_from_info_where_run_id_equals(run_id):
    'return set of cusips'
    result = {
        cusip
        for cusip, info in info.iteritems()
        if info.run_id == run_id
    }
    return result


def select_ticker_from_info_where_run_id_equals(run_id):
    'return set of tickers'
    result = {
        info.ticker
        for cusip, info in info.iteritems()
        if info.run_id == run_id
    }
    return result


def select_ticker_from_info_where_cusip_equals(selected_cusip):
    'return set of tickers'
    result = {
        info.ticker
        for cusip, info in info.iteritems()
        if cusip == selected_cusip
    }
    return result


def to_str(dt):
    'convert datetime.date to str'
    result = '%4d-%02d-%02d' % (dt.year, dt.month, dt.day)
    return result


def effective_dates_for_cusip(cusip):
    'return list of dates (:str) to be fed to fit_predict'
    cusip_info = infos[cusip]
    next_date = cusip_info.first_date
    last_date = cusip_info.last_date
    result = []
    while next_date <= last_date:
        result.append(to_str(next_date))
        next_date += datetime.timedelta(days=1)
    return result


def hpset_for(cusip):
    return 'grid3'


def commands_for(target_selector):
    'call command(*args) for appropriate targets'
    for cusip, info in infos.iteritems():
        def select(*args):
            if target_selector(cusip, info, *args):
                # print 'command', args
                command(*args)

        ticker = info.ticker
        issuer = ticker  # some program use one name, some the other
        select(seven.build.cusips, ticker)
        hpset = hpset_for(cusip)
        for effective_date in effective_dates_for_cusip(cusip):
            select(seven.build.features_targets, issuer, cusip, effective_date)
        # select(seven.build.report03_compare_predictions, ticker, cusip, hpset)
        # select(seven.build.report04_predictions, ticker, cusip, hpset)
        # n = 49 if hpset == 'grid1' else 100  # number of best models cannot exceed size of hpset
        # select(seven.build.report05_compare_importances, ticker, cusip, hpset, n)


# main program
def select_all_targets(cusip, info, *args):
    return True


commands_for(select_all_targets)
# print 'ARGUMENTS', ARGUMENTS

# arg_run_id = ARGUMENTS.get('run_id', None)
# if arg_run_id is not None:
#     print 'arg_run_id', arg_run_id
#     run_id = 'test' if arg_run_id is None else arg_run_id

#     def select_run_id(cusip, info, *args):
#         return arg_run_id == info.run_id

#     commands_for(select_run_id)

# arg_cusip = ARGUMENTS.get('cusip', None)
# if arg_cusip is not None:

#     def select_cusip(cusip, info, *args):
#         return cusip == arg_cusip

#     commands_for(select_cusip)

# arg_just_reports = ARGUMENTS.get('just_reports', None)
# if arg_just_reports is not None:
#     all_report_builds = [
#             seven.build.report03_compare_predictions,
#             seven.build.report04_predictions,
#             seven.build.report05_compare_importances,
#     ]

#     def select_just_reports(cusip, info, *args):
#         args_build = args[0]
#         return args_build in all_report_builds

#     commands_for(select_just_reports)

