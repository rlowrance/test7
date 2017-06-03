# invocations:
#   scons -f sconstruct.py /
#   scons -n -f sconstruct.py /
#   scons --debug=explain -f sconstruct.py /
#   scons -j <nprocesses> -f sconstruct.py /

# where / means to build everything (not just stuff in the current working directory .)

# examples:
# cons -n -f sconstruct-master.py /
# cons -f sconstruct-master.py /

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


info = {
    '68389XAS4': Info('ORCL', dt(2016, 11, 1), dt(2016, 11, 2), "test"),  # test this scons file
    '68389XAC9': Info('ORCL', dt(2012, 1, 3), dt(2017, 3, 16), "dell"),   # requests from Katrina in email
    '68389XBJ3': Info('ORCL', dt(2016, 6, 29), dt(2017, 3, 16), "alien"),
    '68389XBC8': Info('ORCL', dt(2015, 4, 28), dt(2017, 3, 16), "mac"),
    '68389XBM6': Info('ORCL', dt(2016, 6, 29), dt(2017, 3, 16), "dell"),
    '68389XAG0': Info('ORCL', dt(2012, 1, 4), dt(2017, 3, 16), "done"),
    '68389XAK1': Info('ORCL', dt(2012, 1, 3), dt(2017, 3, 16), "mac"),
    '68389XAM7': Info('ORCL', dt(2012, 1, 3), dt(2017, 3, 16), "mac"),
    '68389XBA2': Info('ORCL', dt(2014, 6, 30), dt(2017, 3, 16), "dell"),
    '68389XBD6': Info('ORCL', dt(2015, 4, 28), dt(2017, 3, 16), "alien"),
    '68389XBE4': Info('ORCL', dt(2015, 4, 28), dt(2017, 3, 16), "alien"),
    '68389XBK0': Info('ORCL', dt(2016, 6, 29), dt(2017, 3, 16), "dell"),
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
    'return set of cusips'
    result = {
        info.ticker
        for cusip, info in info.iteritems()
        if info.run_id == run_id
    }
    return result


def to_str(dt):
    'convert datetime.date to str'
    result = '%4d-%02d-%02d' % (dt.year, dt.month, dt.day)
    return result


def effective_dates_for_cusip(cusip):
    'return list of dates (:str) to be fed to fit_predict'
    cusip_info = info[cusip]
    next_date = cusip_info.first_date
    last_date = cusip_info.last_date
    result = []
    while next_date <= last_date:
        result.append(to_str(next_date))
        next_date += datetime.timedelta(days=1)
    return result


def hpset_for(cusip):
    result = 'grid1' if cusip == '68389XAS4' else 'grid3'
    return result


def commands_for(run_id):
    for ticker in select_ticker_from_info_where_run_id_equals(run_id):
        command(seven.build.cusips, ticker)
        for cusip in select_cusip_from_info_where_run_id_equals(run_id):
            hpset = hpset_for(cusip)
            for effective_date in effective_dates_for_cusip(cusip):
                command(seven.build.fit_predict, ticker, cusip, hpset, effective_date)
            command(seven.build.report03_compare_predictions, ticker, cusip, hpset)
            command(seven.build.report04_predictions, ticker, cusip, hpset)
            n = 49 if hpset == 'grid1' else 100  # number of best models cannot exceed size of hpset
            command(seven.build.report05_compare_importances, ticker, cusip, hpset, n)


# main program
commands_for('dell')
