# invocations:
#   scons -f sconstruct.py /
#   scons -n -f sconstruct.py /
#   scons --debug=explain -f sconstruct.py /
#   scons -j <nprocesses> -f sconstruct.py /

# where / means to build everything (not just stuff in the current working directory .)
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


# main program
tickers = ['ORCL']
cusips = ['68389XBE4']

# dates for the cusip
days_in_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
years = (2015, 2016, 2017)
dates = []
for year in years:
    # see cusips-ORCL for the range of trade dates for cusip 68389XAS4
    if year == 2015:
        months = (4, 5, 6, 7, 8, 9, 10, 11, 12)
    elif year == 2016:
        months = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    else:
        months = (1, 2, 3)
    for month in months:
        for day in xrange(days_in_month[month]):
            dates.append('%d-%02d-%02d' % (year, month, day + 1))

for ticker in tickers:
    command(seven.build.cusips, ticker)
    for cusip in cusips:  # just one cusip, for now
        for hpset in ['grid3']:
            for effective_date in dates:
                command(seven.build.fit_predict, ticker, cusip, hpset, effective_date)
            command(seven.build.report03_compare_models, ticker, cusip, hpset)
            command(seven.build.report04_predictions, ticker, cusip, hpset)
