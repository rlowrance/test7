# invocations:
#   scons -f SConstruct.py /
#   scons -n -f SConstruct.py /
#   scons --debug=Explain -f SConstruct.py /

# where / means to build everything (not just stuff in the current working directory .)
import os
import pdb
import pprint

import build
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
    scons = build.make_scons(make_paths(*other_args))
    env.Command(
        scons['targets'],
        scons['sources'],
        scons['commands'],
    )


# main program
tickers = ['orcl']
for ticker in tickers:
    command(build.cusips, ticker)
    command(build.features, ticker)
    command(build.targets, ticker)
    for cusip in ['68389XAS4']:
        for hpset in ['grid2']:
            for effective_date in ['2016-11-01']:
                command(build.fit_predict, ticker, cusip, hpset, effective_date)

