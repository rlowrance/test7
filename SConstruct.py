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


env= Environment(
    ENV={
        'PATH': os.environ['Path'],
        'PYTHONPATH': os.environ['PYTHONPATH'],
    },
)
env.Decider('MD5-timestamp')  # if timestamp out of date, examine MD5 checksum


def command(ticker, make_paths):
    scons = build.make_scons(make_paths(ticker))
    env.Command(
        scons['targets'],
        scons['sources'],
        scons['commands'],
    )


# main program
tickers = ['orcl']
for ticker in tickers:
    command(ticker, build.cusips)
    command(ticker, build.features)
    command(ticker, build.fit_predict)
    command(ticker, build.targets)
