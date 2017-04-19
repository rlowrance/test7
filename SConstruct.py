# invocations:
#   scons -f SConstruct.py /
#   scons -n -f SConstruct.py /   

# where / means to build everything (not just stuff in the current working directory .)
import os
import pdb
import cPickle as pickle
import pprint

import build
pp = pprint.pprint

dir_home = os.path.join('C:', r'\Users', 'roylo')
dir_dropbox = os.path.join(dir_home, 'Dropbox')
dir_working = os.path.join(dir_dropbox, 'data', '7chord', '7chord-01', 'working')
dir_midpredictor_data = os.path.join(dir_dropbox, 'MidPredictor', 'data')


def maybe_read_cusips(ticker):
    'return list of (precomputed) cusisp for the ticker'
    try:
        with open(os.path.join(dir_working, 'cusips-orcl', 'orcl.pickle'), 'r') as f:
            return pickle.load(f).keys()  # a dict is stored
    except:
        return None

env= Environment(
    ENV={
        'PATH': os.environ['Path'],
        'PYTHONPATH': os.environ['PYTHONPATH'],
    },
)
env.Decider('MD5-timestamp')  # if timestamp out of date, examine MD5 checksum


def build_cusips(ticker):
    'issue command to run cusips.py'
    scons = build.make_scons(build.cusips(ticker))
    env.Command(
        scons['targets'],
        scons['sources'],
        scons['commands'],
    )


def build_features(ticker):
    'issue commands to run features.py'
    scons = build.make_scons(build.features(ticker))
    env.Command(
        scons['targets'],
        scons['sources'],
        scons['commands'],
    )


 # main program
tickers = ['orcl']
for ticker in tickers:
    build_cusips(ticker)
    build_features(ticker)
