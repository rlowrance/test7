'''determine CUSIPS in a ticker file

INVOCATION: python cusips.py {ticker}.csv [--test] [--trace]

See build.py for input and output files
'''

from __future__ import division

import argparse
import cPickle as pickle
import pandas as pd
import pdb
import random
import sys

import applied_data_science

import applied_data_science.dirutility

from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import build


def make_control(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])  # ignore invocation name

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    # put all output in directory
    paths = build.cusips(arg.ticker, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        test=arg.test,
        timer=Timer(),
        )


def do_work(control):
    def read_csv(path, parse_dates=None):
        df = pd.read_csv(
            path,
            index_col=0,
            nrows=100 if control.arg.test else None,
            usecols=None,
            low_memory=False,
            parse_dates=parse_dates,
        )
        print 'read %d rows from file %s' % (len(df), path)
        print df.columns
        return df

    # BODY STARTS HERE
    df_ticker = read_csv(control.path['in_trace'])
    cusips = set(df_ticker.cusip)
    result = {}
    for cusip in cusips:
        mask = cusip == df_ticker.cusip
        count = sum(mask)
        result[cusip] = count
        print cusip, count
    with open(control.path['out_cusips'], 'w') as f:
        pickle.dump(result, f)
    return None


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(logfile_path=control.path['out_log'])  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.test:
        print 'DISCARD OUTPUT: test'
    print control
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pass
    main(sys.argv)
