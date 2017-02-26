'''determine CUSIPS in a ticker file

INVOCATION: python describe.py {ticker}.csv [--test] [--trace]

INPUT FILES:
 MidPredictor/data/{ticker}.csv

OUTPUT FILES:
 WORKING/cusips[-test]/0log-{ticker}.txt  whatever is printed when this program last ran
 WORKING/cusips[-test]/{ticker}.pickle  # Dict[cusip, count]
'''

from __future__ import division

import argparse
import cPickle as pickle
import os
import pandas as pd
import pdb
import random
import sys

import Bunch
import dirutility
import seven
import seven.path
import Logger
import Timer


def make_control(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])  # ignore invocation name
    arg.me = parser.prog.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    # put all output in directory
    path_out_dir = os.path.join(
        seven.path.working(),
        arg.me + ('-test' if arg.test else '')
    )
    dirutility.assure_exists(path_out_dir)

    return Bunch.Bunch(
        arg=arg,
        path_in_ticker=os.path.join(seven.path.midpredictor_data(), arg.ticker + '.csv'),
        path_out_cusips=os.path.join(path_out_dir, '%s.pickle' % arg.ticker),
        path_out_log=os.path.join(path_out_dir, '0log.txt'),
        random_seed=random_seed,
        test=arg.test,
        timer=Timer.Timer(),
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
    df_ticker = read_csv(control.path_in_ticker)
    cusips = set(df_ticker.cusip)
    result = {}
    for cusip in cusips:
        mask = cusip == df_ticker.cusip
        count = sum(mask)
        result[cusip] = count
        print cusip, count
    with open(control.path_out_cusips, 'w') as f:
        pickle.dump(result, f)
    return None


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger.Logger(logfile_path=control.path_out_log)  # now print statements also write to the log file
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
