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


class Doit(object):
    def __init__(self, ticker, test=False, me='cusip'):
        self.ticker = ticker
        self.me = me
        self.test = test
        # define directories
        midpredictor = seven.path.midpredictor_data()
        working = seven.path.working()
        out_dir = os.path.join(working, me + ('-test' if test else ''))
        # path to files abd durectirs
        self.in_ticker = os.path.join(midpredictor, ticker + '.csv')
        self.out_cusips = os.path.join(out_dir, '%s.pickle' % ticker)
        self.out_dir = out_dir
        self.out_log = os.path.join(out_dir, '0log.txt')
        # used by Doit tasks
        self.actions = [
            'python %s.py %s' % (me, ticker)
        ]
        self.targets = [
            self.out_cusips,
            self.out_log,
        ]
        self.file_dep = [
            self.me + '.py',
            self.in_ticker,
        ]

    def __str__(self):
        for k, v in self.__dict__.iteritems():
            print 'doit.%s = %s' % (k, v)
        return self.__repr__()


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
    doit = Doit(arg.ticker, test=arg.test, me=arg.me)
    dirutility.assure_exists(doit.out_dir)

    return Bunch.Bunch(
        arg=arg,
        doit=doit,
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
    df_ticker = read_csv(control.doit.in_ticker)
    cusips = set(df_ticker.cusip)
    result = {}
    for cusip in cusips:
        mask = cusip == df_ticker.cusip
        count = sum(mask)
        result[cusip] = count
        print cusip, count
    with open(control.doit.out_cusips, 'w') as f:
        pickle.dump(result, f)
    return None


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger.Logger(logfile_path=control.doit.out_log)  # now print statements also write to the log file
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
