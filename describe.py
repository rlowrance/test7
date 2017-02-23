'''describe CSV files in specified directory

INVOCATION: python describe.py path_to_dir

INPUT FILES:
 path_to_dir/*.csv

OUTPUT FILES:
 log.txt  whatever is printed when this program last ran
 *.csv    a description of each file in the input file

Written in Python 2.7
'''

from __future__ import division

import argparse
import glob
import os
import pandas as pd
import pdb
import random
import sys

import Bunch
import dirutility
import filter_describe
import seven
import seven.path
import Logger
import Timer


def make_control(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_dir')
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
        path_in_glob='*.csv',
        path_out_dir=path_out_dir,
        path_out_log=os.path.join(path_out_dir, '0log.txt'),
        random_seed=random_seed,
        test=arg.test,
        timer=Timer.Timer(),
        )


def unittests():
    return


def do_work(control):
    'process each file in the input directory, creating a description CSV in the output directory'
    # we may need to explicitly set column types of provide a function to convert to more usable type
    # column_types = {
    #    'issuepriceid': np.int64,
    #    }
    verbose = True
    glob_spec = os.path.join(control.arg.path_to_dir, control.path_in_glob)
    for path in glob.glob(glob_spec):
        print 'reading file', path
        df = pd.read_csv(
            path,
            low_memory=False,   # needed because of mixed dtypes in columns 57, 58, 59, 60, 21 for file ms.csv
            index_col=0,
            nrows=10 if control.arg.test else None,
            #  dtype=column_types,
        )
        if verbose:
            print df.head()
        categorical, numeric = filter_describe.describe(df)  # let Pandas deduce column types
        if verbose:
            print categorical
            print numeric
        dirs, filename = os.path.split(path)
        filename_base = filename.split('.')[0]
        categorical.to_csv(os.path.join(control.path_out_dir, filename_base + '-categorical.csv'))
        numeric.to_csv(os.path.join(control.path_out_dir, filename_base + '-numeric.csv'))
        continue


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

    unittests()
    main(sys.argv)
