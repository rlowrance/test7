'''describe all files in directory 7chord_ticker_universe_nyu_poc

INVOCATION: python describe.py 

INPUT FILES: each *.csv file in NYU/7chord_ticker_universe_nyu_poc

OUTPUT FILES:
 log.txt  whatever is printed when this program last ran
 *.csv    a description of each file in the input file

Written in Python 2.7
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import glob
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import Bunch
import dirutility
import Logger
import Timer


def make_control(argv):
    print 'argv', argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])  # ignore invocation name
    arg.me = 'describe'

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    # put all output in directory
    path_out_dir = dirutility.assure_exists('../data/working/' + arg.me + '/')

    return Bunch.Bunch(
        arg=arg,
        path_in_dir='../data/input/7chord_team_folder/NYU/7chord_ticker_universe_nyu_poc/',
        path_in_glob='*.csv',
        path_out_dir=path_out_dir,
        path_out_log=path_out_dir + '0log.txt',
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
    path = control.path_in_dir + control.path_in_glob
    all_dtypes = {}
    for path in glob.glob(control.path_in_dir + control.path_in_glob):
        filename = path.split('/')[-1]
        print 'reading file', filename
        df = pd.read_csv(
            path,
            low_memory=False,   # needed because of mixed dtypes in columns 57, 58, 59, 60, 21 for file ms.csv
            index_col=0,
            #  dtype=column_types,
        )
        if control.test:
            print df.head()
        # print column types
        print 'file %s: %d records' % (filename, len(df))
        for column_name in df.columns:
            v = df[column_name]
            print '%22s %-8s' % (column_name, v.dtype), v.iloc[0], v.iloc[1], v.iloc[2], v.iloc[3], v.iloc[4], v.iloc[5]
            if column_name in all_dtypes:
                if all_dtypes[column_name] != v.dtype:
                    print 'new type %s for column %s in file %s' % (v.dtype, column_name, filename)
        print
        description = df.describe(
            percentiles=[.25, .50, .75],
            include='all',
            #  include=[np.number, object],  # both numeric and categorical colum typics)
            )
        if control.test:
            print description
        for column_name in description.columns:
            print column_name
            print description[column_name]
            print
        description.to_csv(
            control.path_out_dir + filename,
            )


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
