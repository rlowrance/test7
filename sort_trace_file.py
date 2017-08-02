'''sort a trace file so that it is in datetime order

INVOCATION
  python sort_trade_file.py {issuer} {--debug} {--test} {--trace}
where
 issuer the issuer (ex: AAPL)
 cusip is the cusip id (9 characters; ex: 68389XAS4)
 --debug means to call pdb.set_trace() instead of raisinng an exception, on calls to logging.critical()
   and logging.error()
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
  python sort_trace_file.py AAPL

See build.py for input and output files.

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''

from __future__ import division

import argparse
import csv
import pdb
from pprint import pprint
import random
import sys

import applied_data_science.debug
import applied_data_science.dirutility
import applied_data_science.lower_priority
import applied_data_science.pickle_utilities

from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.accumulators
import seven.arg_type
import seven.build
import seven.EventId
import seven.feature_makers2
import seven.fit_predict_output
import seven.logging
import seven.read_csv

pp = pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('issuer', type=seven.arg_type.issuer)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')

    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()
    if arg.debug:
        # logging.error() and logging.critial() call pdb.set_trace() instead of raising an exception
        seven.logging.invoke_pdb = True

    random_seed = 123
    random.seed(random_seed)

    paths = seven.build.sort_trace_file(arg.issuer, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    timer = Timer()

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=timer,
    )


def do_work(control):
    'read a trace file, sort it, then write it'
    with open(control.path['in_trace_file']) as f:
        dict_reader = csv.DictReader(f)
        all_rows = []
        for row in dict_reader:
            event_id = seven.EventId.TraceEventId(
                row['effectivedate'],
                row['effectivetime'],
                control.arg.issuer,
                row['issuepriceid'],
            )
            all_rows.append((event_id, row))
            if control.arg.test and len(all_rows) > 2:
                break
    print 'read %d rows' % len(all_rows)

    # sort the input
    all_rows_sorted = sorted(all_rows, key=lambda x: x[0])

    # write the sorted file
    count = 0
    with open(control.path['out_sorted_trace_file'], 'w') as f:
        dict_writer = csv.DictWriter(f, dict_reader.fieldnames, lineterminator='\n')
        dict_writer.writeheader()
        for event_id, row in all_rows_sorted:
            dict_writer.writerow(row)
            count += 1
            if control.arg.test and count > 2:
                break
    print 'wrote %d rows sorted by event id' % (count)

    if control.arg.test:
        # read it back in
        pdb.set_trace()
        with open(control.path['out_sorted_trace_file']) as f:
            dict_reader = csv.DictReader(f)
            for row in dict_reader:
                print row

    return None


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    # print control
    print control.arg
    print 'done'
    return


if __name__ == '__main__':
    main(sys.argv)
