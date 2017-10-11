'''create information needed to control the build process

Read each trace print file and the security master. Build a SQLITE data base holding the info.

Read each ticker file and update the files association information with CUSIPs.

NOTE: This program looks at all trades for the CUSIP and related OTR CUSIPs from the
beginning of time. Thus trades 10 years ago are used to predict next oasspreads today.
That is way too much history. We should fix this when we have a streaming infrastructure.
Most likely, only the last 1000 or so trades are relevant.

INVOCATION
  python buildinfo.py {--test} {--trace}  --get {id}  --analyze trace_prints
where
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution
 -- get {id} means to print information about the object with the id
 {id} is either a  CUSIP or an issuepriceid
 --analyze trace_print means to analyze the trace print table

EXAMPLES OF INVOCATIONS
  python buildinfo.py
  python buildinfo.py --get 68389XAU9  # ORCL cusip
  python buildinfo.py --get 127387311  # issue price id

See build.py for input and output files.

IDEAS FOR FUTURE:
1. Also scan all the trace files and create mappings across them. For example, it would be good
   to verify that every cusip has exactly one issuer and every issuepriceid occurs once.

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''

from __future__ import division

import argparse
import csv
import datetime
import os
import pdb
from pprint import pprint
import random
import sqlite3 as sqlite
import sys

import applied_data_science.debug
import applied_data_science.dirutility
import applied_data_science.lower_priority
import applied_data_science.pickle_utilities

from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.arg_type
import seven.build
import seven.feature_makers
import seven.fit_predict_output
import seven.read_csv
import seven.target_maker

pp = pprint


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze', action='store')
    parser.add_argument('--get', action='store')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')

    arg = parser.parse_args(argv[1:])

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    path = seven.build.buildinfo(test=arg.test)
    applied_data_science.dirutility.assure_exists(path['dir_out'])

    return Bunch(
        arg=arg,
        path=path,
        random_seed=random_seed,
        timer=Timer(),
    )


def create_tables(conn):
    conn.execute(
        '''CREATE TABLE cusip
        ( cusip text PRIMARY KEY
        , ticker text
        , coupon_type text
        , original_amount_issued real
        , issue_date text
        , maturity_date text
        , is_callable integer
        , is_puttable integer
        )
        '''
    )
    conn.execute(
        '''CREATE TABLE cusip_issuepriceid
        ( cusip text
        , issuepriceid integer
        , PRIMARY KEY (cusip, issuepriceid)
        )
        '''
    )
    conn.execute(
        '''CREATE TABLE trace_print
        ( issuepriceid integer PRIMARY KEY
        , cusip text
        , salescondcode text
        , secondmodifier text
        , wiflag
        , commissionflag
        , asofflag
        , specialpriceindicator
        , price real
        , yield real
        , yielddirection text
        , quantity real
        , estimatedquantity real
        , effectivedate text
        , effectivetime text
        , effectivedatetime text
        , halt text
        , cancelflag text
        , correctionflag text
        , trade_type text
        , is_suspect text
        , mka_oasspread real
        , oasspread real
        , convexity real
        )
        '''
    )


def etl_secmaster(conn, path):
    'return number of records inserted'
    count = 0
    with open(path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, restkey='extras')
        for row in reader:
            assert 'extras' not in row
            conn.execute(
                '''INSERT INTO cusip VALUES
                (:cusip,
                :ticker,
                :coupon_type,
                :original_amount_issued,
                :issue_date,
                :maturity_date,
                :is_callable,
                :is_puttable)
                ''',
                {
                    'cusip': row['CUSIP'],
                    'ticker': row['ticker'],
                    'coupon_type': row['coupon_type'],
                    'original_amount_issued': float(row['original_amount_issued']),
                    'issue_date': row['issue_date'],
                    'maturity_date': row['maturity_date'],
                    'is_callable': 1 if row['is_callable'] == 'true' else 0,
                    'is_puttable': 1 if row['is_puttable'] == 'true' else 0,
                }
            )

            count += 1
    return count


def etl_trace_print_files(conn, paths):
    def etl_trace_print_file(path):
        count = 0
        with open(path, 'rb') as csvfile:
            reader = csv.DictReader(csvfile, restkey='extrax')
            for row in reader:
                assert 'extras' not in row
                # print row
                year, month, day = row['effectivedate'].split('-')
                hour, minute, second = row['effectivetime'].split(':')
                date_time = datetime.datetime(
                    int(year),
                    int(month),
                    int(day),
                    int(hour),
                    int(minute),
                    int(second),
                )
                row['effectivedatetime'] = str(date_time)
                conn.execute(
                    '''INSERT into trace_print VALUES (
                    :issuepriceid,
                    :cusip,
                    :salescondcode,
                    :secondmodifier,
                    :wiflag,
                    :commissionflag,
                    :asofflag,
                    :specialpriceindicator,
                    :price,
                    :yield,
                    :yielddirection,
                    :quantity,
                    :estimatedquantity,
                    :effectivedate,
                    :effectivetime,
                    :effectivedatetime,
                    :halt,
                    :cancelflag,
                    :correctionflag,
                    :trade_type,
                    :is_suspect,
                    :mka_oasspread,
                    :oasspread,
                    :convexity)
                    ''',
                    row
                )
                conn.execute(
                    '''INSERT INTO cusip_issuepriceid VALUES
                    (:cusip,
                     :issuepriceid)
                     ''',
                    row,
                )
                count += 1
        return count

    count = 0
    for path in paths:
        path_count = etl_trace_print_file(path)
        print 'inserted %d records from trace file %s' % (path_count, path)
        count += path_count
    return count


def do_work(control):
    'accumulate information on the trace prints for the issuer and write that info to the file system'
    def lap():
        'return ellapsed wall clock time:float since previous call to lap()'
        return control.timer.lap('lap', verbose=False)[1]

    # reduce process priority, to try to keep the system responsive to user if multiple jobs are run
    applied_data_science.lower_priority.lower_priority()

    db_path = control.path['out_db']
    if os.path.isfile(db_path):
        os.remove(db_path)

    conn = sqlite.connect(db_path)  # create the file
    conn.row_factory = sqlite.Row

    create_tables(conn)
    print 'inserted %d records from sec master' % etl_secmaster(conn, control.path['in_secmaster'])

    trace_print_paths = control.path['list_in_trace']
    print 'inserted %d records from all %d trace print files' % (
        etl_trace_print_files(conn, trace_print_paths),
        len(trace_print_paths)
    )

    conn.commit()
    print 'made %d changes' % conn.total_changes
    return
    pass

    control.timer.lap('wrote output files')

    return None


def do_work_analyze(control):
    'print table of codes from the trace print tables'
    # def try_cusip(get, conn):
    #     'return n_records_found:int'
    #     stmt = r"SELECT * FROM cusip WHERE cusip = '%s'" % get
    #     n_found = 0
    #     for row in conn.execute(stmt):
    #         n_found += 1
    #         for k in row.keys():
    #             print '%25s: %10s' % (k, row[k])
    #         print
    #     return n_found

    # def try_issuepriceid(get, conn):
    #     'return n_records_found:int'
    #     stmt = r"SELECT * FROM issuepriceid WHERE issuepriceid = '%s'" % get
    #     n_found = 0
    #     for row in conn.execute(stmt):
    #         n_found += 1
    #         for k in row.keys():
    #             print '%25s: %10s' % (k, row[k])
    #         print
    #     return n_found

    if control.arg.analyze != 'trace_prints':
        print 'I know only how to analyze all of trace_print files'
        print' invoke me with --analyze trace_prints)'
        os.exit(1)

    db_path = control.path['out_db']
    if os.path.isfile(db_path):
        conn = sqlite.connect(db_path)
        conn.row_factory = sqlite.Row
    else:
        print 'did not find a sqlite database at path %s' % db_path
        os.exit(1)

    with conn as conn:
        cursor = conn.cursor()
        cursor.execute('select * from trace_print')
        # cursor.description is an iteratble of descriptions
        # each description is a tuple where tuple[0] is the column name
        names = [description[0] for description in cursor.description]
        skipped_names = [
            'issuepriceid', 'cusip', 'price', 'yield', 'quantity',
            'estimatedquantity', 'effectivedate', 'effectivetime', 'effectivedatetime', 'mka_oasspread',
            'oasspread', 'convexity',
            ]
        threshold = 20
        for name in names:
            if name in skipped_names:
                continue
            distinct_values = set()
            stmt = 'select %s from trace_print' % name
            for row in conn.execute(stmt):
                distinct_values.add(row[name])
                if len(distinct_values) > threshold:
                    break
            print name, sorted(distinct_values), ('...' if len(distinct_values) > threshold else '')
    return


def do_work_get(control):
    def try_cusip(get, conn):
        'return n_records_found:int'
        stmt = r"SELECT * FROM cusip WHERE cusip = '%s'" % get
        n_found = 0
        for row in conn.execute(stmt):
            n_found += 1
            for k in row.keys():
                print '%25s: %10s' % (k, row[k])
            print
        return n_found

    def try_issuepriceid(get, conn):
        'return n_records_found:int'
        stmt = r"SELECT * FROM trace_print WHERE issuepriceid = '%s'" % get
        n_found = 0
        for row in conn.execute(stmt):
            n_found += 1
            for k in row.keys():
                print '%25s: %10s' % (k, row[k])
            print
        return n_found

    db_path = control.path['out_db']
    if os.path.isfile(db_path):
        conn = sqlite.connect(db_path)
        conn.row_factory = sqlite.Row
    else:
        print 'did not find a sqlite database at path %s' % db_path
        os.exit(1)

    # attempt to retrieve records
    with conn as conn:
        n_found = try_cusip(control.arg.get, conn)
        if n_found == 0:
            n_found = try_issuepriceid(control.arg.get, conn)
    print 'found %d records' % n_found
    return


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
    if not control.arg.get:
        print control
        lap = control.timer.lap

    if control.arg.get:
        do_work_get(control)
    elif control.arg.analyze:
        do_work_analyze(control)
    else:
        do_work(control)

    if control.arg.get:
        sys.exit(0)

    lap('work completed')
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    # print control
    print control.arg
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
        datetime

    main(sys.argv)
