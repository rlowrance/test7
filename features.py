'''create feature sets for a CUSIP

INVOCATION
  python features.py {trade-print-filename} [--test] [--trace]

where
 {ilename} is a CSV file in MidPredictors/data
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python features.py orcl_order_imb_sample1.csv

INPUTS
 MidPredictor/{trade-print-filename}
 WORKING/order-imbalance3/{cusip}.csv

where
 {cusip} is identified in the {trade-print-filename}

OUTPUTS
 WORKING/order-imbalance3/{cusip}.csv in asending order by datetime
  containing X features
   original_index, datetime, trade_type, order_imbalance, days_to_maturity_ coupon
   priot_price_b, prior_price_D, prior_price_S
  containing y targes
   next_price_B, next_price_D, next_price_S

NOTES:
- how to read datetimes: http://stackoverflow.com/questions/33397871/pandas-to-csv-and-then-read-csv-results-to-numpy-datetime64-messed-up-due-to-utc
  pd.read_csv(path, parse_dates=['column name'])
'''

from __future__ import division

import argparse
import collections
import datetime
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import arg_type
from Bunch import Bunch
import dirutility
from Logger import Logger
import seven
import seven.path
from Timer import Timer


def make_control(argv):
    'return a Bunch'

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('trade_print_filename', type=arg_type.filename_csv)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])
    arg.me = parser.prog.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    dir_working = seven.path.working()
    if arg.test:
        dir_out = os.path.join(dir_working, arg.me + '-test')
    else:
        dir_out = os.path.join(dir_working, arg.me)
    dirutility.assure_exists(dir_out)
    args_str = arg.trade_print_filename

    return Bunch(
        arg=arg,
        path_in_trade_print_filename=os.path.join(seven.path.midpredictor_data(), arg.trade_print_filename),
        path_in_feature_order_imbalance=os.path.join(seven.path.working(), 'order-imbalance3'),
        path_dir_out=dir_out,  # file {cusip}.csv is created here
        path_out_log=os.path.join(dir_out, '0log-' + args_str + '.txt'),
        random_seed=random_seed,
        timer=Timer(),
    )


month_number = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}


def make_next_price(trade_type, datetime, index, df):
    mask = df.datetime > datetime
    after_current_trade = df.loc[mask]
    # sorted = after_current_trade.sort_values('datetime')
    for index, row in after_current_trade.iterrows():
        if row.trade_type == trade_type:
            return row.trade_price
    return None


def make_days_to_maturity(maturity_date, datetime):
    'return float'
    seconds_per_day = 24 * 60 * 60
    diff = maturity_date - datetime
    days = diff.days
    seconds = diff.seconds
    result = days + seconds / seconds_per_day
    return result


def to_datetime_time(s):
    'convert str like HH:MM:SS to datetime.time value'
    hour, minute, second = s.split(':')
    return datetime.time(
        int(hour),
        int(minute),
        int(second),
    )


def test_equal(a, b):
    'trace if a != b'
    if a != b:
        print 'not equal'
        print a
        print b
        pdb.set_trace()


def test_true(b):
    if not b:
        print 'not true'
        print b
        pdb.set_trace()


class Skipped(object):
    def __init__(self):
        self.counter = collections.Counter()
        self.n_skipped = 0

    def skip(self, reason):
        self.counter[reason] += 1
        self.n_skipped += 1


def do_work(control):
    'write order imbalance for each trade in the input file'
    def read_csv(path, date_columns):
        df = pd.read_csv(
            path,
            # nrows=20 if control.arg.test else None,
            usecols=None,
            low_memory=False,
            parse_dates=None if date_columns is None else date_columns,
        )
        print 'read %d rows from file %s' % (len(df), path)
        print df.columns
        return df

    # BODY STARTS HERE
    verbose = False
    df_trade_print = read_csv(
        control.path_in_trade_print_filename,
        ['maturity', 'effectivedate'],
    )
    cusip = df_trade_print.cusip[0]
    df_order_imbalance = read_csv(
        os.path.join(control.path_in_feature_order_imbalance, cusip + '.csv'),
        ['datetime'],
    )
    print len(df_trade_print), len(df_order_imbalance)

    result = pd.DataFrame()
    skipped = Skipped()

    # setup tracking of prior prices and quantities
    prior_price = {}
    prior_quantity = {}
    trade_types = ('B', 'D', 'S')
    for trade_type in trade_types:
        prior_price[trade_type] = None
        prior_quantity[trade_type] = None

    for order_imbalance_index, row_order_imbalance in df_order_imbalance.iterrows():
        original_index = row_order_imbalance.trade_print_index
        if control.arg.test:
            if original_index not in df_trade_print.index:
                print 'testing: skipping order_imbalance %s, as not in trades' % original_index
                continue
        else:
            test_true(original_index in df_trade_print.index)

        row_trade_print = df_trade_print.loc[original_index]
        if verbose:
            print order_imbalance_index, original_index
            print row_order_imbalance
            print row_trade_print

        test_equal(row_order_imbalance.datetime.date(), row_trade_print.effectivedate.date())
        test_equal(row_order_imbalance.datetime.time(), to_datetime_time(row_trade_print.effectivetime))

        (
            trade_print_index,
            datetime,
            trade_type,
            trade_quantity,
            trade_price,
            order_imbalance,
            trade_print_index,
        ) = row_order_imbalance

        # create last price and quantity features in list historic_price_quantity
        prior_price[trade_type] = trade_price
        prior_quantity[trade_type] = trade_quantity
        historic_price_quantity = {}
        for trade_type in trade_types:
            if prior_price[trade_type] is None or prior_quantity[trade_type] is None:
                skipped.skip('missing prior %s trade' % trade_type)
                break
            historic_price_quantity['prior_price_%s' % trade_type] = prior_price[trade_type]
            historic_price_quantity['prior_quantity_%s' % trade_type] = prior_quantity[trade_type]
        if len(historic_price_quantity) != 2 * len(trade_types):
            continue  # we found a prior_price or prior_quantity was None

        # create targets, which are the next prices for each type of trade
        targets = {}
        for trade_type in trade_types:
            next_price = make_next_price(trade_type, datetime, original_index, df_order_imbalance)
            if next_price is None:
                skipped.skip('no future %s price' % trade_type)
                break
            targets['next_price_%s' % trade_type] = next_price
        if len(targets) != 3:
            continue  # something was skipped

        next_row = {
            'original_index': original_index,
            'datetime': datetime,
            'trade_type': trade_type,
            'order_imbalance': order_imbalance,
            'days_to_maturity': make_days_to_maturity(row_trade_print.maturity, datetime),
            'coupon': float(row_trade_print.coupon),
        }
        next_row.update(targets)
        next_row.update(historic_price_quantity)
        next_df = pd.DataFrame(
            next_row,
            index=[original_index],
        )
        result = result.append(
            next_df,
            verify_integrity=True,
        )
        print 'appended features for original input row %d (%d of %d)' % (
            original_index,
            len(result) + skipped.n_skipped,
            len(df_order_imbalance.index),
        )
        if control.arg.test and len(result) > 10:
            break

    print 'len result', len(result)

    print 'skipped reasons for skipping input records'
    for k, v in skipped.counter.iteritems():
        print '%30s: %d' % (k, v)

    # write output file
    # NOTE: the csv file is in ascending order by datetime
    path_out = os.path.join(control.path_dir_out, cusip + '.csv')
    print 'write csv file', path_out
    result.to_csv(path_out)
    return


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path_out_log)  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    print control
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflake warnings for debugging tools
        pdb.set_trace()
        pprint()

    main(sys.argv)
