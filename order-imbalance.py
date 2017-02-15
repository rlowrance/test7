'''determine OrderImbalance features for a file containing TRACE prints

INVOCATION
  python order-imbalance.py {filename}.csv [--test] [--trace]

where
 filename is a CSV file in MidPredictor that contains TRACE prints
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATIONS
 python order-imbalance.py orcl_order_imb_sample1.csv

INPUTS
 MidPredictor/{filename}

OUTPUTS
 WORKING/order-imbalance/{filename}.pickle containing obj: Dict[index, order_imbalance]
 WORKING/order-imbalance/{filename}.csv containing columns:
   datetime, trade_type, trade_quantity, trade_price, order_imbalance
   for each trade in {filename}.csv
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import datetime
import numpy as np
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
from seven.OrderImbalance2 import OrderImbalance
import seven.path
from Timer import Timer


def make_control(argv):
    'return a Bunch'

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=arg_type.filename_csv)
    parser.add_argument('lookback', type=arg_type.positive_int)
    parser.add_argument('typical_bid_offer', type=arg_type.positive_int)
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
    args_str = '%s-%s-%s' % (
                arg.filename.split('.')[0],
                arg.lookback,
                arg.typical_bid_offer,
    )

    return Bunch(
        arg=arg,
        path_in_file=os.path.join(seven.path.midpredictor_data(), arg.filename),
        path_out_file_csv=os.path.join(dir_out, args_str + '.csv'),
        path_out_file_pickle=os.path.join(dir_out, args_str + '.pickle'),
        path_out_log=os.path.join(dir_out, '0log-' + args_str + '.txt'),
        random_seed=random_seed,
        timer=Timer(),
    )


class InputLayout(object):
    'know the layout of the input file'
    def __init__(self):

        self.expected_cusip = None
        self.last_timestamp = None

    def extract_info(self, row):
        'return validated (cusip, datetime, trade_type, trade_quantity, trade_price)'
        # only one of the fields buy_price, dlr_price, sell_price should be set
        # the other two should be NaN
        if not np.isnan(row.buy_price):
            trade_type = 'B'
            trade_price = row.buy_price
        elif not np.isnan(row.dlr_price):
            trade_type = 'D'
            trade_price = row.dlr_price
        elif not np.isnan(row.sell_price):
            trade_type = 'S'
            trade_price = row.sell_price
        else:
            print row
            print 'datasets is not as expected'
            pdb.set_trace()
            return None
        cusip = row.cusip
        self.validate_cusip(cusip)
        timestamp = row.datetime
        self.validate_timestamp(timestamp)
        return (
            cusip,
            timestamp,
            trade_type,
            row.quantity,
            trade_price,
        )

    def validate_cusip(self, cusip):
        'check that CUSIPs are identical'
        if self.expected_cusip is not None:
            if cusip != self.expected_cusip:
                print 'bad cusip', cusip, self.expected_cusip
                pdb.set_trace()
        else:
            self.expected_cusip = cusip

    def validate_timestamp(self, timestamp):
        'check that rows are in timestamp order'
        # print 'validate_timestamp', timestamp, self.last_timestamp
        if self.last_timestamp is not None:
            if timestamp < self.last_timestamp:
                print 'bad timestamp', timestamp, self.last_timestamp
                pdb.set_trace()
        self.last_timestamp = timestamp


month_number = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}


def make_datetime(date, time):
    day, month, year = date.split('-')
    hour, minute, second = time.split(':')
    try:
        assert int(year) < 50  # detect potential Y2K date issues
        result = datetime.datetime(
            int(year) + 2000,
            month_number[month],
            int(day),
            int(hour),
            int(minute),
            int(second),
        )
        return result
    except:
        print 'bad date time', date, time
        print 'Exception: ', sys.exc_info()[0]
        pdb.set_trace()
        return None


def transform_raw(df):
    'mutate df: add datetime column, and sort on datetimes'
    print 'transforming raw df'

    # add datetime column
    print 'adding datetime column'
    datetimes = []
    for index, row in df.iterrows():
        datetimes.append(make_datetime(row.effectivedate, row.effectivetime))
    df['datetime'] = pd.Series(datetimes, index=df.index)

    # sort on datetime column
    print 'sorting on datetime column'
    sorted = df.sort_values('datetime', ascending=True)
    print 'transforming finished'
    return sorted


def do_work(control):
    'write order imbalance for each trade in the input file'
    def read_csv(path):
        df = pd.read_csv(
            path,
            nrows=8000 if control.arg.test else None,
            usecols=None,  # TODO: change to columns we actually use
            low_memory=False
        )
        print 'read %d samples from file %s' % (len(df), path)
        return df

    # BODY STARTS HERE
    verbose = False
    df_raw = read_csv(control.path_in_file)
    df = transform_raw(df_raw)
    oi = OrderImbalance(
        lookback=control.arg.lookback,
        typical_bid_offer=control.arg.typical_bid_offer,
    )
    result = []
    result_dict = {}
    result_df = pd.DataFrame(  # pre-allocate dataframe
        columns=('datetime', 'trade_type', 'trade_quantity', 'trade_price', 'open_interest'),
        index=df.index,
    )
    counters = collections.Counter()
    input_layout = InputLayout()

    for index, row in df.iterrows():
        if verbose:
            print index
            print row
            pdb.set_trace()
        cusip, timestamp, trade_type, trade_quantity, trade_price = input_layout.extract_info(row)

        counters[trade_type] += 1
        open_interest = oi.open_interest(
            trade_type=trade_type,
            trade_quantity=trade_quantity,
            trade_price=trade_price,
        )
        result.append((index, open_interest))
        result_dict[index] = open_interest
        result_df.loc[index] = (
            timestamp,
            trade_type,
            trade_quantity,
            trade_price,
            np.nan if open_interest.value is None else open_interest.value,
        )
    print 'len result', len(result)
    assert len(df) == len(result)

    # print the counters
    for k, v in counters.iteritems():
        print 'trade_type', k, 'occurred', v

    # print the first and last non-None results
    n_none = 0
    while True:
        index, open_interest = result[n_none]
        if open_interest.value is not None:
            break
        n_none += 1
    print 'first %d open_interest values are None' % n_none

    print 'index, open_interest: head and tail'
    for i in xrange(n_none, n_none + 10):
        index, open_interest = result[i]
        print index, open_interest.value
    for i in xrange(len(result) - 10, len(result) - 1):
        index, open_interest = result[i]
        print index, open_interest.value

    # write output files
    with open(control.path_out_file_pickle, 'w') as f:
        pickle.dump((control, result_dict), f)
    result_df.to_csv(control.path_out_file_csv)
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
