'''create targets sets for all CUSIPS in a ticker file

INVOCATION
  python targets.py {ticker}.csv [--cusip CUSIP] [--test] [--trace]

where
 {ticker}.csv is a CSV file in MidPredictors/data
 --cusip CUSIP means to create the targets only for the specified CUSIP
 --test means to set control.test, so that test code is executed
 --trace means to invoke pdb.set_trace() early in execution

EXAMPLES OF INVOCATION
 python targets.py orcl.csv

INPUTS
 MidPredictor/{ticker}.csv

where
 {cusip} is identified in the {trade-print-filename}

OUTPUTS
 WORKING/targets/{ticker}-{cusip}.csv: columns: next_B_price, next_D_price, next_S_price index:<from ticker>
   One file for each cusip in the ticker
 WORKING/tagets/0log-{ticker}.txt

'''

from __future__ import division

import argparse
import datetime
import os
import numpy as np
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

    parser = argparse.ArgumentParser()
    parser.add_argument('ticker_filename', type=arg_type.filename_csv)
    parser.add_argument('--cusip', action='store', type=arg_type.cusip)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])
    arg.me = parser.prog.split('.')[0]
    arg.ticker = arg.ticker_filename.split('.')[0]

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

    return Bunch(
        arg=arg,
        path_in_ticker_filename=os.path.join(seven.path.midpredictor_data(), arg.ticker_filename),
        path_out_dir=dir_out,  # file {cusip}.csv is created here
        path_out_log=os.path.join(dir_out, '0log-' + arg.ticker + '.txt'),
        random_seed=random_seed,
        timer=Timer(),
    )


def do_work(control):
    'write order imbalance for each trade in the input file'
    def read_csv(path, date_columns=None, usecols=None):
        debug = False
        df = pd.read_csv(
            path,
            nrows=20 if control.arg.test else None,
            usecols=None if debug else usecols,
            low_memory=False,
            parse_dates=date_columns,
        )
        print 'read %d rows from file %s' % (len(df), path)
        print df.columns
        return df

    def next_prices(df, index):
        'return Dict of next prices for each trade_type'
        mask = df.effectivedatetime > df.loc[index].effectivedatetime
        after_current_trade = df.loc[mask]
        result = {}
        for index, row in after_current_trade.iterrows():
            if row.trade_type not in result:
                result[row.trade_type] = row.price
            if len(result) == 3:
                # stop when we have a D, B, and S trade
                break
        return result

    def validate(df):
        'Raise if a problem'
        assert (df.ticker == control.arg.ticker.upper()).all()

    def add_datetime(df):
        'create new column that combines the effectivedate and effective time'
        values = []
        for the_date, the_time in zip(df.effectivedate, df.effectivetime):
            values.append(datetime.datetime(
                the_date.year,
                the_date.month,
                the_date.day,
                the_time.hour,
                the_time.minute,
                the_time.second,
            ))
        df['effectivedatetime'] = pd.Series(values, index=df.index)

    # BODY STARTS HERE
    # read and transform the input ticker file
    df_ticker = read_csv(
        control.path_in_ticker_filename,
        date_columns=['effectivedate', 'effectivetime'],
        usecols=['cusip', 'price', 'effectivedate', 'effectivetime', 'ticker', 'trade_type'],
    )
    print df_ticker.columns
    print 'read %d input trades' % len(df_ticker)
    cusips = set(df_ticker.cusip)
    print 'containing %d CUSIPS' % len(cusips)
    validate(df_ticker)
    add_datetime(df_ticker)
    del df_ticker['effectivedate']
    del df_ticker['effectivetime']
    del df_ticker['ticker']

    # create a result file for each CUSIP in the input ticker file
    for i, cusip in enumerate(cusips):
        print 'cusip %s %d of %d' % (cusip, i + 1, len(cusips))
        if control.arg.cusip is not None:
            if cusip != control.arg.cusip:
                print 'skipping', cusip
                continue
        mask = df_ticker.cusip == cusip
        df_cusip_unsorted = df_ticker.loc[mask]
        df_cusip = df_cusip_unsorted.sort_values(by='effectivedatetime')
        result = pd.DataFrame(
            columns=['next_price_B', 'next_price_D', 'next_price_S'],
            index=df_cusip.index,
        )  # pre-allocate for speed
        for index, row in df_cusip.iterrows():
            prices = next_prices(df_cusip, index)
            next_row = (prices.get('B', np.nan), prices.get('D', np.nan), prices.get('S', np.nan))
            result.loc[index] = next_row
        print 'cusip %s len result %d' % (cusip, len(result))
        path = os.path.join(control.path_out_dir, '%s-%s.csv' % (control.arg.ticker, cusip))
        print 'writing to', path
        result.to_csv(path)
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
