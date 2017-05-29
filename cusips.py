'''determine CUSIPS in a ticker file

INVOCATION: python cusips.py {ticker}.csv [--test] [--trace]

See build.py for input and output files
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import pandas as pd
import pdb
import random
import sys

import applied_data_science.debug
import applied_data_science.dirutility

from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

import seven.build
import seven.read_csv


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
    paths = seven.build.cusips(arg.ticker, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        test=arg.test,
        timer=Timer(),
        )


def first_datetime(d, cusip, dates):
    'return None and set d[cusip] to the earlist date in datetimes'
    earliest = dates.min()
    if cusip in d:
        return min(d[cusip], earliest)
    else:
        return earliest


def last_datetime(d, cusip, dates):
    'return None and set d[cusip] to the earlist date in datetimes'
    latest = dates.max()
    if cusip in d:
        return max(d[cusip], latest)
    else:
        return latest


def reduce_trace_records(trace_records, cusips):
    'return Dict[cusip, count], Dict[cusip, first_datetime], Dict[cusip, last_datetime]'
    counts = {}
    first_datetimes = {}
    last_datetimes = {}
    for cusip in cusips:
        mask = cusip == trace_records.cusip
        count = sum(mask)
        counts[cusip] = count

        for_cusip = trace_records[mask]

        effectivedates = for_cusip['effectivedate']
        first_datetimes[cusip] = first_datetime(first_datetimes, cusip, effectivedates)
        last_datetimes[cusip] = last_datetime(last_datetimes, cusip, effectivedates)
    return counts, first_datetimes, last_datetimes


def make_first_last_dataframe(first_datetimes, last_datetimes):
    'return DataFrame'
    data = collections.defaultdict(list)
    for cusip in sorted(first_datetimes.keys()):
        data['cusip'].append(cusip)
        data['first_effectivedate'].append(first_datetimes[cusip])
        data['last_effectivedate'].append(last_datetimes[cusip])
    result = pd.DataFrame(data=data, index=data['cusip'])
    return result


def make_counts_by_cusip_year_month(trace_records):
    'return sorted DataFrame'

    # reduce across cusip, year, month
    reduction = collections.defaultdict(int)
    for index, row in trace_records.iterrows():
        cusip = row['cusip']
        year = row['effectivedate'].year
        month = row['effectivedate'].month
        reduction[(cusip, year, month)] += 1

    # create data to construct DataFrame
    data = collections.defaultdict(list)
    for (cusip, year, month), count in reduction.iteritems():
        data['cusip'].append(cusip)
        data['year'].append(year)
        data['month'].append(month)
        data['count'].append(count)

    counts_by_month = pd.DataFrame(data)
    reordered = counts_by_month[['cusip', 'year', 'month', 'count']]
    sorted_dataframe = reordered.sort_values(by=['cusip', 'year', 'month'])
    return sorted_dataframe


def do_work(control):
    # BODY STARTS HERE
    trace_records = seven.read_csv.input(
            control.arg.ticker,
            'trace',
            nrows=1000 if control.arg.test else None,
    )
    cusips = set(trace_records['cusip'])

    counts, first_datetimes, last_datetimes = reduce_trace_records(trace_records, cusips)
    with open(control.path['out_cusips'], 'w') as f:
        pickle.dump(counts, f)

    first_last = make_first_last_dataframe(first_datetimes, last_datetimes)
    with open(control.path['out_first_last'], 'w') as f:
        first_last.to_csv(f)

    counts_by_cusip_year_month = make_counts_by_cusip_year_month(trace_records)
    with open(control.path['out_counts_by_month'], 'w') as f:
        counts_by_cusip_year_month.to_csv(f)

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
