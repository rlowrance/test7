'''reduce all the fit-predict output into a single large CSV file with all predictions

INVOCATION
  python fit_predict_reduce.py {ticker} {cusip} {hpset} {effective_date} [--test] [--trace]

EXAMPLES OF INVOCATIONS
 python fit_predict_reduce.py orcl 68389XAS4 grid2 2016-11-01

INPUTS
 WORKING/fit-predict/{ticker}-{cusip}.csv  # TODO: change to .pickle, once fit-predict is fixed

OUTPUTS
 WORKING/fit-predict-reduce/{ticker}-{cusip}-actuals-predictions.pickle
 TODO: also produce an analysis of the importances
'''

from __future__ import division

import argparse
import collections
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import applied_data_science.dirutility
import applied_data_science.lower_priority
import applied_data_science.pickle_utilities

from applied_data_science.Bunch import Bunch
from applied_data_science.Logger import Logger
from applied_data_science.Timer import Timer

from seven import arg_type
# from seven.FitPredictOutput import Id, Payload, Record

import build


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', type=arg_type.ticker) 
    parser.add_argument('cusip', type=arg_type.cusip)
    parser.add_argument('hpset', type=arg_type.hpset)
    parser.add_argument('effective_date', type=arg_type.date)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])
    arg.me = parser.prog.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    paths = build.fit_predict_reduce(arg.ticker, arg.cusip, arg.hpset, arg.effective_date, test=arg.test)
    applied_data_science.dirutility.assure_exists(paths['dir_out'])

    return Bunch(
        arg=arg,
        path=paths,
        random_seed=random_seed,
        timer=Timer(),
    )


class ProcessObject(object):
    def __init__(self):
        self.count = 0
        self.list = collections.defaultdict(list)

    def process(self, obj):
        'tabulate accurate by model_spec'
        verbose = True
        pdb.set_trace()
        self.count += 1
        if verbose or self.count % 1000 == 1:
            print obj
        self.list['query_index'].append(obj.id.query_index)
        self.list['model_spec'].append(str(obj.id.model_spec))
        self.list['predicted_feature_name'].append(obj.id.predicted_feature_name)
        self.list['prediction'].append(obj.payload.predicted_feature_value)
        self.list['actual'].append(obj.payload.actual_value)
        self.list['is_naive'].append(obj.id.model_spec.name == 'n')

    def as_dataframe(self):
        print 'stub: write me'
        pdb.set_trace()
        result = pd.DataFrame(
            data=self.list,
            index=self.list['query_index'],
        )
        return result


def on_EOFError(e):
    print 'EOFError', e
    return


def on_ValueError(e):
    pdb.set_trace()
    print 'ValueError', e
    if e.args[0] == 'insecure string pickle':
        return
    else:
        raise e


def do_work(control):
    'create csv file that summarizes all actual and predicted prices'
    # BODY STARTS HERE
    # determine training and testing transactions
    pdb.set_trace()
    applied_data_science.lower_priority.lower_priority()  # try to give priority to interactive tasks

    # read input file record by record
    process_object = ProcessObject()
    applied_data_science.pickle_utilities.unpickle_file(
        path=control.path['in_file'],
        process_unpickled_object=process_object.process,
        on_EOFError=on_EOFError,
        on_ValueError=on_ValueError,
    )
    print 'processed %d results' % process_object.count
    print 'writing', control.path['out_file']
    process_object.as_dataframe.to_csv(control.path['out_file'])


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path['out_log'])  # now print statements also write to the log file
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
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()

    main(sys.argv)
