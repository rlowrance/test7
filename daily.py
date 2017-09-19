'''run daily development jobs

1. test_train on a date
2. accuracies-* on the new date

INVOCATION
  python daily.py trade_date jobs upstream_version feature_version [--secmaster path] [--debug] [--debugtesttrain] [--trace] [--test]
where
 trade_date is the trade of the trace prints: For example< 2017-08-24
 jobs is the number of jobs to run in parallel
 upstream_version: any sequence of characters without spaces, identifies the versions of all the input files
   The stream source specifies this invocation parameter
   It is designed to reflect different versions of the input file streams.
 feature_version: any sequence of characters without spaces, identifies the version of the feature set used
   The developer of this program sets the feature set
   The feature_version will correspond to a HEAD in git
   The feature_version could be a git tag, but that requires the developer to make it so.
  {--secmasst path]  specifies the path to the security master
                     default value is: ~/Dropbox/MidPredictor/automatic_feeds/secmaster.csv

EXAMPLES OF INVOCATIONS
  python daily.py 2017-08-24 1 1 1 # one job
  python dail6.py 2017-08-24 16  1 2 # 16 jobs
  python dail6.py 2017-08-24 16  -- secmaster /home/ubuntu/secmaster.csv # 16 jobs

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a license.
'''
import argparse
import copy
import csv
import multiprocessing
import os
import pdb
import pprint
import random
import sys

import seven.arg_type
import seven.build
import seven.debug
import seven.dirutility
import seven.Logger
import seven.logging
import seven.path
import seven.Timer

pp = pprint.pprint


class Control:
    def __init__(self, arg, path, random_seed, timer):
        self.arg = arg
        self.path = path
        self.random_seed = random_seed
        self.timer = timer

    def __repr__(self):
        return 'Control(arg=%s, len(path)=%d, random_seed=%f, timer=%s)' % (
            self.arg,
            len(self.path),
            self.random_seed,
            self.timer,
        )

    def new_with_path(self, new_logical_name, new_location):
        result = copy.copy(self)
        result.path[new_logical_name] = new_location
        return result

    @classmethod
    def make_control(cls, argv):
        'return a Control'
        parser = argparse.ArgumentParser()
        parser.add_argument('trade_date', type=seven.arg_type.date)
        parser.add_argument('jobs', type=seven.arg_type.positive_int)
        parser.add_argument('upstream_version', type=str)
        parser.add_argument('feature_version', type=str)
        parser.add_argument('--secmaster', action='store')
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--debugtesttrain', action='store_true')
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

        paths = seven.build.daily(
            arg.trade_date,
            arg.jobs,
            arg.upstream_version,
            arg.feature_version,
            debug=arg.debug,
            test=arg.test,
            trace=arg.trace,
        )
        seven.dirutility.assure_exists(paths['dir_out'])            

        timer = seven.Timer.Timer()

        control = cls(
            arg=arg,
            path=paths,
            random_seed=random_seed,
            timer=timer,
        )
        return control


def handle_return_codes(return_codes, commands, what):
    print('all %s workers returned 0 return codes except these' % what)
    max_return_code = report_non_zero_return_codes(return_codes, commands)
    if max_return_code > 0:
        print('stopping because max return code (%d) is positive' % max_return_code)
        sys.exit(max_return_code)


def make_commands_analysis(trade_date, whats, upstream_version, feature_version):
    'return List[command:str]'
    result = []
    for what in whats:
        result.append('python analysis_%s.py dev %s %s %s %s --debug' % (
            what,
            trade_date,
            trade_date,
            upstream_version,
            feature_version,
        ))
    return result


def make_commands_test_train(arg_secmaster, trade_date, debug_test_train, upstream_version, feature_version):
    'return List[command: str]'
    path_secmaster = (
        seven.path.input(issuer=None, logical_name='security master') if arg_secmaster is None else
        arg_secmaster
    )
    result = []
    with open(path_secmaster) as f:
        dict_reader = csv.DictReader(f)
        for row in dict_reader:
            result.append('python test_train.py %s %s %s %s %s %s %s %s %s %s' % (
                row['ticker'],
                row['CUSIP'],
                'oasspread',
                'grid5',
                '2017-04-01',
                str(trade_date),
                str(trade_date),
                upstream_version,
                feature_version,
                '--debug' if debug_test_train else ''
            ))
    return result


def report_non_zero_return_codes(return_codes, commands):
    'return max non-zero return code or zero; print all non-zero rc and command'
    max_return_code = 0
    for index, return_code in enumerate(return_codes):
        max_return_code = max(max_return_code, return_code)
        if return_code != 0:
            print(' rc: %5d command: %s' % (
                return_code,
                commands[index],
            ))
    return max_return_code


def worker(command):
    debug = True
    debug = False
    print('worker %20s %4s %4s' % (
        multiprocessing.current_process().name,
        os.getppid(),  # parent process id
        os.getpid(),   # process id
    ))
    if debug:
        print('pretending to execute: %s' % command)
        return 0
    else:
        print('executing: %s' % command)
        return os.system(command)


def do_work(control):
    # applied_data_science.lower_priority.lower_priority()
    p = multiprocessing.Pool(control.arg.jobs)

    # run the test_train program for each cusip (for now, about 250 of them)
    test_train_commands = make_commands_test_train(
        arg_secmaster=control.arg.secmaster,
        trade_date=control.arg.trade_date,
        debug_test_train=control.arg.debugtesttrain,
        upstream_version=control.arg.upstream_version,
        feature_version=control.arg.feature_version,
        )
    test_train_return_codes = p.map(worker, test_train_commands)
    handle_return_codes(
        test_train_return_codes,
        test_train_commands,
        'test_train'
    )

    # run the analysis programs
    analysis_commands = make_commands_analysis(
        control.arg.trade_date,
        ('accuracy', 'experts', 'importances'),
        control.arg.upstream_version,
        control.arg.feature_version,
        )
    analysis_return_codes = p.map(worker, analysis_commands)
    handle_return_codes(
        analysis_return_codes,
        analysis_commands,
        'analysis'
    )
    return None


def main(argv):
    control = Control.make_control(argv)
    sys.stdout = seven.Logger.Logger(control.path['out_log'])  # now print statements also write to the log file
    print(control)
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.arg.test:
        print('DISCARD OUTPUT: test')
    # print control
    print(control.arg)
    print('done')
    return


if __name__ == '__main__':
    main(sys.argv)
