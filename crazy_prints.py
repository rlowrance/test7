'''determine all crazy prints identified by the security master

A "crazy print" is a trace print with an out-of-pattern oasspread

A crazy print is found by feature_version 2 and higher of
  seven.make_event_attributes.Trace()

INVOCATION
 python crazy_prints.py --debug

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''

import argparse
import collections
import csv
import datetime
import pdb
import pprint
import random
import sys

import seven.build
import seven.debug
import seven.dirutility
import seven.EventAttributes
import seven.event_readers
import seven.logging
import seven.Logger
import seven.make_event_attributes
import seven.Timer

pp = pprint.pprint


class Control:
    def __init__(self, arg, path, random_seed, timer):
        self.arg = arg
        self.path = path
        self.random_seed = random_seed
        self.timer = timer

    def __repr__(self):
        return 'Control(arg=%s, n path=%d, random_seed=%f, timer=%s)' % (
            self.arg,
            len(self.path),
            self.random_seed,
            self.timer,
        )

    @staticmethod
    def make(argv):
        'return a new Control'
        parser = argparse.ArgumentParser()
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

        paths = seven.build.crazy_prints(
            test=arg.test,
        )
        seven.dirutility.assure_exists(paths['dir_out'])

        timer = seven.Timer.Timer()

        return Control(
            arg=arg,
            path=paths,
            random_seed=random_seed,
            timer=timer,
        )


class MockControl:
    def __init__(self, path_trace, feature_version):
        self.path = {}
        self.path['in_trace'] = path_trace
        self.arg = argparse.ArgumentParser().parse_args()
        self.arg.feature_version = feature_version
        self.event_reader = {
            'trace': {
                'number_variances_per_cusip': 2,
                # 'max_deviation': 2.95,
                # 'max_deviation': 10.00,
                # 'max_deviation': 30.00,
                'max_deviation': 100.00,
                'min_seen': 10,
            },
        }

    def __str__(self):
        return ''.join([
            'Mockcontrol(',
            '\n.path=%s' % self.path,
            '\n.arg=%s' % self.arg,
            '\n.event_reader=%s' % self.event_reader,
            '\n)',
            ])
        # print('Mockcontrol')
        # print(' .path=%s' % self.path)
        # print(' .arg=%s' % self.arg)
        # print(' .event_reader=')
        # pdb.set_trace()
        # for k, v in self.event_reader.items():
        #     for k2, v2 in v.items():
        #         print('   ["%s"]["%s"]=%s' % (k, k2, v2))


def summarize_event(event, err):
    dt = datetime.datetime(
        int(event.year),
        int(event.month),
        int(event.day),
        int(event.hour),
        int(event.minute),
        int(event.second),
    )
    return {
        'datetime': dt,
        'cusip': event.payload['cusip'],
        'source_identifier': event.source_identifier,
        'oasspread': float(event.payload['oasspread']),
        'is_crazy': err is not None and err == 'crazy',
        }


def test(mock_control):
    trace_event_reader = seven.event_readers.Trace(mock_control)
    attempt = 0
    reader_errs = collections.Counter()
    seven.logging.verbose_warning = False  # suppress printing of warning from event readers
    classifications = collections.defaultdict(list)
    while True:
        try:
            attempt += 1
            event, err = trace_event_reader.__next__()
            if event is not None:
                classifications[event.payload['cusip']].append(summarize_event(event, err))
            if err is not None:
                reader_errs[err] += 1
                # print('no event for attempt %d err %s file %s' % (attempt, err, path_trace))

        except StopIteration:
            break
    return reader_errs, classifications


def do_work(control):
    all_crazies = []
    for k, path in control.path.items():
        if k.startswith('in_trace_'):
            ticker = k.split('_')[2]
            print('\n')
            for feature_version in (1, 2):
                mock_control = MockControl(path, feature_version)
                counters, classifications = test(mock_control)
                print('reasons rejected feature_version %s %s' % (feature_version, k))
                for k in sorted(counters.keys()):
                    print(' %50s: %d' % (k, counters[k]))
                # write csv for the ticker
                if feature_version == 1:
                    continue
                print('')
                with open(control.path['out_%s' % ticker], 'w') as f:
                    writer = csv.DictWriter(
                        f,
                        ['cusip', 'datetime', 'source_identifier', 'oasspread', 'is_crazy'],
                        lineterminator='\n',
                    )
                    writer.writeheader()
                    recent = collections.deque([], maxlen=10)  # 10 trace prints preceeding a crazy print
                    for cusip, infos in classifications.items():
                        for info in infos:
                            row = {
                                'cusip': cusip,
                                'datetime': info['datetime'],
                                'source_identifier': info['source_identifier'],
                                'oasspread': info['oasspread'],
                                'is_crazy': 'crazy' if info['is_crazy'] else '',
                            }
                            writer.writerow(row)
                            recent.append(row)
                            if row['is_crazy']:
                                print('')
                                print('found crazy')
                                for r in recent:
                                    print('%s %s %s %s %0.2f %s' % (
                                        ticker,
                                        r['cusip'],
                                        r['datetime'],
                                        r['source_identifier'],
                                        r['oasspread'],
                                        'crazy' if r['is_crazy'] else '',
                                    ))
                                all_crazies.append((ticker, cusip, row))
    print(control)
    print(mock_control)
    print('\nFound %d crazy oasspreads' % len(all_crazies))


def main(argv):
    control = Control.make(argv)
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
