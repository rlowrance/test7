'measure wall clock elapsed time'
import collections
import csv
import datetime
import pdb
import time
from typing import Dict


accumulated_lap_time = collections.defaultdict(datetime.timedelta)  # type: Dict[str, datetime.timedelta]
lap_count = collections.Counter()  # type: Dict[str, int]


last_lap_end_time = datetime.datetime.now()


def end_lap(lap_name: str) -> None:
    global last_lap_end_time
    now = datetime.datetime.now()
    accumulated_lap_time[lap_name] += now - last_lap_end_time
    lap_count[lap_name] += 1
    last_lap_end_time = now


def write_csv(path: str, also_print=False):
    def append_to_csv(lap_name, total_seconds, fraction_total, lap_count, seconds_per_count):
        writer.writerow({
            'lap_name': lap_name,
            'accumulated_lap_seconds': accumulated,
            'fraction_total_seconds': fraction_total,
            'lap_count': count,
            'wallclock_seconds_per_count': seconds_per_count,
        })

    def append_to_print_detail(lap_name, total_seconds, fraction_total, lap_count, seconds_per_count):
        print('%30s %10.6f %10.6f %7d %10.6f' % (
            lap_name,
            total_seconds,
            fraction_total,
            count,
            seconds_per_count
        ))

    def append_to_print_total(total):
        print('%30s %10.6f %10.6f' % ('** TOTALS **', total, 1.0))

    # pass 1 accumulate total seconds
    total = 0.0
    for lap_name, accumulated in accumulated_lap_time.items():
        total += accumulated.total_seconds()

    with open(path, 'w') as f:
        writer = csv.DictWriter(
            f,
            ['lap_name',
             'accumulated_lap_seconds', 'fraction_total_seconds',
             'lap_count', 'wallclock_seconds_per_count',
             ],
            lineterminator='\n',
        )
        writer.writeheader()
        for lap_name in sorted(accumulated_lap_time.keys()):
            accumulated = accumulated_lap_time[lap_name].total_seconds()
            fraction_total = accumulated / total
            count = lap_count[lap_name]
            seconds_per_count = accumulated / count
            append_to_csv(lap_name, accumulated, fraction_total, count, seconds_per_count)
            if also_print:
                append_to_print_detail(lap_name, accumulated, fraction_total, count, seconds_per_count)
        writer.writerow({
            'lap_name': '** TOTALS **',
            'accumulated_lap_seconds': total,
            'fraction_total_seconds': 1.0,
        })
        if also_print:
            append_to_print_total(total)


if __name__ == '__main__':
    if False:
        pdb  # avoid pyflake8 warning
        Dict
    end_lap('startup')
    for i in range(3):
        time.sleep(1)
        end_lap('big computation')
    end_lap('cleanup start')
    time.sleep(1)
    end_lap('cleanup finish')
    write_csv('wallclock_timer_test.csv', also_print=True)
