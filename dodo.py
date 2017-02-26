import os
import pdb


import order_imbalance3
import seven.path

DOIT_CONFIG = {
    'check_file_uptodate': 'md5',
}


def show_command(task):
    return 'executing actions for task %s' % task.name


def program_features(filename):
    program_basename = 'order_imbalance3'
    return {
        'actions': [
            'python order_imbalance3.py %s %s' % (filename, typical_bid_offer),
        ],
        'targets': [
            os.path.join(seven.path.working(), program_basename, order_imbalance3.get_cusip_filename(filename)),
        ],
        'file_dep': [
            os.path.join(seven.path.midpredictor_data(), filename),
            os.path.join(program_basename + '.py'),
        ],
        'title': show_command,
        'verbosity': 2,
    }


def task_features_orcl_sample1():
    return program_features('orcl_order_imb_sample1.csv')


def task_features_orcl_sample2():
    return program_features('orcl_order_imb_sample2.csv')
