import os
import pdb


import seven.path


def task_build_order_imbalance_orcl_sample1():
    pdb.set_trace()
    pgm = 'order-imbalance'
    lookback = '10'
    spread = '2'
    infile = 'orcl_order_imb_sample1'
    out = '%s-%s-%s' % (infile, lookback, spread)
    return {
        'actions': [
            'python %s %s %s %s' % (pgm + '.py', infile + '.csv', lookback, spread),
        ],
        'file_dep': [
            os.path.join(seven.path.midpredictor_data(), infile + '.csv'),
            os.path.join('seven', 'OrderImbalance2.py'),
            os.path.join(pgm + '.py')
        ],
        'targets': [
            os.path.join(seven.path.working(), pgm, out + '.csv'),
            os.path.join(seven.path.working(), pgm, out + '.pickle'),
        ],
    }


def task_build_order_imbalance_orcl_sample2():
    pdb.set_trace()
    pgm = 'order-imbalance'
    lookback = '10'
    spread = '2'
    infile = 'orcl_order_imb_sample1'
    out = '%s-%s-%s' % (infile, lookback, spread)
    return {
        'actions': [
            'python %s %s %s %s' % (pgm + '.py', infile + '.csv', lookback, spread),
        ],
        'file_dep': [
            os.path.join(seven.path.midpredictor_data(), infile + '.csv'),
            os.path.join('seven', 'OrderImbalance2.py'),
            os.path.join(pgm + '.py')
        ],
        'targets': [
            os.path.join(seven.path.working(), pgm, out + '.csv'),
            os.path.join(seven.path.working(), pgm, out + '.pickle'),
        ],
    }
