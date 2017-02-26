import cPickle as pickle
import os
import pdb
from pprint import pprint

import cusips
import features
import seven.path

DOIT_CONFIG = {
    'check_file_uptodate': 'md5',
}

midpredictor = seven.path.midpredictor_data()
working = seven.path.working()


def show_command(task):
    return 'executing actions for task %s' % task.name

common_return = {
    'title': show_command,
    'verbosity': 2,
}


def program_cusips(ticker):
    doit = cusips.Doit(ticker)
    return common_return.update({
        'actions': doit.actions,
        'targets': doit.targets,
        'file_dep': doit.file_dep,
    })


def program_features(ticker):
    doit = features.Doit(ticker)
    return common_return.update({
        'actions': doit.actions,
        'targets': doit.targets,
        'file_dep': doit.file_dep,
    })


def task_cusips_orcl():
    return program_cusips('orcl')


def task_cusips_msft():
    return program_cusips('msft')


def task_features_orcl():
    return program_features('orcl')


def task_features_msft():
    return program_features('msft')
