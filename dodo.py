# invocations
#  doit help
#  doit help run
#  doit list
#  doit cusips:filename*  # must contain a star, leave out the task_ prefix
#  doit -n 3  # run 3 processes
import pdb
from pprint import pprint

import cusips
import features
import seven.path
import targets

DOIT_CONFIG = {
    'check_file_uptodate': 'md5',
}

midpredictor = seven.path.midpredictor_data()
working = seven.path.working()


def show_command(task):
    return 'executing actions for task %s' % task.name


def make_task_result(doit_value):
    result = {
        'title': show_command,
        'verbosity': 2,
        'actions': doit_value.actions,
        'targets': doit_value.targets,
        'file_dep': doit_value.file_dep,
    }
    return result


def task_cusips_msft():
    return make_task_result(cusips.Doit('msft'))


def task_cusips_orcl():
    return make_task_result(cusips.Doit('orcl'))


def task_features_msft():
    return make_task_result(features.Doit('msft'))


def task_features_orcl():
    return make_task_result(features.Doit('orcl'))


def task_targets_msft():
    return make_task_result(targets.Doit('msft'))


def task_targets_orcl():
    return make_task_result(targets.Doit('orcl'))


if False:  # avoid pyflakes warnings
    pdb
    pprint