'''
Copyright 2017 Roy E. Lowrance

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on as "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing premission and
limitation under the license.
'''
import datetime
import os
import sys
import pdb


from . import directory


if False:
    pdb.set_trace()  # avoid warning message from pyflakes


class Logger(object):
    # from stack overflow: how do i duplicat sys stdout to a log file in python

    def __init__(self, logfile_path=None, logfile_mode='w', base_name=None):
        def path(s):
            return directory('log') + s + '-' + datetime.datetime.now().isoformat('T') + '.log'
        self.terminal = sys.stdout
        if os.name == 'posix':
            clean_path = logfile_path.replace(':', '-') if base_name is None else path(base_name)
        else:
            clean_path = logfile_path
        self.log = open(clean_path, logfile_mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        pass


if False:
    # usage example
    sys.stdout = Logger('path/to/log/file')
    # now print statements write on both stdout and the log file
