'''lower priority of current process

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
import os
import pdb


def lower_priority():
    # ref: http://stackoverflow.com/questions/1023038/change-process-priority-in-python-cross-platform
    assert os.name in ('nt', 'posix'), os.name
    if os.name == 'nt':
        import win32api
        import win32process
        import win32con
        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
    elif os.name == 'posix':
        os.nice(1)

if __name__ == '__main__':
    lower_priority()