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
import pickle as pickle
import os.path
import pdb


def unpickle_file(
    path=None,
    process_unpickled_object=None,
    on_EOFError=None,
    on_ValueError=None,
    on_FileNotExists=None,
    ):
    'unpickle each object in the file at the path'
    # NOTE: caller must define the type of the object by, for example, importing a class
    if not os.path.isfile(path):
        if on_FileNotExists is None:
            return  # simulate end of file
        else:
            on_FileNotExists('file does not exist: %s' % path)
            return
    with open(path, 'r') as f:
        unpickler = pickle.Unpickler(f)
        try:
            while True:
                obj = unpickler.load()
                process_unpickled_object(obj)
        except EOFError as e:
            if on_EOFError is None:
                raise e
            else:
                on_EOFError(e)
                return
        except ValueError as e:
            if on_ValueError is None:
                raise e
            else:
                on_ValueError(e)
                return


if False:
    # avoid pyflake8 warnings
    pdb