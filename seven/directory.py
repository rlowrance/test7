def directory(name):
    '''Return path to specified directory in file system.

    Parameters
    ----------
    name : string
      name of the directory, one of cells, input, log, working

    Returns
    -------
    string: path to the named directory, ending with a "/"

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

    root = '../'  # root of this project
    if name == 'cells' or name == 'cv-cell' or name == 'cv-cell-natural':
        return root + 'data/working/cv-cell/'
    elif name == 'cv-cell-rescaled':
        return root + 'data/working/cv-cell-rescaled/'
    elif name == 'log':
        return root + 'data/working/log/'
    elif name == 'working':
        return root + 'data/working/'
    elif name == 'src':
        return root + 'src/'
    else:
        raise ValueError(name)

if __name__ == '__main__':
    import unittest
    # import pdb

    class TestDirectory(unittest.TestCase):
        def is_ok(self, cell_name):
            # pdb.set_trace()
            s = directory(cell_name)
            if False:
                print(cell_name, s)
            self.assertTrue(isinstance(s, str))
            self.assertTrue(s.endswith('/'))

        def test_cells(self):
            self.is_ok('cells')

        def test_input(self):
            self.is_ok('input')

        def test_log(self):
            self.is_ok('log')

        def test_working(self):
            self.is_ok('working')

    unittest.main()
