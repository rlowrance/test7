'''utilities for managing directories

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
import string
import unittest


def assure_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  # make all intermediate directories
    return dir_path


def _make_invalid_characters():
    'return translation table suitable for s.translate()'
    bad_linux = '/\x00'
    # bad_macos = ':/'
    bad_ntfs = '"*:<>?\|'   # plus 0x00 - 0x1F plus 0x7F
    bad_os = bad_ntfs + bad_linux  # bad_macos is in these two
    low_codes = '\x01\x02\x03\x04\x05\x06\x07\x08\x09\x10\x11\x12\x13\x14\x15'
    assert len(low_codes) == 15, low_codes
    high_code = '\x7f'
    return bad_os + low_codes + high_code


_invalid_characters = _make_invalid_characters()


def always_valid_filename(s, replacement_char='-'):
    '''return a new string that is acceeptable to Linux, MacOS, and Linux
    ref: https://en.wikipedia.org/wiki/Filename
    '''
    # ref: https://docs.python.org/3.6/library/stdtypes.html?highlight=maketrans#str.maketrans
    table = str.maketrans(_invalid_characters, replacement_char * len(_invalid_characters))
    return s.translate(table)


class Test_always_valid_filename(unittest.TestCase):
    def test1(self):
        s = 'abc\x03/:"*<>?\\|'
        result = always_valid_filename(s)
        self.assertEqual('abc----------', result)


if __name__ == '__main__':
    if False:
        pdb
    unittest.main()
