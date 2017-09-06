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
import collections
import pdb
import unittest


class Windowed(object):
    'computation on the most recent window of items'
    def __init__(self, window_size):
        self.window_size = window_size
        self.items = collections.deque([], window_size)

    def __repr__(self):
        return 'Windowed(%s' % self.items

    def append(self, item):
        self.items.append(item)

    def sum(self):
        'return sum of items in the window'
        sum = 0
        for item in self.items:
            sum += item
        return sum


class TestWindowed(unittest.TestCase):
    def test_window_size_1(self):
        w = Windowed(1)
        tests = (
            (None, 0),
            (1, 1),
            (2, 2),
            (3, 3),
        )
        for test in tests:
            item, expected_sum = test
            if item is not None:
                w.append(item)
            self.assertEqual(expected_sum, w.sum())

    def test_window_size_2(self):
        w = Windowed(2)
        tests = (
            (None, 0),
            (1, 1),
            (2, 3),
            (3, 5),
        )
        for test in tests:
            item, expected_sum = test
            if item is not None:
                w.append(item)
            self.assertEqual(expected_sum, w.sum())


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
