'''return path as str to logical file or directory'''
import os
import unittest


def dropbox():
    return os.path.join(home(), 'Dropbox')


def home():
    # TODO: make work on unix as well as Windows
    return os.path.join('C:', r'\Users', 'roylo')


def midpredictor_data():
    return os.path.join(dropbox(), 'MidPredictor', 'data')


def working():
    return os.path.join(dropbox(), 'data', '7chord', '7chord-01', 'working')


class TestPath(unittest.TestCase):
    def setUp(self):
        self.verbose = True

    def test_all(self):
        s = dropbox()
        self.assertTrue(isinstance(s, str))

        s = home()
        self.assertTrue(isinstance(s, str))

        s = midpredictor()
        self.assertTrue(isinstance(s, str))

        s = working()
        self.assertTrue(isinstance(s, str))


if __name__ == '__main__':
    unittest.main()
