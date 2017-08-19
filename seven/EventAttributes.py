'''an Event may generate EventAttributes

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''
import pdb


class EventAttributes(object):
    'a dictionary-like object with feature names (str) as keys and non-None values'
    'has a dictionary with restriction on keys and values'
    def __init__(self, *args, **kwargs):
        self.value = dict(*args, **kwargs)

    def __getitem__(self, key):
        assert isinstance(key, str)
        return self.value[key]

    def __setitem__(self, key, value):
        assert isinstance(key, str)
        assert value is not None
        self.value[key] = value

    def __repr__(self):
        return 'EventAttributes(%s)' % self.value

    def __str__(self):
        return 'EventAttributes(%d items)' % len(self.value)

    def pp(self):
        'pretty print'
        print '{',
        for k in sorted(self.value.keys()):
            print k, ': ', self.value[k], ','
        print '}'

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).iteritems():
            self.value[k] = v


if __name__ == '__main__':
    pdb
