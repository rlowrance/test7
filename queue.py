import abc
import copy
import pika
import unittest


class Reader(abc.ABC):
    def __iter__(self):
        return self

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstracmethod
    def next(self):
        pass

    
class ReaderRabbit(Reader):
    def __init__(self, connection_paramaters: pika.ConnectionParameters, queue_name: str):
        pass

    def next(self):
        # rewrite me
        if True:
            return None  # rewrite me
        else:
            raise StopIteration


class ReaderFile(Reader):
    def __init__(self, path):
        self._path = path

    def next(self):
        with open(self._path) as f:
            for line in f:
                yield line.rstrip('\n')

    def close(self):
        pass


class ReaderIterable(Reader):
    def __init__(self, iterable):
        self.iterable = copy.copy(iterable)

    def next(self):
        for line in self.iterable:
            yield line

    def close(self):
        pass

    
class Test(unittest.TestCase):
    def test1(self):
        pass


if __name__ == '__main__':
    unittest.main()
