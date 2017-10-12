import abc
import copy
import pika
import pdb
import unittest

import message


class Reader(abc.ABC):
    # TODO: review design of the Reader class
    def __iter__(self):
        return self

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def __next__(self):
        # should behave like the equivalent pika call, so needs a call back function
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
        self._file = open(path)

    def __next__(self):
        'return str or raise StopIteration'
        pdb.set_trace()
        try:
            line = self._file.readline().rstrip('\n\r')
            return line
        except EOFError:
            raise StopIteration

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


class Writer(abc.ABC):
    @abc.abstractmethod
    def write(self, routing_key: str, message: message.Message):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class WriterFile(Writer):
    def __init__(self, path):
        self._file = open(path, 'w')

    def write(self, routing_key: str, message: message.Message):
        # ignore the routing key
        self._file.write(str(message))
        self._file.write('\n')

    def close(self):
        self._file.close()


###########################################################
class Test(unittest.TestCase):
    def test1(self):
        pass


if __name__ == '__main__':
    unittest.main()
