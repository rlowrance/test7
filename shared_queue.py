import abc
import copy
import pika
import unittest

import shared_message


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
    def __init__(self, connection_parameters: pika.ConnectionParameters, queue_name: str):
        pass

    def __next__(self):
        # rewrite me
        if True:
            return None  # rewrite me
        else:
            raise StopIteration

    def close(self):
        pass


class ReaderFile(Reader):
    def __init__(self, path):
        self._path = path
        self._file = open(path)

    def __next__(self):
        'return str or raise StopIteration'
        line = self._file.readline()
        if len(line) == 0:
            raise StopIteration
        return line[:-1]  # remove final \n supplied by readlie(_

    def close(self):
        self._file.close()


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
    def write(self, routing_key: str, message: shared_message.Message):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class WriterFile(Writer):
    def __init__(self, path):
        self._file = open(path, 'w')

    def write(self, routing_key: str, message: shared_message.Message):
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
