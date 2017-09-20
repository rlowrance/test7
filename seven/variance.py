'''
various ways to compute variances

INVOCATION TO RUN UNIT TESTS
  cd src
  python -mseven.variance

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com
You may not use this file except in compliance with a License.
'''
import collections
import pdb
import unittest


class Online:
    '''ref: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm'''
    def __init__(self):
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._recent = collections.deque(maxlen=10)

    def __repr__(self):
        if self.has_values():
            return 'Online(%d, %0.2f, %0.2f)[%s]' % (
                self._n,
                self.get_mean(),
                self.get_sample_variance(),
                self._recent,
            )
        else:
            return 'Online(%d)' % self._n

    def update(self, x: float) -> None:
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._m2 += delta * delta2
        self._recent.append(x)

    def has_values(self):
        return self._n >= 2

    def get_n(self) -> int:
        return self._n
    
    def get_mean(self) -> float:
        return float('nan') if self._n < 2 else self._mean

    def get_sample_variance(self) -> float:
        return float('nan') if self._n < 2 else self._m2 / (self._n - 1)

    def get_population_variance(self) -> float:
        return float('nan') if self._n < 2 else self._m2 / self._n


class Test(unittest.TestCase):
    def test1(self):
        # ref for first test case: http://www.mathsisfun.com/data/standard-deviation.html
        tests = (
            ((600, 470, 170, 430, 300), (394, 27130, 21704)),
        )
        for test in tests:
            xs, expected = test
            expected_mean, expected_sample_variance, expected_population_variance = expected
            o = Online()
            for x in xs:
                o.update(x)
            self.assertAlmostEqual(expected_mean, o.get_mean())
            self.assertAlmostEqual(expected_sample_variance, o.get_sample_variance())
            self.assertAlmostEqual(expected_population_variance, o.get_population_variance())


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
