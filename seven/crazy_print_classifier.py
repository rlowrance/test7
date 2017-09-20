import collections
import math
import pdb

from . import variance


class CrazyPrintClassifier:
    def __init__(self, hyperparameter, verbose=False):
        self._number_variances_per_cusip = hyperparameter['number_variances_per_cusip']
        self._max_deviation = hyperparameter['max_deviation']
        self._min_seen = hyperparameter['min_seen']
        self._verbose = verbose

        self._cusip_variance = {}
        self._seen = collections.Counter()

    def __repr__(self):
        return 'CrazyPrintClassifier(%d, %f, %s)' % (
            self._number_variances_per_cusip,
            self._max_deviation,
            self._cusip_variance,
            )

    def _vp(self, *args):
        if self._verbose:
            print(*args)

    def is_crazy(self, cusip, oasspread):
        def restart(i):
            self._cusip_variance[cusip][i] = variance.Online()

        def update(i):
            self._cusip_variance[cusip][i].update(oasspread)

        def is_outlier(i):
            v = self._cusip_variance[cusip][i]
            if v.get_n() >= self._min_seen and v.has_values():
                sample_variance = v.get_sample_variance()
                if sample_variance > 0:
                    delta = v.get_mean() - oasspread
                    z = abs(delta) / math.sqrt(sample_variance)
                    if z > self._max_deviation:
                        self._vp('outlier found', i, cusip, oasspread, z, v)
                        # pdb.set_trace()
                        return True
                else:
                    return False
            else:
                return False

        self._vp('is_crazy entered: %s %f' % (cusip, oasspread))
        if cusip not in self._cusip_variance:  # create variance data structures lazily
            self._cusip_variance[cusip] = collections.deque(maxlen=self._number_variances_per_cusip)
            for i in range(self._number_variances_per_cusip):
                self._cusip_variance[cusip].append(variance.Online())

        result = False
        if is_outlier(0):
            if is_outlier(1):
                restart(0)
                result = True
        update(0)
        update(1)
        if result:
            self._vp('crazy found')
            self._vp(0, self._cusip_variance[cusip][0])
            self._vp(1, self._cusip_variance[cusip][1])
            # pdb.set_trace()
        return result
