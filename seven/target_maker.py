'''make targets from trace prints
'''
import collections
import numpy as np
import pandas as pd
import pdb

import seven.InterarrivalTime


class TargetMaker(object):
    def __init__(self):
        self.data = collections.defaultdict(list)  # Dict[target_name, target_value]
        self.indices = []
        self.interarrivaltime = collections.defaultdict(lambda: seven.InterarrivalTime.InterarrivalTime())

    def append_targets(self, trace_index, trace_record):
        'append info to self.data and self.indices; return None or err'
        def append_to_data(targets_values):
            for target, value in targets_values.iteritems():
                self.data[target]. append(value)
            return None

        def append_to_indices(trace_index):
            self.indices.append(trace_index)

        iat = self.interarrivaltime[trace_record['cusip']]
        interarrival_seconds, err = iat.interarrival_seconds(trace_index, trace_record)
        if err is not None:
            return 'interarrivaltime: %s' % err
        targets_values, err = self._make_targets_values(trace_index, trace_record)
        if err is not None:
            return err
        targets_values['id_interarrival_seconds'] = interarrival_seconds  # PLAN: will become a target
        append_to_data(targets_values)
        append_to_indices(trace_index)
        return None

    def get_targets_dataframe(self):
        'return DataFrame'
        result = pd.DataFrame(
            data=self.data,
            index=self.indices,
        )
        return result

    def _make_targets_values(self, trace_index, trace_record):
        'return (dict, err)'
        oasspread = trace_record['oasspread']
        if np.isnan(oasspread):
            return None, 'oasspread is NaN'
        trade_type = trace_record['trade_type']
        assert trade_type in ('B', 'D', 'S')
        result = {
            'id_trace_index': trace_index,
            'id_trade_type': trade_type,
            'id_oasspread': oasspread,
            'id_effectivedatetime': trace_record['effectivedatetime'],
            'id_quantity': trace_record['quantity'],
            'target_oasspread_B': oasspread if trade_type == 'B' else np.nan,
            'target_oasspread_D': oasspread if trade_type == 'D' else np.nan,
            'target_oasspread_S': oasspread if trade_type == 'S' else np.nan,
        }
        return result, None

if False:
    pdb
