'visit every output directory created by test_train.py'
import datetime
import os
import pdb


class WalkTestTrainOutputDirectories(object):
    'apply a function to each data-containing directory produced by test_train.py'
    def __init__(self, root_directory):
        self._root_directory = root_directory

    def walk(self, visit):
        for issuer in os.listdir(self._root_directory):
            path_issuer = os.path.join(self._root_directory, issuer)
            for cusip in os.listdir(path_issuer):
                path_cusip = os.path.join(path_issuer, cusip)
                for target in os.listdir(path_cusip):
                    path_target = os.path.join(path_cusip, target)
                    for hpset in os.listdir(path_target):
                        path_hpset = os.path.join(path_target, hpset)
                        for start_events in os.listdir(path_hpset):
                            if not self._is_date(start_events):
                                # skip directories without a date
                                # these are possibly left-over from prior versions of the application
                                # in any case, they were not created by test_train.py
                                continue
                            path_startevents = os.path.join(path_hpset, start_events)
                            for start_predictions in os.listdir(path_startevents):
                                if not self._is_date(start_predictions):
                                    continue
                                path_startpredictions = os.path.join(path_startevents, start_predictions)
                                for stop_predictions in os.listdir(path_startpredictions):
                                    if not self._is_date(stop_predictions):
                                        continue
                                    path_stoppredictions = os.path.join(path_startpredictions, stop_predictions)
                                    invocation_parameters = {
                                        'issuer': issuer,
                                        'target': target,
                                        'hpset': hpset,
                                        'start_events': start_events,
                                        'start_predictions': start_predictions,
                                        'stop_predictions': stop_predictions,
                                    }
                                    visit(
                                        directory_path=path_stoppredictions,
                                        invocation_parameters=invocation_parameters,
                                    )

    def _is_date(self, s):
        'does string s contain a valid date?'
        try:
            year, month, day = s.split('-')
            datetime.date(int(year), int(month), int(day))
            return True
        except Exception:
            return False


if __name__ == '__main__':
    pdb
