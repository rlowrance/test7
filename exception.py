'all machine learning exceptions'
# ref: https://julien.danjour.info/blog/2016/python-exceptions.guide
import copy
import typing

############################################
# base class for all other Exception classes
############################################


class MachineLearningException(Exception):
    'basic exception for machine learning code'
    def __init__(self, msg: str):
        if msg is None:
            msg = 'an exception was raised by the machine learning code'
        super(MachineLearningException, self).__init__(msg)

#############################################
# specific machine learning exception classes
#############################################


class FeatureVector(MachineLearningException):
    def __init__(self, msg: str):
        super(FeatureVector, self).__init__(msg)

       
class NoFeatures(MachineLearningException):
    def __Init__(self, msg: str):
        super(NoFeatures, self).__init__(msg)
        

class NoMessageWithCusip(MachineLearningException):
    def __init__(self, cusip: str, msgs: typing.List):
        super(NoMessageWithCusip, self).__init__(
            msg='cusip %s not seen in %d messages' % (cusip, len(msgs)),
        )
        self.cusip = cusip
        self.msgs = copy.copy(msgs)
        

class NoPriorEventWithCusipAndRtt(MachineLearningException):
    def __init__(self, cusip: str, rtt: str, msgs: typing.List):
        super(NoPriorEventWithCusipAndRtt, self).__init__(
            msg='primary cusip %s with rtt %s not seen in %d messages' % (cusip, rtt, len(msgs)),
        )
        self.cusip = cusip
        self.rtt = rtt
        self.msgs = copy.copy(msgs)
        
