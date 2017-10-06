'''messages bodies in queues

The bodies of messages in queues are JSON-formatted strings. The implied
JSON object is a dictionary with these key/value pairs:

- key="message_id", valeu=a str, the name of the class (see below)
- key="payload", value=a JSON-formatted string

Messags on RabbitMQ queues have additional fields called
headers. Those fields are invisible to this code.
'''
import datetime
import json
import pdb
import unittest

    
class Message:
    def __init__(self, message_id: str):
        self.message_id = message_id

        
class SetOtrCusip(Message):
    def __init__(self, primary_cusip: str, otr_level: int, otr_cusip: str):
        assert isinstance(otr_level, int)
        assert otr_level >= 1
        super(SetOtrCusip, self).__init__("SetOtrCusip")
        self.primary_cusip = primary_cusip
        self.otr_level = otr_level
        self.otr_cusip = otr_cusip

    @staticmethod
    def from_dict(d: dict):
        'factory method'
        return SetOtrCusip(
            d['primary_cusip'],
            d['otr_level'],
            d['otr_cusip'],
            )

    def __str__(self):
        'return JSON-formatted string'
        return json.dumps({
            'message_id': self.message_id,
            'primary_cusip': self.primary_cusip,
            'otr_level': self.otr_level,
            'otr_cusip': self.otr_cusip,
            })


class SetPrimaryCusip(Message):
    def _init__(self, cusip: str):
        super(SetPrimaryCusip, self).__init__("SetPrimaryCusip")
        self.cusip = cusip

        
class StartOutput(Message):
    def __init__(self):
        super(StartOutput, self).__init__("StartOutput")

    @staticmethod
    def from_dict(d: dict):
        return StartOutput()
    
    def __str__(self):
        'return JSON-formatted string'
        return json.dumps({
            'message_id': self.message_id,
            })
    
        
class TracePrint(Message):
    def __init__(
            self, cusip: str,
            issuepriceid: str,
            datetime: datetime.datetime,
            oasspread: float,
            cancellation_probability: float):
        super(TracePrint, self).__init__("TracePrint")
        self.cusip = cusip
        self.issuepriceid = issuepriceid
        self.datetime = datetime
        self.oasspread = oasspread
        self.cancellation_probability = cancellation_probability
        

class TracePrintCancel(Message):
    def __init__(self, issuepriceid: str):
        super(TracePrintCancel, self).__init__("TracePrintCancel")
        self.issuepriceid = issuepriceid


class SetVersionETL(Message):
    def __init__(self, version: str):
        super(SetVersionETL, self).__init__("SetVersionETL")
        self.version = version


class SetVersionFeatures(Message):
    def __init__(self, version: str):
        super(SetVersionFeatures, self).__init("SetVersionFeatures")
        self.version = version


class SetVersionMachineLearning(Message):
    def __init__(self, version: str):
        super(SetVersionMachineLearning, self).__init__("SetVersionMachineLearning")
        self.version = version

        
###################################################################
def from_string(s: str):
    'return an appropriate subclass of Message'
    obj = json.loads(s)
    assert isinstance(obj, dict)
    message_id = obj['message_id']
    if message_id == 'StartOutput':
        return StartOutput.from_dict(obj)
    if message_id == 'SetOtrCusip':
        return SetOtrCusip.from_dict(obj)
    assert False, 'message_id %s is not known' % message_id
    

####################################################################
class Test(unittest.TestCase):
    def test_StartOutput(self):
        pdb.set_trace()
        m = StartOutput()
        s = str(m)
        m2 = from_string(s)
        assert isinstance(m2, StartOutput)

    def test_SetOtrCusip(self):
        test_primary_cusip = "primary"
        test_otr_level = 2
        test_otr_cusip = "otr"
        m = SetOtrCusip(
            primary_cusip=test_primary_cusip,
            otr_cusip=test_otr_cusip,
            otr_level=test_otr_level,
            )
        s = str(m)
        m2 = from_string(s)
        assert isinstance(m2, SetOtrCusip)
        self.assertEqual(m.primary_cusip, m2.primary_cusip)
        self.assertEqual(m.otr_cusip, m2.otr_cusip)
        self.assertEqual(m.otr_level, m2.otr_level)

##################################################################
if __name__ == '__main__':
    unittest.main()
