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
    def __init__(self, message_type: str):
        self.message_type = message_type

#####################################################
# subsclasses of Message
#####################################################


class SetCusipOtr(Message):
    def __init__(self, primary_cusip: str, otr_level: int, otr_cusip: str):
        assert isinstance(otr_level, int)
        assert otr_level >= 1
        super(SetCusipOtr, self).__init__("SetCusipOtr")
        self.primary_cusip = primary_cusip
        self.otr_level = otr_level
        self.otr_cusip = otr_cusip

    @staticmethod
    def from_dict(d: dict):
        'factory method'
        return SetCusipOtr(
            d['primary_cusip'],
            d['otr_level'],
            d['otr_cusip'],
            )

    def __str__(self):
        'return JSON-formatted string'
        return json.dumps({
            'message_type': self.message_type,
            'primary_cusip': self.primary_cusip,
            'otr_level': self.otr_level,
            'otr_cusip': self.otr_cusip,
            })


class SetCusipPrimary(Message):
    def __init__(self, primary_cusip: str):
        super(SetCusipPrimary, self).__init__("SetPrimaryCusip")
        self.primary_cusip = primary_cusip

    @staticmethod
    def from_dict(d: dict):
        'factor method'
        return SetCusipPrimary(
            d['primary_cusip'],
            )

    def __str__(self):
        'return JSON-formatted string'
        return json.dumps({
            'message_type': self.message_type,
            'primary_cusip': self.primary_cusip,
            })

    
class SetVersion(Message):
    def __init__(self, what: str, version: str):
        super(SetVersion, self).__init__("SetVersion")
        self.what = what
        self.version = version

    @staticmethod
    def from_dict(d: dict):
        return SetVersion(
            d['what'],
            d['version'],
            )

    def __str__(self):
        return json.dumps({
            'message_type': self.message_type,
            'what': self.what,
            'version': self.version,
            })

        
class TracePrint(Message):
    def __init__(
            self,
            cusip: str,
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


class OutputStart(Message):
    def __init__(self):
        super(OutputStart, self).__init__("OutputStart")

    @staticmethod
    def from_dict(d: dict):
        return OutputStart()

    def __str__(self):
        'return JSON-formatted string'
        return json.dumps({
            'message_type': self.message_type,
            })

    
class OutputStop(Message):
    def __init__(self):
        super(OutputStop, self).__init__("OutputStop")

    @staticmethod
    def from_dict(d: dict):
        return OutputStop()

    def __str__(self):
        'return JSON-formatted string'
        return json.dump({
            'message_type': self.message_type,
            })

    
###################################################################
def from_string(s: str):
    'return an appropriate subclass of Message'
    obj = json.loads(s)
    assert isinstance(obj, dict)
    message_type = obj['message_type']
    if message_type == 'OutputStart':
        return OutputStart.from_dict(obj)
    if message_type == 'SetCusipOtr':
        return SetCusipOtr.from_dict(obj)
    if message_type == 'SetPrimaryCusip':
        return SetCusipPrimary.from_dict(obj)
    if message_type == 'SetVersion':
        return SetVersion.from_dict(obj)
    assert False, 'message_type %s is not known' % message_type


####################################################################
class Test(unittest.TestCase):
    def test_OutputStart(self):
        m = OutputStart()
        s = str(m)
        m2 = from_string(s)
        assert isinstance(m2, OutputStart)

    def test_SetCusipOtr(self):
        test_primary_cusip = "primary"
        test_otr_level = 2
        test_otr_cusip = "otr"
        m = SetCusipOtr(
            primary_cusip=test_primary_cusip,
            otr_cusip=test_otr_cusip,
            otr_level=test_otr_level,
            )
        s = str(m)
        m2 = from_string(s)
        assert isinstance(m2, SetCusipOtr)
        self.assertEqual(m.primary_cusip, m2.primary_cusip)
        self.assertEqual(m.otr_cusip, m2.otr_cusip)
        self.assertEqual(m.otr_level, m2.otr_level)

    def test_SetPrimaryCusip(self):
        test_primary_cusip = 'primary'
        m = SetCusipPrimary(
            primary_cusip=test_primary_cusip,
            )
        m2 = from_string(str(m))
        self.assertTrue(isinstance(m2, SetCusipPrimary))
        self.assertEqual(m2.primary_cusip, test_primary_cusip)

    def test_SetVersion(self):
        # they all have the same API
        what = 'machine_learning'
        version = 'a.b.c.d'
        m = SetVersion(
            what=what,
            version=version,
        )
        m2 = from_string(str(m))
        self.assertTrue(isinstance(m2, SetVersion))
        self.assertEqual(m2.what, what)
        self.assertEqual(m2.version, version)

        
##################################################################
if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
