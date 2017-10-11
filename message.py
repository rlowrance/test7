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


#####################################################
# utility functions
#####################################################
def json_serial(obj):
    'JSON serializer for objects not serialized by default json code'
    # ref: search stack overflow "how to overcome datetime not json serializable"
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    raise TypeError('Type %s is not JSON serializable' % type(obj))


def str_to_datetime(s: str):
    'return datetime.datetime'
    date, time = s.split('T')
    year, month, day = date.split('-')
    hour, minute, second = time.split(':')
    if '.' in second:
        seconds, microseconds = second.split('.')
        return datetime.datetime(
            int(year), int(month), int(day),
            int(hour), int(minute), int(seconds), int(microseconds),
        )
    else:
        return datetime.datetime(
            int(year), int(month), int(day),
            int(hour), int(minute), int(second),
        )


#####################################################
# base class
#####################################################
class Message:
    def __init__(self, message_type: str):
        self.message_type = message_type

#####################################################
# subsclasses of Message
#####################################################


class BackToZero(Message):
    def __init__(self):
        super(BackToZero, self).__init__("BackToZero")

    @staticmethod
    def from_dict(d: dict):
        return BackToZero()

    def __str__(self):
        return json.dumps({
            'message_type': self.message_type,
            })

    
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
        super(SetCusipPrimary, self).__init__("SetCusipPrimary")
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
            trade_type: str,
            reclassified_trade_type: str,
            cancellation_probability: float):
        super(TracePrint, self).__init__("TracePrint")
        self.cusip = cusip
        self.issuepriceid = issuepriceid
        self.datetime = datetime
        self.oasspread = oasspread
        self.trade_type = trade_type
        self.reclassified_trade_type = reclassified_trade_type
        self.cancellation_probability = cancellation_probability

    @staticmethod
    def from_dict(d: dict):
        return TracePrint(
            d['cusip'],
            d['issuepriceid'],
            str_to_datetime(d['datetime']),
            d['oasspread'],
            d['cancellation_probability'],
            )

    def __str__(self):
        return json.dumps({
            'message_type': self.message_type,
            'cusip': self.cusip,
            'issuepriceid': self.issuepriceid,
            'datetime': self.datetime.isoformat(),
            'oasspread': self.oasspread,
            'cancellation_probability': self.cancellation_probability,
            })

    
class TracePrintCancel(Message):
    def __init__(self, issuepriceid: str):
        super(TracePrintCancel, self).__init__("TracePrintCancel")
        self.issuepriceid = issuepriceid

    def from_dict(d: dict):
        return TracePrintCancel(
            d['issuepriceid'],
            )

    def __str__(self):
        return json.dumps({
            'message_type': self.message_type,
            'issuepriceid': self.issuepriceid,
            })
    

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
        return json.dumps({
            'message_type': self.message_type,
            })

    
###################################################################
def from_string(s: str):
    'return an appropriate subclass of Message'
    obj = json.loads(s)
    assert isinstance(obj, dict)
    message_type = obj['message_type']
    if message_type == 'BackToZero':
        return BackToZero.from_dict(obj)
    if message_type == 'OutputStart':
        return OutputStart.from_dict(obj)
    if message_type == 'SetCusipOtr':
        return SetCusipOtr.from_dict(obj)
    if message_type == 'SetPrimaryCusip':
        return SetCusipPrimary.from_dict(obj)
    if message_type == 'SetVersion':
        return SetVersion.from_dict(obj)
    if message_type == 'TracePrint':
        return TracePrint.from_dict(obj)
    if message_type == 'TracePrintCancel':
        return TracePrintCancel.from_dict(obj)
    if message_type == 'OutputStart':
        return OutputStart.from_dict(obj)
    if message_type == 'OutputStop':
        return OutputStop.from_dict(obj)
    assert False, 'message_type %s is not known' % message_type


####################################################################
class Test(unittest.TestCase):
    def test_str_to_datetime(self):
        tests = (
            datetime.datetime(1, 2, 3, 4, 5, 6),
            datetime.datetime(1, 2, 3, 4, 5, 6, 7),
            )
        for test in tests:
            s = test.isoformat()
            d = str_to_datetime(s)
            self.assertEqual(d, test)

    def test_BackToZero(self):
        m = BackToZero()
        m2 = from_string(str(m))
        self.assertTrue(isinstance(m2, BackToZero))
                        
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

    def test_TracePrint(self):
        cusip = 'cusip'
        issuepriceid = 'issuepriceid'
        dt = datetime.datetime(datetime.MAXYEAR, 1, 1)
        oasspread = 1.23
        cancellation_probability = 0.5
        m = TracePrint(
            cusip=cusip,
            issuepriceid=issuepriceid,
            datetime=dt,
            oasspread=oasspread,
            cancellation_probability=cancellation_probability,
            )
        m2 = from_string(str(m))
        self.assertTrue(isinstance(m2, TracePrint))
        self.assertEqual(m2.cusip, cusip)
        self.assertEqual(m2.issuepriceid, issuepriceid)
        self.assertEqual(m2.datetime, dt)
        self.assertEqual(m2.oasspread, oasspread)
        self.assertEqual(m2.cancellation_probability, cancellation_probability)

    def test_TracePrintCancel(self):
        issuepriceid = 'issuepriceid'
        m = TracePrintCancel(
            issuepriceid=issuepriceid,
            )
        m2 = from_string(str(m))
        self.assertTrue(isinstance(m2, TracePrintCancel))
        self.assertEqual(m2.issuepriceid, issuepriceid)


    def test_OutputStart(self):
        m = OutputStart()
        s = str(m)
        m2 = from_string(s)
        self.assertTrue(isinstance(m2, OutputStart))

    def test_OutputStop(self):
        m = OutputStop()
        m2 = from_string(str(m))
        self.assertTrue(isinstance(m2, OutputStop))

        
##################################################################
if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
