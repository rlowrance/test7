'''messages bodies in queues

The bodies of messages in queues are JSON-formatted strings. The implied
JSON object is a dictionary with these key/value pairs:

- key="message_id", valeu=a str, the name of the class (see below)
- key="payload", value=a JSON-formatted string

Messags on RabbitMQ queues have additional fields called
headers. Those fields are invisible to this code.
'''
import abc
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
    print('not seriable', obj, type(obj))
    
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
class Message(abc.ABC):
    def __init__(self, message_type: str, source: str, identifier: str):
        self.message_type = message_type
        self.source = source
        self.identifier = identifier

    @abc.abstractmethod
    def __repr__(self, message_name='Message', other_fields=''):
        # the message type field is not used by the constructor
        return "%s(source='%s', identifier='%s'%s)" % (
            message_name,
            self.source,
            self.identifier,
            '' if other_fields == '' else ', %s' % other_fields,
            )

    def __str__(self):
        'return json-compliant string'
        # the message type field is part of the str representation
        return json.dumps(self.as_dict())

    @abc.abstractmethod
    def as_dict(self):
        return {
            'message_type': self.message_type,
            'source': self.source,
            'identifier': self.identifier,
        }

    @staticmethod    # NOTE: staticmethod must come before abstractmethod
    @abc.abstractmethod
    def from_dict(d: dict):
        return Message(
            message_type=d['message_type'],
            source=d['source'],
            identifier=d['identifier'],
            )

#####################################################
# subsclasses of Message
#####################################################


class BackToZero(Message):
    def __init__(self, source:str, identifier: str):
        self._super = super(BackToZero, self)
        self._super.__init__("BackToZero", source, identifier)

    def __repr__(self):
        return self._super.__repr__(
            message_name='BackToZero',
            other_fields='',
            )

    def __str__(self):
        'return a json-compliant string'
        return json.dumps(self.as_dict())
    
    def as_dict(self):
        result = self._super.as_dict()
        return result

    @staticmethod
    def from_dict(d: dict):
        return BackToZero(
            source=d['source'],
            identifier=d['identifier'],
        )

    
class SetCusipOtr(Message):
    def __init__(self, source: str, identifier: str, primary_cusip: str, otr_level: int, otr_cusip: str):
        self._super = super(SetCusipOtr, self)
        self._super.__init__('SetCusipOtr', source, identifier)
        assert isinstance(otr_level, int)
        assert otr_level >= 1
        self.primary_cusip = primary_cusip
        self.otr_level = otr_level
        self.otr_cusip = otr_cusip

    def __repr__(self):
        return self._super.__repr__(
            message_name='SetCusipOtr',
            other_fields="primary_cusip='%s', otr_level=%s, otr_cusip='%s'" % (
                self.primary_cusip,
                self.otr_level,
                self.otr_cusip,
                ),
            )

    def __str__(self):
        return json.dumps(self.as_dict())

    def as_dict(self):
        result = self._super.as_dict()
        result.update({
            'primary_cusip': self.primary_cusip,
            'otr_level': self.otr_level,
            'otr_cusip': self.otr_cusip,
            })
        return result
        
    @staticmethod
    def from_dict(d: dict):
        'factory method'
        return SetCusipOtr(
            d['source'],
            d['identifier'],
            d['primary_cusip'],
            d['otr_level'],
            d['otr_cusip'],
            )


class SetCusipPrimary(Message):
    def __init__(self, source: str, identifier: str, primary_cusip: str):
        self._super = super(SetCusipPrimary, self)
        self._super.__init__('SetCusipPrimary', source, identifier)
        self.primary_cusip = primary_cusip

    def __repr__(self):
        return self.__super__.__repr__(
            message_name='SetCusipPrimary',
            other_fields="primary_cusip='%s" % (
                self.primary_cusip,
                ),
            )

    def __str__(self):
        return json.dumps(self.as_dict())

    def as_dict(self):
        result = self._super.as_dict()
        result.update({
            'primary_cusip': self.primary_cusip,
            })
        return result

    @staticmethod
    def from_dict(d: dict):
        'factor method'
        return SetCusipPrimary(
            d['source'],
            d['identifier'],
            d['primary_cusip'],
            )

    
class SetVersion(Message):
    def __init__(self, source: str, identifier: str, what: str, version: str):
        self._super = super(SetVersion, self)
        self._super.__init__('SetVersion', source, identifier)
        self.what = what
        self.version = version

    def __repr__(self):
        return self._super_.__repr__(
            message_name='setVersion',
            other_fields="what='%s', version='%s'" % (
                self.what,
                self.version,
                ))

    def __str__(self):
        return json.dumps(self.as_dict())

    def as_dict(self):
        result = self._super.as_dict()
        result.update({
            'what': self.what,
            'version': self.version,
            })
        return result

    @staticmethod
    def from_dict(d: dict):
        return SetVersion(
            d['source'],
            d['identifier'],
            d['what'],
            d['version'],
            )

        
class TracePrint(Message):
    def __init__(
            self,
            source: str,
            identifier: str,
            cusip: str,
            issuepriceid: str,
            datetime: datetime.datetime,
            oasspread: float,
            trade_type: str,
            reclassified_trade_type: str,
            cancellation_probability: float):
        self._super = super(TracePrint, self)
        self._super.__init__('TracePrint', source, identifier)
        self.cusip = cusip
        self.issuepriceid = issuepriceid
        self.datetime = datetime
        self.oasspread = oasspread
        self.trade_type = trade_type
        self.reclassified_trade_type = reclassified_trade_type
        self.cancellation_probability = cancellation_probability

    def __repr__(self):
        return self._super.__repr__(
            message_name='TracePrint',
            other_fields=(
                "cusip='%s'" % self.cusip +
                ", issuepriceid='%s'" % self.issuerpriceid +
                ", datetime=%s" % self.datetime +
                ", oasspread=%s" % self.oasspread +
                ", trade_type='%s'" % self.trade_type +
                ", reclassified_trade_type='%s'" % self.reclassified_trade_type +
                ", cancellation_probability=%s" % self.cancellation_probability)
            )

    def __str__(self):
        x = self.as_dict()
        x['datetime'] = x['datetime'].isoformat()
        return json.dumps(x)

    def as_dict(self):
        result = self._super.as_dict()
        result.update({
            'cusip': self.cusip,
            'issuepriceid': self.issuepriceid,
            'datetime': self.datetime,
            'oasspread': self.oasspread,
            'trade_type': self.trade_type,
            'reclassified_trade_type': self.reclassified_trade_type,
            'cancellation_probability': self.cancellation_probability,
            })
        return result
    
    @staticmethod
    def from_dict(d: dict):
        return TracePrint(
            d['source'],
            d['identifier'],
            d['cusip'],
            d['issuepriceid'],
            str_to_datetime(d['datetime']),
            d['oasspread'],
            d['trade_type'],
            d['reclassified_trade_type'],
            d['cancellation_probability'],
            )

    
class TracePrintCancel(Message):
    def __init__(self, source, identifier, issuepriceid: str):
        self._super = super(TracePrintCancel, self)
        self._super.__init__('TracePrintCancel', source, identifier)
        self.issuepriceid = issuepriceid

    def __repr__(self):
        return self._super.__repr__(
            message_name='TracePrintCancel',
            other_fields="issuerpriceid='%s'" % self.issuepriceid,
        )

    def __str__(self):
        return json.dumps(self.as_dict())

    def as_dict(self):
        result = self._super.as_dict()
        result.update({
            'issuepriceid': self.issuepriceid,
            })
        return result

    @staticmethod
    def from_dict(d: dict):
        return TracePrintCancel(
            d['source'],
            d['identifier'],
            d['issuepriceid'],
            )


class OutputStart(Message):
    def __init__(self, source, identifier):
        self._super = super(OutputStart, self)
        self._super.__init__('OutputStart', source, identifier)

    def __repr__(self):
        return self._super.__repr__(
            message_name='OutputStart',
            otherfields='',
            )

    def __str__(self):
        return json.dumps(self.as_dict())

    def as_dict(self):
        result = self._super.as_dict()
        return result

    @staticmethod
    def from_dict(d: dict):
        return OutputStart(
            d['source'],
            d['identifier'],
        )

    
class OutputStop(Message):
    def __init__(self, source, identifier):
        self._super = super(OutputStop, self)
        self._super.__init__('OutputStop', source, identifier)

    def __repr__(self):
        return self._super.__repr__(
            message_name='OutputStop',
            other_fields='',
            )

    def __str__(self):
        return json.dumps(self.as_dict())

    def as_dict(self):
        result = self._super.as_dict()
        return result

    @staticmethod
    def from_dict(d: dict):
        return OutputStop(
            d['source'],
            d['identifier'],
            )

    
###################################################################
def from_string(s: str):
    'return an appropriate subclass of Message'
    # s is a json-encoded string
    obj = json.loads(s)
    assert isinstance(obj, dict)
    message_type = obj['message_type']
    if message_type == 'BackToZero':
        return BackToZero.from_dict(obj)
    if message_type == 'OutputStart':
        return OutputStart.from_dict(obj)
    if message_type == 'SetCusipOtr':
        return SetCusipOtr.from_dict(obj)
    if message_type == 'SetCusipPrimary':
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
        source = 'unittest'
        identifier = 123
        m = BackToZero(source, identifier)
        m2 = from_string(str(m))
        self.assertTrue(isinstance(m2, BackToZero))
        self.assertEqual(m2.source, source)
        self.assertEqual(m2.identifier, identifier)
                        
    def test_SetCusipOtr(self):
        source = 'unittest'
        identifier = 123
        test_primary_cusip = "primary"
        test_otr_level = 2
        test_otr_cusip = "otr"
        m = SetCusipOtr(
            source=source,
            identifier=identifier,
            primary_cusip=test_primary_cusip,
            otr_cusip=test_otr_cusip,
            otr_level=test_otr_level,
            )
        s = str(m)
        m2 = from_string(s)
        assert isinstance(m2, SetCusipOtr)
        self.assertEqual(m2.source, source)
        self.assertEqual(m2.identifier, identifier)
        self.assertEqual(m.primary_cusip, m2.primary_cusip)
        self.assertEqual(m.otr_cusip, m2.otr_cusip)
        self.assertEqual(m.otr_level, m2.otr_level)

    def test_SetPrimaryCusip(self):
        source = 'unittest'
        identifier = 123
        test_primary_cusip = 'primary'
        m = SetCusipPrimary(
            source=source,
            identifier=identifier,
            primary_cusip=test_primary_cusip,
            )
        m2 = from_string(str(m))
        self.assertEqual(m2.source, source)
        self.assertEqual(m2.identifier, identifier)
        self.assertTrue(isinstance(m2, SetCusipPrimary))
        self.assertEqual(m2.primary_cusip, test_primary_cusip)

    def test_SetVersion(self):
        source = 'unittest'
        identifier = 123
        what = 'machine_learning'
        version = 'a.b.c.d'
        m = SetVersion(
            source=source,
            identifier=identifier,
            what=what,
            version=version,
        )
        m2 = from_string(str(m))
        self.assertEqual(m2.source, source)
        self.assertEqual(m2.identifier, identifier)
        self.assertTrue(isinstance(m2, SetVersion))
        self.assertEqual(m2.what, what)
        self.assertEqual(m2.version, version)

    def test_TracePrint(self):
        source = 'unittest'
        identifier = 123
        cusip = 'cusip'
        issuepriceid = 'issuepriceid'
        dt = datetime.datetime(datetime.MAXYEAR, 1, 1)
        oasspread = 1.23
        trade_type = 'D'
        reclassified_trade_type = 'B'
        cancellation_probability = 0.5
        m = TracePrint(
            source=source,
            identifier=identifier,
            cusip=cusip,
            issuepriceid=issuepriceid,
            datetime=dt,
            oasspread=oasspread,
            trade_type=trade_type,
            reclassified_trade_type=reclassified_trade_type,
            cancellation_probability=cancellation_probability,
            )
        m2 = from_string(str(m))
        self.assertEqual(m2.source, source)
        self.assertEqual(m2.identifier, identifier)
        self.assertTrue(isinstance(m2, TracePrint))
        self.assertEqual(m2.cusip, cusip)
        self.assertEqual(m2.issuepriceid, issuepriceid)
        self.assertEqual(m2.datetime, dt)
        self.assertEqual(m2.oasspread, oasspread)
        self.assertEqual(m2.trade_type, trade_type)
        self.assertEqual(m2.reclassified_trade_type, reclassified_trade_type)
        self.assertEqual(m2.cancellation_probability, cancellation_probability)

    def test_TracePrintCancel(self):
        source = 'unittest'
        identifier = 123
        issuepriceid = 'issuepriceid'
        m = TracePrintCancel(
            source=source,
            identifier=identifier,
            issuepriceid=issuepriceid,
            )
        m2 = from_string(str(m))
        self.assertEqual(m2.source, source)
        self.assertEqual(m2.identifier, identifier)
        self.assertTrue(isinstance(m2, TracePrintCancel))
        self.assertEqual(m2.issuepriceid, issuepriceid)

    def test_OutputStart(self):
        source = 'unittest'
        identifier = 123
        m = OutputStart(
            source=source,
            identifier=identifier,
        )
        s = str(m)
        m2 = from_string(s)
        self.assertEqual(m2.source, source)
        self.assertEqual(m2.identifier, identifier)
        self.assertTrue(isinstance(m2, OutputStart))

    def test_OutputStop(self):
        source = 'unittest'
        identifier = 123
        m = OutputStop(
             source=source,
            identifier=identifier,
        )
        m2 = from_string(str(m))
        self.assertEqual(m2.source, source)
        self.assertEqual(m2.identifier, identifier)
        self.assertTrue(isinstance(m2, OutputStop))

        
##################################################################
if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
