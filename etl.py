'''process messages in queue events.{cusip}'''
# configuration file values used
#   programs/etl.py
#     cusip: str                     : primary cusip to create
#     debug: boolean                 ; if true, invoke pdb if there is a runtime error
#     in_secmaster_path              : str, path to secmaster
#     in_trace_template              : str, template to generate paths to trace print files
#     in_liq_flow_on_the_run_template: str, tempalte to generate paths to liq flow files
#     out_events: optional str       : if supplied, events are written to specified file
#                                      if not supplied, events are written to "queue,{cusip}"
#     output_start: iso-date str     : OutputStart message is sent first thing on this simulated date
#     output_stop: iso-date str      : OutputStop message is sent first thing on this simulated date
# sample invocations
#   python etl.py etl_configuration.json debug.True
import abc
import collections
import copy
import csv
import datetime
import heapq
import pdb
from pprint import pprint as pp
# import pika
import sys

import configuration
import message
import queue


class Event:
    def __init__(
            self,
            datetime: datetime.datetime,
            source: str,
            source_identifier,
            payload: dict):
        assert isinstance(payload, dict)
        self.datetime = datetime
        self.source = source
        self.source_identifier = source_identifier
        self.payload = payload

    def __repr__(self):
        return 'Event(%s, %s, %s, %s)' % (
            self.datetime,
            self.source,
            self.source_identifier,
            self.payload,
            )

    def _as_tuple(self):
        return (self.datetime, self.source, self.source_identifier, len(self.payload))

    def __eq__(self, other):
        # ref: http://portingguide.readthedocs.io/en/latest/comparisons.html
        return self._as_tuple() == other._as_tuple()

    def __ge__(self, other):
        return self._as_tuple() >= other._as_tuple()

    def __gt__(self, other):
        return self._as_tuple() > other._as_tuple()

    def __le__(self, other):
        return self._as_tuple() <= other._as_tuple()

    def __lt__(self, other):
        return self._as_tuple() < other._as_tuple()

    def __ne__(self, other):
        return self._as_tuple() != other._as_tuple()

    def __hash__(self):
        return hash(self._as_tuple())


class EventReader(abc.ABC):
    def __init__(self, config, path):
        self._config = copy.copy(config)
        self._path = path
        
    @abc.abstractmethod
    def __next__(self):
        'return next value or raise StopIteration'
        pass

    @abc.abstractmethod
    def close(self):
        'close the underlying file'
        pass
    

class EventReaderLiqFlowOnTheRun(EventReader):
    def __init__(self, config, path):
        super(EventReaderLiqFlowOnTheRun, self).__init__(
            config=config,
            path=path,
            )
        self._file = open(path, 'r')
        self._reader = csv.DictReader(self._file)
        self._last_datetime = datetime.datetime(datetime.MINYEAR, 1, 1, 0, 0, 0)

    def close(self):
        self._file.close()

    def __next__(self):
        'return an Event or raise StopIteration'
        try:
            row = self._reader.__next__()
            year, month, day = row['date'].split('-')
            hour, minute, second = 0, 0, 0
            event = Event(
                datetime=datetime.datetime(int(year), int(month), int(day), hour, minute, second),
                source='liq_flow_on_the_run',
                source_identifier='%s:%s:%s' % (row['date'], row['primary_cusip'], row['otr_cusip']),
                payload=row,
            )
            assert self._last_datetime <= event.datetime
            self._last_datetime = event.datetime
            return event
        except StopIteration:
            raise StopIteration


class EventReaderTrace(EventReader):
    def __init__(self, config, path):
        super(EventReaderTrace, self).__init__(
            config=config,
            path=path,
            )
        self._file = open(path, 'r')
        self._reader = csv.DictReader(self._file)
        self._last_datetime = datetime.datetime(datetime.MINYEAR, 1, 1, 0, 0, 0)

    def close(self):
        self._file.close()

    def __next__(self):
        'return an Event or raise StopIteration'
        try:
            row = self._reader.__next__()
            year, month, day = row['effectivedate'].split('-')
            hour, minute, second = row['effectivetime'].split(':')
            event = Event(
                datetime=datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second)),
                source='trace',
                source_identifier=row['issuepriceid'],
                payload=row,
            )
            assert self._last_datetime <= event.datetime
            self._last_datetime = event.datetime
            return event
        except StopIteration:
            raise StopIteration


class EventReaderOnce(EventReader):
    'create one event at a pre-specified datetime'
    def __init__(self, config, path, datetime, source, source_identifier, payload):
        super(EventReaderOnce, self).__init__(config=config, path=path)
        self._event = Event(
            datetime=datetime,
            source=source,
            source_identifier=source_identifier,
            payload=payload,
            )
        self._event_returned = False

    def close(self):
        pass

    def __next__(self):
        if self._event_returned:
            raise StopIteration
        else:
            self._event_returned = True
            return self._event


def to_datetime(s):
    year, month, day = s.split('-')
    return datetime.datetime(int(year), int(month), int(day), 0, 0, 0)


class EventReaderOutputStart(EventReaderOnce):
    def __init__(self, config):
        super(EventReaderOutputStart, self).__init__(
            config=config,
            path=None,
            datetime=to_datetime(config.get('output_start')),
            source='etl.py',
            source_identifier='output_start',
            payload={},
            )
            

class EventReaderOutputStop(EventReaderOnce):
    def __init__(self, config):
        super(EventReaderOutputStop, self).__init__(
            config=config,
            path=None,
            datetime=to_datetime(config.get('output_stop')),
            source='etl.py',
            source_identifier='output_stop',
            payload={},
            )

       
class EventReaderSetVersionETL(EventReaderOnce):
    def __init__(self, config):
        super(EventReaderSetVersionETL, self).__init__(
            config=config,
            path=None,
            datetime=datetime.datetime(datetime.MINYEAR, 1, 1, 0, 0, 0),
            source='etl.py',
            source_identifier='set_version',
            payload={
                'what': 'etl',
                'version': '1.0.0.0',
                },
            )
           
       
class EventReaderPrimaryCusip(EventReaderOnce):
    def __init__(self, config):
        super(EventReaderPrimaryCusip, self).__init__(
            config=config,
            path=None,
            datetime=datetime.datetime(datetime.MINYEAR, 1, 1, 0, 0, 0),
            source='etl.py',
            source_identifier='primary_cusip',
            payload={
                'primary_cusip': config.get('primary_cusip'),
                },
            )
            

class EventQueue:
    'maintain datetime-ordered queue of events'
    def __init__(self, event_readers, config):
        self._config = copy.copy(config)
        self._event_readers = copy.copy(event_readers)
        
        self._event_queue = []  # will be a python heapq
        for event_reader in event_readers:
            try:
                event = next(event_reader)
                print('\ninitial event', event)
                heapq.heappush(self._event_queue, (event, event_reader))
            except StopIteration:
                print('event queue is initially empty', event_reader)
                pdb.set_trace()

    def __next__(self):
        'return Event or raise StopIteration'
        try:
            oldest_event, event_reader = heapq.heappop(self._event_queue)
        except IndexError:
            raise StopIteration

        # replace the oldest event with the next event from the same event reader
        try:
            new_event = next(event_reader)
            heapq.heappush(self._event_queue, (new_event, event_reader))
        except StopIteration:
            # there is no new event available from the event reader
            event_reader.close()

        return oldest_event


class SecMaster:
    def __init__(self, path, debug=False):
        file = open(path, 'r')
        reader = csv.DictReader(file)
        self._cusip_to_issuer = {}
        for row in reader:
            cusip = row['CUSIP']
            issuer = row['ticker']
            if debug:
                if 'AJ' in cusip:
                    print(issuer, cusip)
            self._cusip_to_issuer[cusip] = issuer
        file.close()

    def get_issuer(self, cusip: str):
        return self._cusip_to_issuer[cusip]


def str_to_date(s: str):
    year, month, day = s.split('-')
    return datetime.date(int(year), int(month), int(day))


def analysis(config, event_queue):
    trace_on_start_date = collections.Counter()
    otr_cusips = collections.defaultdict(set)
    while True:
        try:
            event = next(event_queue)
        except StopIteration:
            break  # all the event readers are empty

        if event.source == 'trace':
            if event.payload['effectivedate'] == config.get('output_start'):
                trace_on_start_date[event.payload['cusip']] += 1
        elif event.source == 'liq_flow_on_the_run':
            otr_cusips[event.payload['primary_cusip']].add(event.payload['otr_cusip'])
        else:
            pass
    print('\ncounts and OTR cusips')
    print('for cusips of trace events on the output start date %s' % config.get('output_start'))
    for k in sorted(trace_on_start_date.keys()):
        print('%s: %3d %s' % (k, trace_on_start_date[k], otr_cusips[k]))
        

def make_event_queue(config, issuer):
    event_queue = EventQueue(
        event_readers=(
            EventReaderLiqFlowOnTheRun(
                config=config,
                path=config.get('in_liq_flow_on_the_run_template') % issuer,
            ),
            EventReaderTrace(
                config=config,
                path=config.get('in_trace_template') % issuer,
            ),
            EventReaderOutputStart(config),
            EventReaderOutputStop(config),
            EventReaderPrimaryCusip(config),
            EventReaderSetVersionETL(config),
            ),
        config=config,
        )
    return event_queue


def do_work(config, verbose=True):
    def vp(*args):
        if verbose:
            print(*args)

    def vpp(*args):
        if verbose:
            pp(*args)
            
    secmaster = SecMaster(
        path=config.get('in_secmaster_path'),
        debug=False,
    )
    issuer = secmaster.get_issuer(config.get('primary_cusip'))

    out_events = queue.WriterFile(config.get('out_events'))  # for now, just write a file
    primary_cusip = config.get('primary_cusip')
    routing_key = 'events.%s' % primary_cusip

    event_queue = make_event_queue(config, issuer)
    otr_cusip = {}  # key: cusip, value: int (>= 1)
    while True:
        try:
            event = next(event_queue)
        except StopIteration:
            break  # all the event readers are empty

        # handle the event
        vp('\nnext event:', event.datetime, event.source, event.source_identifier)
        if event.source == 'trace':
            print('handle trace event')
            if event.payload['cusip'] == primary_cusip or event.payload['cusip'] in otr_cusip:
                vp('trace print for primary or OTR cusip')
                out_events.write(
                    routing_key=routing_key,
                    message=message.TracePrint(
                        source='trace_%s.csv' % issuer,
                        identifier=event.source_identifier,
                        cusip=event.payload['cusip'],
                        issuepriceid=event.source_identifier,
                        datetime=event.datetime,
                        oasspread=float(event.payload['oas']),
                        trade_type=event.payload['trade_type'],
                        reclassified_trade_type=event.payload['reclassified_trade_type'],
                        cancellation_probability=0.0,  # for now
                    ),
                    )
            else:
                vp('trace print for neither primary nor OTR cusip')
        elif event.source == 'liq_flow_on_the_run':
            if event.payload['primary_cusip'] == primary_cusip:
                vp('handle liq_flow event for the primary cusip')
                out_events.write(
                    routing_key=routing_key,
                    message=message.SetCusipOtr(
                        source='liq_flow_on_the_run_%s.csv' % issuer,
                        identifier=event.source_identifier,
                        primary_cusip=event.payload['primary_cusip'],
                        otr_level=1,  # for now, just 1 OTR cusip
                        otr_cusip=event.payload['otr_cusip'],
                    ),
                    )
                otr_cusip[event.payload['otr_cusip']] = 1  # for now, just 1 OTR cusip
            else:
                vp('otr not for primary')
        elif event.source == 'etl.py':
            vp('handle etl.py event')
            if event.source_identifier == 'output_start':
                out_events.write(
                    routing_key=routing_key,
                    message=message.OutputStart(
                        source='elt.py',
                        identifier=str(datetime.datetime.now()),
                    ),
                )
            elif event.source_identifier == 'output_stop':
                out_events.write(
                    routing_key=routing_key,
                    message=message.OutputStop(
                        source='elt.py',
                        identifier=str(datetime.datetime.now()),
                    ),
                )
            elif event.source_identifier == 'primary_cusip':
                out_events.write(
                    routing_key=routing_key,
                    message=message.SetCusipPrimary(
                        source='etl.py',
                        identifier=str(datetime.datetime.now()),
                        primary_cusip=event.payload['primary_cusip'],
                        ),
                    )
            elif event.source_identifier == 'set_version':
                out_events.write(
                    routing_key=routing_key,
                    message=message.SetVersion(
                        source='etl.py',
                        identifier=str(datetime.datetime.now()),
                        what=event.payload['what'],
                        version=event.payload['version'],
                        ),
                    )
            else:
                print('invalid event.source_identifier %s' % event.source_identifier)
                pdb.set_trace()
        else:
            print(event)
            print('unknown event source')
            pdb.set_trace()
    print('processed all of the events')
    out_events.close()
    
    analysis(config, make_event_queue(config, issuer))
    
    
def main(argv):
    program = 'etl.py'
    config = configuration.make(
        program=program,
        argv=argv[1:],  # ignore program name
    )
    print('started %s with configuration' % program)
    print(str(config))
    if config.get('debug', False):
        # enter pdb if run-time error
        # (useful during development)
        import debug
        if False:
            debug.info
    do_work(config)


if __name__ == '__main__':
    if False:
        pdb
        pp
    main(sys.argv)
