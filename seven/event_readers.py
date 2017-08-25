'event readers read a file and create events from it'
import abc
import csv
import datetime
import pdb
import sys

# imports from directory seven
from . import Event
from . import make_event_attributes
from . import logging


class EventReader(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def close(self):
        'close any underlying file'

    @abc.abstractmethod
    def next(self):
        'return Event or raise StopIteration()'


class EventReaderDate(EventReader, metaclass=abc.ABCMeta):
    'the events have a date, but not a time'

    def __init__(self,
                 control,
                 date_column_name,
                 event_feature_maker_class,
                 event_source,
                 path,
                 source_identifier_function,
                 test=False):
        self._control = control
        self._date_column_name = date_column_name
        self._event_feature_maker_class = event_feature_maker_class
        self._event_source = event_source
        self._path = path
        self._source_identifier_function = source_identifier_function

        # prepare to test that the file is in date order
        self._prior_event_date = datetime.date(datetime.MINYEAR, 1, 1)

        # prepare to read the CSV file
        self._file = open(path)
        self._dict_reader = csv.DictReader(self._file)

        self._records_read = 0

    def close(self):
        self._file.close()

    def __next__(self):
        'return (Event, err) or raise StopIteration'
        try:
            row = next(self._dict_reader)
        except StopIteration:
            raise StopIteration()
        self._records_read += 1

        # create the Event
        year, month, day = row[self._date_column_name].split('-')
        event = Event.Event(
            year=year,
            month=month,
            day=day,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            source=self._event_source,
            source_identifier=self._source_identifier_function(row),
            payload=row,
            make_event_attributes_class=self._event_feature_maker_class,
        )

        # make sure file is sorted by increasing date
        if not event.date() >= self._prior_event_date:
            logging.critical(
                'input file not sorted in increasing date order at record %d: %s' % (
                    self._records_read,
                    self._path,
                ))
            sys.exit(1)
        self._prior_event_date = event.date()

        return event, None

    def records_read(self):
        return self._records_read


class AmtOutstandingHistory(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'amt_outstanding_history'
        super(AmtOutstandingHistory, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.AmtOutstandingHistory,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: '%s-%s' % (row['date'], row['CUSIP']),
        )


class CurrentCoupon(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'current_coupon'
        super(CurrentCoupon, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.CurrentCoupon,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: '%s-%s' % (row['date'], row['CUSIP']),
        )


class EtfWeightOfCusipPctAgg(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'etf_weight_of_cusip_pct_agg'
        super(EtfWeightOfCusipPctAgg, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.CurrentCoupon,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: '%s-%s' % (row['date'], row['cusip']),
        )


class EtfWeightOfCusipPctLqd(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'etf_weight_of_cusip_pct_lqd'
        super(EtfWeightOfCusipPctLqd, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.EtfWeightOfCusipPctLqd,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: '%s-%s' % (row['date'], row['cusip']),
        )


class EtfWeightOfIssuerPctAgg(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'etf_weight_of_issuer_pct_agg'
        super(EtfWeightOfIssuerPctAgg, self).__init__(
            control=control,
            date_column_name='date',
            event_source=event_source,
            event_feature_maker_class=make_event_attributes.EtfWeightOfIssuerPctAgg,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: '%s-%s' % (row['date'], row['ticker']),
        )


class EtfWeightOfIssuerPctLqd(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'etf_weight_of_issuer_pct_lqd'
        super(EtfWeightOfIssuerPctLqd, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.EtfWeightOfIssuerPctLqd,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: '%s-%s' % (row['date'], row['ticker']),
        )


class EtfWeightOfSectorPctAgg(EventReaderDate):
    # TODO: finding mapping of issuers to one or more sectors
    def __init__(self, control, test=False):
        event_source = 'etf_weight_of_sector_pct_agg'
        super(EtfWeightOfSectorPctAgg, self).__init__(
            control=control,
            event_feature_maker_class=make_event_attributes.EtfWeightOfSectorPctAgg,
            event_source=event_source,
            date_column_name='date',
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: '%s-%s' % (row['date'], row['sector']),
        )


class EtfWeightOfSectorPctLqd(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'etf_weight_of_sector_pct_lqd'
        super(EtfWeightOfSectorPctLqd, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.EtfWeightOfSectorPctLqd,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: '%s-%s' % (row['date'], row['sector']),
        )


class FunExpectedInterestCoverage(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'fun_expected_interest_coverage'
        super(FunExpectedInterestCoverage, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.FunExpectedInterestCoverage,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: row['date'],
        )


class FunGrossLeverage(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'fun_gross_leverage'
        super(FunGrossLeverage, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.FunGrossLeverage,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: row['date'],
        )


class FunLtmEbitda(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'ltm_ebitda'
        super(FunLtmEbitda, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.FunLtmEbitda,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: row['date'],
        )


class FunMktCap(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'fun_mkt_cap'
        super(FunMktCap, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.FunMktCap,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: row['date'],
        )


class FunMktGrossLeverage(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'fun_mkt_gross_leverage'
        super(FunMktGrossLeverage, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.FunMktGrossLeverage,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: row['date'],
        )


class FunReportedInterestCoverage(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'fun_reported_interest_coverage'
        super(FunReportedInterestCoverage, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.FunReportedInterestCoverage,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: row['date'],
        )


class FunTotalAssets(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'fun_total_assets'
        super(FunTotalAssets, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.FunTotalAssets,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: row['date'],
        )


class FunTotalDebt(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'fun_total_debt'
        super(FunTotalDebt, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.FunTotalDebt,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: row['date'],
        )


class HistEquityPrices(EventReader):
    # TODO: write me out; I'm a matrix
    # TODO: make sure file is in date increasing order
    def __init__(self, control, test=False):
        pass

    def close(self):
        pass

    def __next__(self):
        # TODO: parse the matrix input
        logging.warning('HistEquityPricesEventReader.next: STUB')
        raise StopIteration()  # pretend that the file is empty


class LiqFlowOnTheRun(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'liq_flow_on_the_run'
        super(LiqFlowOnTheRun, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=make_event_attributes.LiqFlowOnTheRun,
            event_source=event_source,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: '%s-%s' % (row['date'], row['primary_cusip']),
        )


class SecMaster(EventReader):
    # TODO: write me out; the date should be the control.arg.event_start_date
    # TODO: make sure file is in date increasing order
    def __init__(self, control, test=False):
        pass

    def close(self):
        pass

    def __next__(self):
        logging.warning('SecurityMasterEventReader.next: STUB')
        raise StopIteration()  # pretend that the file is empty


class Trace(EventReader):
    'read trace events for a specific cusip'
    def __init__(self, control, test=False):
        self._control = control
        self._event_source = 'trace'

        path = control.path['in_' + self._event_source]
        self._file = open(path)
        self._dict_reader = csv.DictReader(self._file)
        self._prior_record_datetime = datetime.datetime(datetime.MINYEAR, 1, 1)
        self._records_read = 0

    def __iter__(self):
        return self

    def close(self):
        self._file.close()

    def __next__(self):
        'return (Event, err) or raise StopIteration'
        try:
            row = next(self._dict_reader)
        except StopIteration:
            raise StopIteration()
        self._records_read += 1
        if row['reclassified_trade_type'] not in ('B', 'S'):
            err = 'invalid reclassified trade type value "%s"' % (
                row['reclassified_trade_type'],
            )
            logging.warning(err)
        else:
            err = None

        # create the Event
        year, month, day = row['effectivedate'].split('-')
        hour, minute, second = row['effectivetime'].split(':')
        event = Event.Event(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            microsecond=0,
            source=self._event_source,
            source_identifier=row['issuepriceid'],
            payload=row,
            make_event_attributes_class=make_event_attributes.Trace,
        )

        # make sure inpupt file is sorted by increasing datetime
        assert event.datetime() >= self._prior_record_datetime
        self._prior_record_datetime = event.datetime()

        return event, err

    def records_read(self):
        return self._records_read


if __name__ == '__main__':
    pdb
