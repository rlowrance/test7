'event readers read a file and create events from it'
import abc
import csv
import datetime
import sys

# imports from directory seven
import feature_makers2
import input_event
import logging


class EventReader(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def close(self):
        'close any underlying file'

    @abc.abstractmethod
    def next(self):
        'return Event or raise StopIteration()'


class EventReaderDate(EventReader):
    'the events have a date, but not a time'
    __metaclass__ = abc.ABCMeta

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

    def next(self):
        'return Event or raise StopIteration'
        try:
            row = self._dict_reader.next()
        except StopIteration:
            raise StopIteration()
        self._records_read += 1

        # create the Event
        year, month, day = row[self._date_column_name].split('-')
        event = input_event.Event(
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
            event_feature_maker_class=self._event_feature_maker_class,
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

        return event

    def records_read(self):
        return self._records_read


class AmtOutstandingHistory(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'amt_outstanding_history'
        super(AmtOutstandingHistory, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=feature_makers2.AmtOutstandingHistory,
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
            event_feature_maker_class=feature_makers2.CurrentCoupon,
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
            event_feature_maker_class=feature_makers2.CurrentCoupon,
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
            event_feature_maker_class=feature_makers2.EtfWeightOfCusipPctLqd,
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
            event_feature_maker_class=feature_makers2.EtfWeightOfIssuerPctAgg,
            path=control.path['in_' + event_source],
            source_identifier_function=lambda row: '%s-%s' % (row['date'], row['ticker']),
        )


class EtfWeightOfIssuerPctLqd(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'etf_weight_of_issuer_pct_lqd'
        super(EtfWeightOfIssuerPctLqd, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=feature_makers2.EtfWeightOfIssuerPctLqd,
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
            event_feature_maker_class=feature_makers2.EtfWeightOfSectorPctAgg,
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
            event_feature_maker_class=feature_makers2.EtfWeightOfSectorPctLqd,
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
            event_feature_maker_class=feature_makers2.FunExpectedInterestCoverage,
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
            event_feature_maker_class=feature_makers2.FunGrossLeverage,
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
            event_feature_maker_class=feature_makers2.FunLtmEbitda,
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
            event_feature_maker_class=feature_makers2.FunMktCap,
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
            event_feature_maker_class=feature_makers2.FunMktGrossLeverage,
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
            event_feature_maker_class=feature_makers2.FunReportedInterestCoverage,
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
            event_feature_maker_class=feature_makers2.FunTotalAssets,
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
            event_feature_maker_class=feature_makers2.FunTotalDebt,
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

    def next(self):
        # TODO: parse the matrix input
        logging.warning('HistEquityPricesEventReader.next: STUB')
        raise StopIteration()  # pretend that the file is empty


class LiqFlowOnTheRun(EventReaderDate):
    def __init__(self, control, test=False):
        event_source = 'liq_flow_on_the_run'
        super(LiqFlowOnTheRun, self).__init__(
            control=control,
            date_column_name='date',
            event_feature_maker_class=feature_makers2.LiqFlowOnTheRun,
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

    def next(self):
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

    def next(self):
        try:
            row = self._dict_reader.next()
        except StopIteration:
            raise StopIteration()
        self._records_read += 1
        if row['trade_type'] == 'D' and row['reclassified_trade_type'] == 'D':
            logging.warning('D trade not reclassified to B or S', row['issuepriceid'])

        # create the Event
        year, month, day = row['effectivedate'].split('-')
        hour, minute, second = row['effectivetime'].split(':')
        event = input_event.Event(
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
            event_feature_maker_class=feature_makers2.Trace,
        )

        # make sure inpupt file is sorted by increasing datetime
        assert event.datetime() >= self._prior_record_datetime
        self._prior_record_datetime = event.datetime()

        return event

    def records_read(self):
        return self._records_read
