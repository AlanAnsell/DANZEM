import numpy as np
import pandas as pd

#import bisect
#import math
import time
import datetime
import pytz
from calendar import monthrange

NZST = pytz.timezone('Pacific/Auckland')
UTC = pytz.timezone('UTC')
DT_FORMAT = '%d/%m/%Y %H:%M:%S'

TPID = 'tpid'

DATA_BASE_DIR_ = 'DANZEM/data'
BIDS_DIR_ = '%s/Bids' % DATA_BASE_DIR_
FINAL_PRICE_DIR_ = '%s/FinalPrices' % DATA_BASE_DIR_
LGP_DIR_ = '%s/LoadGenerationPrice' % DATA_BASE_DIR_
OFFERS_DIR_ = '%s/Offers' % DATA_BASE_DIR_
CLEARED_OFFERS_DIR_ = '%s/ClearedOffers' % DATA_BASE_DIR_
GENERATION_DIR_ = '%s/Generation' % DATA_BASE_DIR_


def SetDir(dir_):
    DATA_BASE_DIR_ = dir_


def Today():
    ts = time.localtime()
    return (ts.tm_year, ts.tm_mon, ts.tm_mday) 


def MonthRange(first, last):
    years = list(range(first[0], last[0] + 1))
    first_month = {year: 1 for year in years}
    first_month[first[0]] = first[1]
    last_month = {year: 12 for year in years}
    last_month[last[0]] = last[1]
    return [(year, month)
            for year in years
            for month in range(first_month[year], last_month[year] + 1)]


def DayRange(first, last):
    yms = MonthRange(first[:2], last[:2])
    first_day = {ym: 1 for ym in yms}
    first_day[yms[0]] = first[2]
    last_day = {ym: monthrange(ym[0], ym[1])[1] for ym in yms}
    last_day[yms[-1]] = last[2]

    return [(ym[0], ym[1], day)
            for ym in yms
            for day in range(first_day[ym], last_day[ym] + 1)]


def FileNameFromDateTuple(date):
    if len(date) == 2:
        return '%d%02d' % (date[0], date[1])
    else:
        return '%d%02d%02d' % (date[0], date[1], date[2])


def DateStrFromDateTuple(date):
    if len(date) == 2:
        return '%d/%02d' % (date[0], date[1])
    else:
        return '%d/%02d/%02d' % (date[0], date[1], date[2])


def GetBidFile(ymd):
    file_path = '%s/%s.csv' % (BIDS_DIR_, FileNameFromDateTuple(ymd))
    return pd.read_csv(file_path)


def GetFinalPriceFile(ym):
    file_path = '%s/%s.csv' % (FINAL_PRICE_DIR_, FileNameFromDateTuple(ym))
    return pd.read_csv(file_path)


def GetLGPFile(ymd):
    file_path = '%s/%s.csv' % (LGP_DIR_, FileNameFromDateTuple(ymd))
    return pd.read_csv(file_path)


def GetOfferFile(ymd):
    file_path = '%s/%s.csv' % (OFFERS_DIR_, FileNameFromDateTuple(ymd))
    return pd.read_csv(file_path)


def GetClearedOffersFile(ymd):
    file_path = '%s/%s.csv' % (CLEARED_OFFERS_DIR_, FileNameFromDateTuple(ymd))
    return pd.read_csv(file_path)


def GetGenerationFile(ym):
    file_path = '%s/%s.csv' % (GENERATION_DIR_, FileNameFromDateTuple(ym))
    return pd.read_csv(file_path)


def GetStructTime(year, month, day):
    d = datetime.datetime(year, month, day)
    return NZST.localize(d).timetuple()


def TodayAndTomorrow(year, month, day):
    today = datetime.datetime(year, month, day)
    tomorrow = today + datetime.timedelta(1)
    today = NZST.localize(today)
    tomorrow = NZST.localize(tomorrow)
    return today, tomorrow


def DSTBegins(year=None, month=None, day=None, today=None, tomorrow=None):
    if today is None:
        today, tomorrow = TodayAndTomorrow(year, month, day)
    return today.dst() < tomorrow.dst()


def DSTEnds(year=None, month=None, day=None, today=None, tomorrow=None):
    if today is None:
        today, tomorrow = TodayAndTomorrow(year, month, day)
    return today.dst() > tomorrow.dst()


def NumTPs(year, month, day):
    today, tomorrow = TodayAndTomorrow(year, month, day)
    if DSTEnds(today=today, tomorrow=tomorrow):
        return 50
    elif DSTBegins(today=today, tomorrow=tomorrow):
        return 46
    else:
        return 48


def TPIDToDateTime(tpid):
    year, month, day, tp = (int(x) for x in tpid.split('_'))
    today, tomorrow = TodayAndTomorrow(year, month, day)
    is_dst = None
    if DSTEnds(today=today, tomorrow=tomorrow):
        if tp <= 6:
            is_dst = True
            hour = (tp - 1) // 2
            minute = 30 * ((tp - 1) % 2)
        else:
            is_dst=False
            fold = 1
            hour = (tp - 3) // 2
            minute = 30 * ((tp - 3) % 2)
    elif DSTBegins(today=today, tomorrow=tomorrow):
        if tp <= 4:
            is_dst = False
            hour = (tp - 1) // 2
            minute = 30 * ((tp - 1) % 2)
        else:
            is_dst = True
            hour = (tp + 1) // 2
            minute = 30 * ((tp + 1) % 2)
    else:
        hour = (tp - 1) // 2
        minute = 30 * ((tp - 1) % 2)
       
    dt = datetime.datetime(year, month, day, hour=hour, minute=minute)
    dt = NZST.localize(dt, is_dst=is_dst)

    return dt


def GenerateTPIDs(start, end):
    [start_year, start_month, start_day, start_tp] = [int(x) for x in start.split('_')]
    [end_year, end_month, end_day, end_tp] = [int(x) for x in end.split('_')]
    
    years = list(range(start_year, end_year + 1))
    
    if len(years) == 1:
        yms = [(start_year, month) for month in range(start_month, end_month + 1)]
    else:
        yms = ([(start_year, month) for month in range(start_month, 13)] +
               [(year, month) for year in years[1:-1] for month in range(1, 13)] +
               [(end_year, month) for month in range(1, end_month + 1)])

    if len(yms) == 1:
        ymds = [(start_year, start_month, day)
                for day in range(start_day, end_day + 1)]
    else:
        ymds = ([(start_year, start_month, day)
                  for day in range(start_day,
                                   monthrange(start_year, start_month)[1] + 1)] +
                [(year, month, day)
                 for year, month in yms[1:-1]
                 for day in range(1, monthrange(year, month)[1] + 1)] +
                [(end_year, end_month, day) for day in range(1, end_day + 1)])

    if len(ymds) == 1:
        ymdtps = [(start_year, start_month, start_day, tp)
                  for tp in range(start_tp, end_tp + 1)]
    else:
        ymdtps = ([(start_year, start_month, start_day, tp)
                   for tp in range(start_tp,
                                   NumTPs(start_year, start_month, start_day) + 1)] + 
                  [(year, month, day, tp)
                   for year, month, day in ymds[1:-1]
                   for tp in range(1, NumTPs(year, month, day) + 1)] +
                  [(end_year, end_month, end_day, tp)
                   for tp in range(1, end_tp + 1)])
    
    return ['%d_%02d_%02d_%02d' % ymdtp for ymdtp in ymdtps]


def _LoadHolidays():
    holidays = pd.read_csv('Data/Holidays/holidays_1980_2020.csv')
    national_holidays = set([])
    for day in holidays.index.values:
        if holidays['NationalHoliday'][day] == 'Y':
            dmy = holidays['HolidayDateNZ'][day].split('/')
            padded_dmy = '%02d/%02d/%d' % tuple([int(d) for d in dmy])
            national_holidays.add(padded_dmy)
    return national_holidays


class InvalidDateException(Exception):
    pass


DATE_SEPARATORS_ = ['/', '-']

def DateTupleFromStr(date):
    parts = None
    for sep in DATE_SEPARATORS_:
        split = date.split(sep)
        if len(split) == 3:
            parts = split
            break

    if parts is None:
        raise InvalidDateException(date)

    first_number = int(parts[0])
    if first_number <= 31:
        parts.reverse()

    return (int(parts[0]), int(parts[1]), int(parts[2]))


def MakeTPID(year, month, day, tp):
    return '%d_%02d_%02d_%02d' % (year, month, day, tp)


def MakeTPIDSeries(date_series, tp_series):
    make_date_str = lambda d: '%d_%02d_%02d' % DateTupleFromStr(d)
    date_str_series = date_series.apply(make_date_str)
    tp_str_series = tp_series.apply(lambda tp: '%02d' % tp)
    return date_str_series + '_' + tp_str_series


def AddTPIDSeries(df, date_series_name, tp_series_name, as_index=True):
    df[TPID] = MakeTPIDSeries(df[date_series_name], df[tp_series_name])
    if as_index:
        df.set_index(TPID, inplace=True)

