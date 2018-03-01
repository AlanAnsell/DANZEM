import numpy as np
import pandas as pd

#import bisect
#import math
import time
import datetime
import pytz
import os
from calendar import monthrange

from . import nodes
from . import utils

NZST = pytz.timezone('Pacific/Auckland')
UTC = pytz.timezone('UTC')
DT_FORMAT = '%d/%m/%Y %H:%M:%S'

TPID = 'tpid'

GetLGPDir = lambda: os.path.join(utils.DATA_BASE_DIR,
                                 'LoadGenerationPrice')
GetBidsDir = lambda: os.path.join(utils.DATA_BASE_DIR,
                                  'Bids')
GetFinalPriceDir = lambda: os.path.join(utils.DATA_BASE_DIR,
                                        'FinalPrices')
GetOffersDir = lambda: os.path.join(utils.DATA_BASE_DIR,
                                    'Offers')
GetClearedOffersDir = lambda: os.path.join(utils.DATA_BASE_DIR,
                                           'ClearedOffers')
GetGenerationDir = lambda: os.path.join(utils.DATA_BASE_DIR,
                                        'Generation')


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


def GetYearStrFromTPID(tpid):
    return tpid[:4]


def GetMonthStrFromTPID(tpid):
    return tpid[:7]


def GetDayStrFromTPID(tpid):
    return tpid[:10]

    
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


def TupleFromTPID(tpid):
    return tuple(int(x) for x in tpid.split('_'))


def GetBidFile(ymd):
    file_name = FileNameFromDateTuple(ymd) + '.csv'
    file_path = os.path.join(GetBidsDir(), file_name)
    return pd.read_csv(file_path)


def GetFinalPriceFile(ym):
    file_name = FileNameFromDateTuple(ym) + '.csv'
    file_path = os.path.join(GetFinalPriceDir(), file_name)
    return pd.read_csv(file_path)


def GetLGPFile(ymd):
    file_name = FileNameFromDateTuple(ymd) + '.csv'
    file_path = os.path.join(GetLGPDir(), file_name)
    return pd.read_csv(file_path)


def GetOfferFile(ymd):
    file_name = FileNameFromDateTuple(ymd) + '.csv'
    file_path = os.path.join(GetOffersDir(), file_name)
    return pd.read_csv(file_path)


def GetClearedOffersFile(ymd):
    file_name = FileNameFromDateTuple(ymd) + '.csv'
    file_path = os.path.join(GetClearedOffersDir(), file_name)
    return pd.read_csv(file_path)


def GetGenerationFile(ym):
    file_name = FileNameFromDateTuple(ym) + '.csv'
    file_path = os.path.join(GetGenerationDir(), file_name)
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


def IsDST(year, month, day):
    return bool(GetStructTime(year, month, day).tm_isdst)


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


def TPRange(first, last):
    ymds = DayRange(first[:3], last[:3])
    first_tp = {ymd: 1 for ymd in ymds}
    first_tp[ymds[0]] = first[3]
    last_tp = {ymd: NumTPs(ymd[0], ymd[1], ymd[2]) for ymd in ymds}
    last_tp[ymds[-1]] = last[3]

    return [(ymd[0], ymd[1], ymd[2], tp)
            for ymd in ymds
            for tp in range(first_tp[ymd], last_tp[ymd] + 1)]


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

def DateTupleFromStr(date, us=False):
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
        if us:
            parts[:2] = reversed(parts[:2])
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


def AddIslandSeries(df, node_series_name):
    df['Island'] = df[node_series_name].apply(nodes.GetIslandFn())


def GetDataForRange(start, end, data_fn):
    dates = None
    if len(start) == 2:
        dates = MonthRange(start, end)
    else:
        dates = DayRange(start, end)
    return pd.concat([data_fn(date)
                      for date in dates])

