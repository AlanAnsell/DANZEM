"""Functions for loading and processing training data."""
import numpy as np
import pandas as pd
from matplotlib import dates
from matplotlib import pyplot as plt
import seaborn as sns

import bisect
import math
import time
import datetime
import pytz
from calendar import monthrange

NZST = pytz.timezone('Pacific/Auckland')
UTC = pytz.timezone('UTC')
DT_FORMAT = '%d/%m/%Y %H:%M:%S'

TPID = 'tpid'

_HEADER_DATE_TIME = 'DateTime'

PRICE = 'price'
TIME_PERIOD = 'time_period'
YEAR = 'year'
MONTH = 'month'
DAY_OF_WEEK = 'day_of_week'
IS_NATIONAL_HOLIDAY = 'is_national_holiday'
DAY_INDEX = 'day_index'
DAY_AGO_PRICE = 'day_ago_price'
TP_ID = 'id'

_TIME_TO_TIME_PERIOD = {}
_TIME_PERIOD_TO_TIME = []

minutes = ['00', '30']
for hour_num in range(24):
    for minute_num in range(2):
        time_str = '%02d:%s' % (hour_num, minutes[minute_num])
        _TIME_TO_TIME_PERIOD[time_str] = 2 * hour_num + minute_num + 1
        _TIME_PERIOD_TO_TIME.append(time_str)


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
    fold = 0
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
        is_dst = None
        hour = (tp - 1) // 2
        minute = 30 * ((tp - 1) % 2)
       
    #print('%d:%02d %d' % (hour, minute, fold))

    dt = datetime.datetime(year, month, day, hour=hour, minute=minute)
    dt = NZST.localize(dt, is_dst=is_dst)
    #dt.fold = fold

    return dt
    #return datetime.datetime(year, month, day, hour=hour, minute=minute,
    #                         tzinfo=NZST, fold=fold)


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


#def MakeTime(date_, tp):
#    date = str(date)
#    year = int(date[:4])
#    month = int(date[4:6])
#    day = int(date[6:])


def _IsValidDateTimeString(s):
    return ';' not in s # check for a typo which occurs in some time strings


def _TimestampFromString(s):
    return time.mktime(time.strptime(s, '%d-%b-%Y %H:%M'))

def _TimestampFromHolidayDateString(s):
    return time.mktime(time.strptime(s, '%d/%m/%Y'))

def _TimestampToDayIndex(timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp)
    return int(dates.date2num(dt))

def _TimestampToNZDay(timestamp):
    return time.strftime('%d/%m/%Y', time.localtime(timestamp))

def _TimeStringToTimePeriod(s):
    return _TIME_TO_TIME_PERIOD[s[-5:]]


def _Processed(data, holidays):
    data = data[data[_HEADER_DATE_TIME].apply(_IsValidDateTimeString)]
    examples = pd.DataFrame()
    examples[PRICE] = data[_HEADER_PRICE]
    date_time = data[_HEADER_DATE_TIME]
    #date_time = data[_HEADER_DATE_TIME].apply(_FixDateTimeString)
    timestamp = date_time.apply(_TimestampFromString)
    examples[DAY_INDEX] = timestamp.apply(_TimestampToDayIndex)
    examples[TIME_PERIOD] = date_time.apply(_TimeStringToTimePeriod)
    localtime = timestamp.apply(lambda t: time.localtime(t))
    examples[YEAR] = localtime.apply(lambda t: t.tm_year)
    examples[MONTH] = localtime.apply(lambda t: t.tm_mon)
    examples[DAY_OF_WEEK] = localtime.apply(lambda t: t.tm_wday)
    nz_day = timestamp.apply(_TimestampToNZDay)
    examples[IS_NATIONAL_HOLIDAY] = nz_day.apply(
            lambda d: float(d in holidays))
    examples.reset_index(drop=True, inplace=True)
    day_of_month = localtime.apply(lambda t: t.tm_mday)
    day_of_month.reset_index(drop=True, inplace=True)
    examples[TP_ID] = ['%d_%02d_%02d_%02d' % (examples[YEAR][i],
                                              examples[MONTH][i],
                                              day_of_month[i],
                                              examples[TIME_PERIOD][i])
                       for i in examples.index]
    return examples 


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


def AddTPIDSeries(df, date_series_name, tp_series_name):
    df[TPID] = MakeTPIDSeries(df[date_series_name], df[tp_series_name])
    df.set_index(TPID, inplace=True)

