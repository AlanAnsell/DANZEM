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

tz = pytz.timezone('Pacific/Auckland')

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
    return tz.localize(d).timetuple()


def DSTBegins(year, month, day):
    today = datetime.date(year, month, day)
    tomorrow = today + datetime.timedelta(1)
    today_dt = GetStructTime(year, month, day)
    tomorrow_dt = GetStructTime(tomorrow.year, tomorrow.month, tomorrow.day)
    return (today_dt.tm_isdst == 0 and tomorrow_dt.tm_isdst == 1)


def DSTEnds(year, month, day):
    today = datetime.date(year, month, day)
    tomorrow = today + datetime.timedelta(1)
    today_dt = GetStructTime(year, month, day)
    tomorrow_dt = GetStructTime(tomorrow.year, tomorrow.month, tomorrow.day)
    return (today_dt.tm_isdst == 1 and tomorrow_dt.tm_isdst == 0)


def NumTPs(year, month, day):
    today = datetime.date(year, month, day)
    tomorrow = today + datetime.timedelta(1)
    today_dt = GetStructTime(year, month, day)
    tomorrow_dt = GetStructTime(tomorrow.year, tomorrow.month, tomorrow.day)
    if today_dt.tm_isdst == tomorrow_dt.tm_isdst:
        return 48
    if today_dt.tm_isdst == 1 and tomorrow_dt.tm_isdst == 0:
        return 50
    else:
        return 46


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


def MakeTime(date_, tp):
    date = str(date)
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:])


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


def GetPriceDF(node, year):
    file_path = 'Data/Prices/Nodes/%s/all_prices_%d.csv' % (node, year)
    data = pd.read_csv(file_path)
    data = data[data[_HEADER_DATE_TIME].apply(_IsValidDateTimeString)]
    date_time = data[_HEADER_DATE_TIME]
    timestamp = date_time.apply(_TimestampFromString)
    time_period = date_time.apply(_TimeStringToTimePeriod)
    localtime = timestamp.apply(lambda t: time.localtime(t))
    year = localtime.apply(lambda t: t.tm_year)
    month = localtime.apply(lambda t: t.tm_mon)
    day_of_month = localtime.apply(lambda t: t.tm_mday)
    data[TP_ID] = ['%d_%02d_%02d_%02d' % (year[i],
                                          month[i],
                                          day_of_month[i],
                                          time_period[i])
                    for i in data.index]
    data['price'] = data[_HEADER_PRICE]
    data.set_index(TP_ID, inplace=True)
    return data[['price']]


def LoadOtahuhu():
    years = []
    for year in range(2004, 2017):
        file_path = 'Data/Prices/Otahuhu/otahuhu_node_results_%d.csv' % year
        data = pd.read_csv(file_path)
        data = data[data['Node'] == _OTAHUHU_MAIN]
        years.append(data)
    otahuhu = pd.concat(years)
    holidays = _LoadHolidays()
    return _Processed(otahuhu, holidays)


def _BinaryFeatureMatrix(feature, name):
    values = feature.unique()
    matrix = pd.DataFrame()
    for value in values:
        column_name = '%s_%s' % (name, str(value))
        matrix[column_name] = feature.apply(lambda x: float(x == value))
    return matrix


def RollExamples(examples, prev=48, step=8, n_steps=48):
    sequences = []
    for i in examples.index[prev:]:
        range_end = i - prev
        range_start = max(range_end - step * n_steps, examples.index[0] + (i - examples.index[0]) % step)
        indices = range(range_start, range_end + 1, step)
        sequence = examples.loc[indices]
        sequence['id'] = examples['id'][i]
        sequences.append(sequence)
    return pd.concat(sequences, ignore_index=True)


def GetExamples(data, day_ago_price=False):
    targets = data[PRICE]
    time_period_mat = _BinaryFeatureMatrix(data[TIME_PERIOD], TIME_PERIOD)
    month_mat = _BinaryFeatureMatrix(data[MONTH], MONTH)
    day_of_week_mat = _BinaryFeatureMatrix(data[DAY_OF_WEEK], DAY_OF_WEEK)
    examples = pd.concat([time_period_mat, month_mat, day_of_week_mat], axis=1)
    examples[IS_NATIONAL_HOLIDAY] = data[IS_NATIONAL_HOLIDAY]
    if day_ago_price:
        price_at_time_period = {
                (data[DAY_INDEX][i],
                 data[TIME_PERIOD][i]): data[PRICE][i]
                for i in data.index}
        has_day_ago_price = [
                (data[DAY_INDEX][i] - 1,
                 data[TIME_PERIOD][i]) in price_at_time_period
                for i in data.index]
        examples = examples[has_day_ago_price]
        targets = targets[has_day_ago_price]
        examples[DAY_AGO_PRICE] = [
                price_at_time_period[
                    (data[DAY_INDEX][i] - 1,
                     data[TIME_PERIOD][i])]
                for i in examples.index]
    return examples, targets

if __name__ == '__main__':
    otahuhu = LoadOtahuhu()
    # distribution_sample = np.random.choice(otahuhu[PRICE], size=100000, replace=False)
    prices = otahuhu[PRICE][otahuhu[PRICE] < 500.0]
    uniform_price_transformation = UniformTransformation(prices, n_partitions=100)
    #print(uniform_price_transformation.quantiles)
    uniform_prices = [uniform_price_transformation.transform(x) for x in prices]
    #plt.hist(uniform_prices, bins=50)
    #plt.show()
    inverse_transformed_prices = [uniform_price_transformation.inverse_transform(y)
                                  for y in uniform_prices]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)

    ax[0].hist(prices, bins=50)
    ax[1].hist(inverse_transformed_prices, bins=50)
    plt.show()


    
