import pandas as pd

import datetime
import os

from . import data
from . import utils


GetHolidaysPath = lambda: os.path.join(utils.DATA_BASE_DIR,
                                       'Periods',
                                       'holidays_1980_2020.csv')


def GetNationalHolidays():
    holiday_df = pd.read_csv(GetHolidaysPath())
    holiday_df = holiday_df[holiday_df['NationalHoliday'] == 'Y']
    holiday_dates = {data.DateTupleFromStr(date, us=True)
                     for date in holiday_df['HolidayDate']}
    observed_dates = {data.DateTupleFromStr(date, us=True)
                      for date in holiday_df['ObservedDate']}
    return holiday_dates, observed_dates


def GetTPData(start, end):
    if isinstance(start, str):
        start = data.TupleFromTPID(start)
    if isinstance(end, str):
        end = data.TupleFromTPID(end)

    tps = data.TPRange(start, end)
    ymds = [tp[:3] for tp in tps]
    tpids = [data.MakeTPID(*tp) for tp in tps]
    year = [tp[0] for tp in tps]
    month = [tp[1] for tp in tps]
    time_period = [tp[3] for tp in tps]
    
    holiday_dates, observed_dates = GetNationalHolidays()
    is_national_holiday = [ymd in holiday_dates
                           for ymd in ymds]
    is_observed_holiday = [ymd in observed_dates
                           for ymd in ymds]

    is_dst = [data.IsDST(*ymd) for ymd in ymds]
    today_and_tomorrow = [data.TodayAndTomorrow(*ymd)
                          for ymd in ymds]
    week_day = [tt[0].weekday()
                for tt in today_and_tomorrow]
    year_day = [tt[0].timetuple().tm_yday
                for tt in today_and_tomorrow]
    dst_begins = [data.DSTBegins(today=tt[0], tomorrow=tt[1])
                  for tt in today_and_tomorrow]
    dst_ends = [data.DSTEnds(today=tt[0], tomorrow=tt[1])
                for tt in today_and_tomorrow]

    tp_df = pd.DataFrame({'year': year,
                          'month': month,
                          'weekday': week_day,
                          'yearday': year_day,
                          'time_period': time_period,
                          'is_national_holiday': is_national_holiday,
                          'is_observed_holiday': is_observed_holiday,
                          'is_dst': is_dst,
                          'dst_begins': dst_begins,
                          'dst_ends': dst_ends}, index=tpids)
    for col in ['is_national_holiday',
                'is_observed_holiday',
                'is_dst',
                'dst_begins',
                'dst_ends']:
        tp_df[col] = tp_df[col].apply(float)

    return tp_df

