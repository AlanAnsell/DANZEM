import numpy as np
import pandas as pd

import datetime

from . import data
from . import nodes
from . import utils


def UTCInfoToDateTime(info):
    year, month, day, time = (x for x in info.split('-'))
    year = int(year)
    month = int(month)
    day = int(day)
    hour, minute, second = (x for x in time.split(':'))
    hour = int(hour)
    minute = int(minute)
    second = int(float(second))
    return datetime.datetime(year, month, day, hour=hour, minute=minute,
                             second=second, tzinfo=data.UTC)
    

def GetLatestOffers(df, minutes_before=0, wind=False):
    df = df[df['ProductType'] == 'Energy'].copy()
    if not wind:
        wind_nodes = nodes.GetWindNodes()
        is_not_wind = [x not in wind_nodes for x in df['PointOfConnection']]
        df = df[is_not_wind].copy()
    
    df['tpid'] = data.MakeTPIDSeries(df['TradingDate'],
                                     df['TradingPeriod'])
    tp_timestamp_fn = lambda x: data.TPIDToDateTime(x).timestamp()
    df['TPBeginTimestamp'] = utils.FastApply(df['tpid'], tp_timestamp_fn)
   
    utc_info_to_nzst_ts = (
            lambda x: UTCInfoToDateTime(x).astimezone(data.NZST).timestamp())
    df['UTCInfo'] = df['UTCSubmissionDate'] + '-' + df['UTCSubmissionTime']
    df['NZSTSubmissionTimestamp'] = utils.FastApply(df['UTCInfo'],
                                                    utc_info_to_nzst_ts)
   
    df['MinutesInAdvance'] = (df['TPBeginTimestamp'] -
                              df['NZSTSubmissionTimestamp']) / 60.0
    keep = df['MinutesInAdvance'].apply(lambda x: x >= minutes_before)
    df = df[keep].copy()

    df.sort_values('MinutesInAdvance', inplace=True)
    return df.groupby(['tpid', 'PointOfConnection', 'Unit',
                       'Trader', 'Band']).first()


def GetStacks(year, month, day, minutes_before=0, wind=False):
    #ym = (year, month)
    #ymd = (year, month, day)
    
    #offer_df = data.GetOfferFile(ymd)
    #offered_stacks = GetLastestOffers(offer_df,
    #                                  minutes_before=minutes_before,
    #                                  wind=wind)
    pass
    
