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
    ym = (year, month)
    ymd = (year, month, day)
    
    offer_df = data.GetOfferFile(ymd)
    offered_stacks = GetLatestOffers(offer_df,
                                     wind=wind,
                                     minutes_before=minutes_before)
    offered_stacks.reset_index(inplace=True)
    #get_node = lambda x: x[:3]
    #offered_stacks['Node'] = offered_stacks['PointOfConnection'].apply(
    #        get_node)
    #offered_stacks.sort_values(['tpid', 'Node', 'DollarsPerMegawattHour'],
    #                           inplace=True)
    #offered_stacks.set_index(['tpid', 'Node'],
    #                         inplace=True)
    return offered_stacks
    #lgp_df = data.GetLGPFile(ymd)
    #data.AddTPIDSeries(lgp_df, 'Date', 'TradingPeriod', as_index=False)
    #lgp_df.sort_values(['tpid', 'PointOfConnection'], inplace=True)
    #lgp_df['Node'] = lgp_df['PointOfConnection'].apply(get_node)
    #grouped_lgp = lgp_df.groupby(['tpid', 'Node']).sum()


    
