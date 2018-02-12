import numpy as np
import pandas as pd

from . import data
from . import nodes

LGP_LOAD_COL_ = 'Load (MW)'
LGP_DATE_COL_ = 'Date'
LGP_TP_COL_ = 'TradingPeriod'
LGP_NODE_COL_ = 'PointOfConnection'


def CalculateLoadSum(df, column, group_fn):
    mask = df[column].apply(group_fn)
    summed = df[mask].groupby(data.TPID).sum()
    return summed[LGP_LOAD_COL_]


def ExtractLoadInfo(df):
    data.AddTPIDSeries(df, LGP_DATE_COL_, LGP_TP_COL_)
    island_fn = nodes.GetIslandFn()
    wind_nodes = nodes.GetWindNodes()

    def make_classifier_fn(island):
        return lambda x: island_fn(x) == island #and x not in wind_nodes

    ni = CalculateLoadSum(df, LGP_NODE_COL_, make_classifier_fn('NI'))
    si = CalculateLoadSum(df, LGP_NODE_COL_, make_classifier_fn('SI'))
    return pd.DataFrame({'NI Load (MW)': ni, 'SI Load (MW)': si})

