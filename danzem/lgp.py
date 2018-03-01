import numpy as np
import pandas as pd

from . import data
from . import nodes

LGP_LOAD_COL_ = 'Load (MW)'
LGP_DATE_COL_ = 'Date'
LGP_TP_COL_ = 'TradingPeriod'
LGP_NODE_COL_ = 'PointOfConnection'


def _CalculateLoadSum(df):
    summed = df.groupby(data.TPID).sum()
    return summed[LGP_LOAD_COL_]


def ExtractLoadInfo(df):
    data.AddTPIDSeries(df, LGP_DATE_COL_, LGP_TP_COL_)
    island_fn = nodes.GetIslandFn()
    wind_nodes = nodes.GetWindNodes()
    non_conforming = nodes.GetNonConformingNodes()

    non_conforming_rows = df[LGP_NODE_COL_].apply(lambda x:
                                                  x in non_conforming and
                                                  x not in wind_nodes)
    wind_rows = df[LGP_NODE_COL_].apply(lambda x: x in wind_nodes)
    conforming_rows = (~non_conforming_rows) & (~wind_rows)
    all_rows = df[LGP_NODE_COL_].apply(lambda x: True)
    
    categories = [('Non-Conforming', non_conforming_rows),
                  ('Conforming', conforming_rows),
                  ('Wind', wind_rows),
                  ('Total', all_rows)]

    islands = df[LGP_NODE_COL_].apply(island_fn)
    ni_rows = (islands == 'NI')
    si_rows = (islands == 'SI')
    island_rows = [('NI', ni_rows),
                   ('SI', si_rows)]

    load_df = pd.DataFrame()
    for category, c_rows in categories:
        for island, i_rows in island_rows:
            rows = c_rows & i_rows
            name = '%s %s Load (MW)' % (island, category)
            load_df[name] = _CalculateLoadSum(df[rows])

    return load_df
