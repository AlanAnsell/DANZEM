import numpy as np
import pandas as pd

import os

from . import utils

NODES_DIR = os.path.join(utils.DATA_BASE_DIR, 'Nodes')

ISLANDS_PATH_ = NODES_DIR + '/islands.csv'

def GetIslandFn():
    islands_df = pd.read_csv(ISLANDS_PATH_, index_col=0)
    island_map = {node_prefix: islands_df['Island'].loc[node_prefix]
                  for node_prefix in islands_df.index}
    
    def Island(node):
        return island_map[node[:3]]

    return Island


WIND_NODES_PATH_ = NODES_DIR + 'wind_nodes.csv'

def GetWindNodes():
    wind_nodes_df = pd.read_csv(WIND_NODES_PATH_)
    return (set(wind_nodes_df['Node']) |
            set([x.split()[0] for x in wind_nodes_df['Node']]))


