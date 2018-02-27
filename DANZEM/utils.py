import pandas as pd

import os

DATA_BASE_DIR = os.path.join(os.path.dirname(__file__), 'data')


def SetDir(dir_):
    global DATA_BASE_DIR
    DATA_BASE_DIR = dir_


def OneHot(df, columns=None, add_df=None):
    if columns is None:
        columns = df.columns
    if isinstance(columns, str):
        columns = [columns]
    if add_df is None:
        to_return = True
        add_df = pd.DataFrame()
    else:
        to_return = False

    for col in columns:
        series = df[col]
        values = sorted(series.unique())
        for value in values:
            col_name = '%s=%s' % (col, str(value))
            add_df[col_name] = series.apply(lambda x: float(x == value))
    
    if to_return:
        return add_df


def FastApply(series, fn, as_list=False):
    fn_map = {}
    for x in series:
        if x not in fn_map:
            fn_map[x] = fn(x)
    
    if as_list:
        return [fn_map[x] for x in series]
    else:
        return series.apply(lambda x: fn_map[x])

