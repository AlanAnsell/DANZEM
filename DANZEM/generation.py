import numpy as np
import pandas as pd

from . import data


def ProcessGenerationFileForTP(df, tp):
    df['TPNum'] = tp
    df['tpid'] = data.MakeTPIDSeries(df['Trading_date'], df['TPNum'])
    df.drop(['Trading_date', 'TPNum'], axis=1,inplace=True)

    df.rename(columns={'TP%d' % tp: 'Generation (MW)',
                       'POC_Code': 'PointOfConnection',
                       'Nwk_Code': 'Trader'}, inplace=True)
    df['Generation (MW)'] /= 500

    to_drop = ['TP%d' % i for i in range(1, 51) if i != tp]
    df.drop(to_drop, axis=1, inplace=True)
    return df


def ProcessGenerationFile(df):
    tps = [ProcessGenerationFileForTP(df.copy(), i)
           for i in range(1, 51)]
    full_df = pd.concat(tps, ignore_index=True)
    not_na = full_df['Generation (MW)'].apply(lambda x: not np.isnan(x))
    return full_df[not_na].copy()
