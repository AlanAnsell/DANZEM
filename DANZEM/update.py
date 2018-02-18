import numpy as np
import pandas as pd
import requests
import time

import os
from calendar import monthrange

from . import data


def DownloadCSV(url, file_path, warn=True):
    print('Downloading %s' % url)
    response = requests.get(url)
    if response.status_code != 200:
        if warn:
            print('Warning: request to %s failed with code %d' %
                    (url, response.status_code))
        return False
    with open(file_path, 'w') as f:
        f.write(response.text.replace('\r', ''))
    return True


def GetFileDates(file_name):
    name = file_name.split('.')[0]
    if len(name) == 8:
        return [(int(name[:4]), int(name[4:6]), int(name[6:]))]
    if len(name) == 6:
        return [(int(name[:4]), int(name[4:]))]
    return []


def GetCurrent(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    csvs = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    days_covered = []
    for csv in csvs:
        days_covered += GetFileDates(csv)
    return days_covered


STANDARD_URL_SUFFIX_ = ['.csv']
EXTENDED_URL_SUFFIXES_ = [
        '.csv', '_F.csv', 'x_F.csv', 'x.csv', '_I.csv', 'x_I.csv']


def DownloadDataset(name, file_dir, url_fn, dates=[],
                    url_suffixes=STANDARD_URL_SUFFIX_):
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    for date in dates:
        file_name = data.FileNameFromDateTuple(date)
        file_path = '%s/%s.csv' % (file_dir, file_name)
        url_base = url_fn(date)
        found = False
        for suffix in url_suffixes:
            url = '%s%s' % (url_base, suffix)
            if DownloadCSV(url, file_path, warn=False):
                found = True
                break
        if not found:
            date_str = data.DateStrFromDateTuple(date)
            print('**** Warning: could not find any %s file for %s' % (
                name, date_str))


def MakeDayUrlFn(url_prefix, after_date):
    def DayUrlFn(ymd):
        return '%s/%d/%d%02d%02d_%s' % (url_prefix, ymd[0], ymd[0], ymd[1],
                                        ymd[2], after_date)
    return DayUrlFn


def MakeMonthUrlFn(url_prefix, after_date):
    def MonthUrlFn(ym):
        return '%s/%d%02d_%s' % (url_prefix, ym[0], ym[1], after_date)
    return MonthUrlFn


def DateRangeToToday(date):
    if len(date) == 2:
        return data.MonthRange(date, data.Today()[:2])
    else:
        return data.DayRange(date, data.Today())


def UpdateDataset(name, file_dir, url_fn, start_date,
                  url_suffixes=STANDARD_URL_SUFFIX_):
    date_range = DateRangeToToday(start_date)
    available_dates = set(date_range[:-2])
    stored_dates = set(GetCurrent(file_dir))
    to_download = sorted(list(available_dates - stored_dates))
    print('Downloading %d new %s files' % (len(to_download), name))
    DownloadDataset(name, file_dir, url_fn, dates=to_download,
                    url_suffixes=url_suffixes)


BID_URL_PREFIX_ = ('https://www.emi.ea.govt.nz/Datasets/Wholesale/'
                   'BidsAndOffers/Bids')
BID_DATA_START_ = (2012, 12, 13)

def UpdateBids():
    url_fn = MakeDayUrlFn(BID_URL_PREFIX_, 'Bids')
    UpdateDataset('Bids', data.BIDS_DIR_, url_fn, BID_DATA_START_)


FINAL_PRICE_URL_PREFIX_ = ('https://www.emi.ea.govt.nz/Wholesale/'
                           'Datasets/Final_pricing/Final_prices')
FINAL_PRICE_START_ = (2010, 1)

def UpdateFinalPrices():
    url_fn = MakeMonthUrlFn(FINAL_PRICE_URL_PREFIX_, 'Final_prices')
    UpdateDataset('FinalPrices', data.FINAL_PRICE_DIR_,
                  url_fn, FINAL_PRICE_START_)


LGP_URL_PREFIX_ = ('https://www.emi.ea.govt.nz/Wholesale/'
                   'Datasets/Final_pricing/Load_Generation_Price')
LGP_START_ = (2013, 1, 1)

def UpdateLGP():
    url_fn = MakeDayUrlFn(LGP_URL_PREFIX_, 'Load_Generation_Price')
    UpdateDataset('LoadGenerationPrice', data.LGP_DIR_, url_fn,
                  LGP_START_, url_suffixes=EXTENDED_URL_SUFFIXES_)


OFFERS_URL_PREFIX_ = ('https://www.emi.ea.govt.nz/Wholesale/'
                      'Datasets/BidsAndOffers/Offers')
OFFERS_START_ = (2013, 1, 1)

def UpdateOffers():
    url_fn = MakeDayUrlFn(OFFERS_URL_PREFIX_, 'Offers')
    UpdateDataset('Offers', data.OFFERS_DIR_, url_fn, OFFERS_START_)


CLEARED_OFFERS_URL_PREFIX_ = ('https://www.emi.ea.govt.nz/Wholesale/'
                              'Datasets/Final_pricing/Cleared_Offers')
CLEARED_OFFERS_START_ = OFFERS_START_

def UpdateClearedOffers():
    url_fn = MakeDayUrlFn(CLEARED_OFFERS_URL_PREFIX_, 'Cleared_Offers')
    UpdateDataset('ClearedOffers', data.CLEARED_OFFERS_DIR_, url_fn,
                  CLEARED_OFFERS_START_, url_suffixes=EXTENDED_URL_SUFFIXES_)


GENERATION_URL_PREFIX_ = ('https://www.emi.ea.govt.nz/Wholesale/'
                          'Datasets/Generation/Generation_MD')
GENERATION_START_ = (2013, 1)

def UpdateGeneration():
    url_fn = MakeMonthUrlFn(GENERATION_URL_PREFIX_, 'Generation_MD')
    UpdateDataset('Generation', data.GENERATION_DIR_, url_fn,
                  GENERATION_START_)


def Update():
    UpdateBids()
    UpdateOffers()
    UpdateLGP()
    UpdateFinalPrices()
    UpdateClearedOffers()
    UpdateGeneration()

if __name__ == '__main__':
    Update()

