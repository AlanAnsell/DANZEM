import numpy as np
import pandas as pd
import requests
import time

from os import listdir

from calendar import monthrange


def Today():
    ts = time.localtime()
    return (ts.tm_year, ts.tm_mon, ts.tm_mday) 


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


def MonthRange(first, last):
    years = list(range(first[0], last[0] + 1))
    first_month = {year: 1 for year in years}
    first_month[first[0]] = first[1]
    last_month = {year: 12 for year in years}
    last_month[last[0]] = last[1]
    return [(year, month)
            for year in years
            for month in range(first_month[year], last_month[year] + 1)]


def DayRange(first, last):
    yms = MonthRange(first[:2], last[:2])
    first_day = {ym: 1 for ym in yms}
    first_day[yms[0]] = first[2]
    last_day = {ym: monthrange(ym[0], ym[1])[1] for ym in yms}
    last_day[yms[-1]] = last[2]

    return [(ym[0], ym[1], day)
            for ym in yms
            for day in range(first_day[ym], last_day[ym] + 1)]


def GetFileDates(file_name):
    name = file_name.split('.')[0]
    if len(name) == 8:
        return [(int(name[:4]), int(name[4:6]), int(name[6:]))]
    if len(name) == 6:
        return [(int(name[:4]), int(name[4:]))]
    return []


def GetCurrent(dir_path):
    csvs = [f for f in listdir(dir_path) if f.endswith('.csv')]
    days_covered = []
    for csv in csvs:
        days_covered += GetFileDates(csv)
    return days_covered


def FileNameFromDateTuple(date):
    if len(date) == 2:
        return '%d%02d' % (date[0], date[1])
    else:
        return '%d%02d%02d' % (date[0], date[1], date[2])


def DateStrFromDateTuple(date):
    if len(date) == 2:
        return '%d/%02d' % (date[0], date[1])
    else:
        return '%d/%02d/%02d' % (date[0], date[1], date[2])


DATA_BASE_DIR_ = 'DANZEM/data'
STANDARD_URL_SUFFIX_ = ['.csv']
EXTENDED_URL_SUFFIXES_ = [
        '.csv', '_F.csv', 'x_F.csv', 'x.csv', '_I.csv', 'x_I.csv']


def DownloadDataset(name, url_fn, dates=[], url_suffixes=STANDARD_URL_SUFFIX_):
    file_dir = '%s/%s' % (DATA_BASE_DIR_, name)
    for date in dates:
        file_name = FileNameFromDateTuple(date)
        file_path = '%s/%s.csv' % (file_dir, file_name)
        url_base = url_fn(date)
        found = False
        for suffix in url_suffixes:
            url = '%s%s' % (url_base, suffix)
            if DownloadCSV(url, file_path, warn=False):
                found = True
                break
        if not found:
            date_str = DateStrFromDateTuple(date)
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
        return MonthRange(date, Today()[:2])
    else:
        return DayRange(date, Today())


def UpdateDataset(name, url_fn, start_date, hr_name=None,
                  url_suffixes=STANDARD_URL_SUFFIX_):
    if hr_name is None:
        hr_name = name

    date_range = DateRangeToToday(start_date)
    available_dates = set(date_range[:-2])
    file_dir = '%s/%s' % (DATA_BASE_DIR_, name)
    stored_dates = set(GetCurrent(file_dir))
    to_download = sorted(list(available_dates - stored_dates))
    print('Downloading %d new %s files' % (len(to_download), hr_name))
    DownloadDataset(name, url_fn, dates=to_download, url_suffixes=url_suffixes)


BID_URL_PREFIX_ = ('https://www.emi.ea.govt.nz/Datasets/Wholesale/'
                   'BidsAndOffers/Bids')
BID_DATA_START_ = (2012, 12, 13)

def UpdateBids():
    url_fn = MakeDayUrlFn(BID_URL_PREFIX_, 'Bids')
    UpdateDataset('Bids', url_fn, BID_DATA_START_)


FINAL_PRICE_URL_PREFIX_ = ('https://www.emi.ea.govt.nz/Wholesale/'
                           'Datasets/Final_pricing/Final_prices')
FINAL_PRICE_START_ = (2010, 1)

def UpdateFinalPrices():
    url_fn = MakeMonthUrlFn(FINAL_PRICE_URL_PREFIX_, 'Final_prices')
    UpdateDataset('FinalPrices', url_fn, FINAL_PRICE_START_)


LGP_URL_PREFIX_ = ('https://www.emi.ea.govt.nz/Wholesale/'
                   'Datasets/Final_pricing/Load_Generation_Price')
LGP_START_ = (2013, 1, 1)

def UpdateLGP():
    url_fn = MakeDayUrlFn(LGP_URL_PREFIX_, 'Load_Generation_Price')
    UpdateDataset('LoadGenerationPrice', url_fn, LGP_START_,
                  url_suffixes=EXTENDED_URL_SUFFIXES_)


OFFERS_URL_PREFIX_ = ('https://www.emi.ea.govt.nz/Wholesale/'
                      'Datasets/BidsAndOffers/Offers')
OFFERS_START_ = (2013, 1, 1)

def UpdateOffers():
    url_fn = MakeDayUrlFn(OFFERS_URL_PREFIX_, 'Offers')
    UpdateDataset('Offers', url_fn, OFFERS_START_)


CLEARED_OFFERS_URL_PREFIX_ = ('https://www.emi.ea.govt.nz/Wholesale/'
                              'Datasets/Final_pricing/Cleared_Offers')
CLEARED_OFFERS_START_ = OFFERS_START_

def UpdateClearedOffers():
    url_fn = MakeDayUrlFn(CLEARED_OFFERS_URL_PREFIX_, 'Cleared_Offers')
    UpdateDataset('ClearedOffers', url_fn, CLEARED_OFFERS_START_,
                  url_suffixes=EXTENDED_URL_SUFFIXES_)


GENERATION_URL_PREFIX_ = ('https://www.emi.ea.govt.nz/Wholesale/'
                          'Datasets/Generation/Generation_MD')
GENERATION_START_ = (2013, 1)

def UpdateGeneration():
    url_fn = MakeMonthUrlFn(GENERATION_URL_PREFIX_, 'Generation_MD')
    UpdateDataset('Generation', url_fn, GENERATION_START_)


def Update():
    UpdateBids()
    UpdateOffers()
    UpdateLGP()
    UpdateFinalPrices()
    UpdateClearedOffers()
    UpdateGeneration()

if __name__ == '__main__':
    Update()

