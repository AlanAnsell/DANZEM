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


BID_DATA_DIR_ = 'DANZEM/data/Bids'
BID_URL_PREFIX_ = 'https://www.emi.ea.govt.nz/Datasets/Wholesale/BidsAndOffers/Bids'
BID_DATA_START_ = (2012, 12, 13)

def DownloadBids(bid_files):
    for year, month, day in bid_files:
        url = '%s/%d/%d%02d%02d_Bids.csv' % (
                BID_URL_PREFIX_, year, year, month, day)
        file_path = '%s/%d%02d%02d.csv' % (BID_DATA_DIR_, year, month, day)
        DownloadCSV(url, file_path)


def UpdateBids():
    day_range = DayRange(BID_DATA_START_, Today())
    all_days = set(day_range[:-2])
    current_days = set(GetCurrent(BID_DATA_DIR_))
    to_download = sorted(list(all_days - current_days))
    print('Downloading %d new bid files' % len(to_download))
    DownloadBids(to_download)


FINAL_PRICE_DIR_ = 'DANZEM/data/FinalPrices'
FINAL_PRICE_URL_PREFIX_ = 'https://www.emi.ea.govt.nz/Wholesale/Datasets/Final_pricing/Final_prices'
FINAL_PRICE_START_ = (2010, 1)

def DownloadFinalPrices(final_price_files):
    for year, month in final_price_files:
        url = '%s/%d%02d_Final_prices.csv' % (FINAL_PRICE_URL_PREFIX_, year, month)
        file_path = '%s/%d%02d.csv' % (FINAL_PRICE_DIR_, year, month)
        DownloadCSV(url, file_path)


def UpdateFinalPrices():
    month_range = MonthRange(FINAL_PRICE_START_, Today()[:2])
    all_months = set(month_range[:-2])
    current_months = set(GetCurrent(FINAL_PRICE_DIR_))
    to_download = sorted(list(all_months - current_months))
    print('Downloading %d new final price files' % len(to_download))
    DownloadFinalPrices(to_download)


LGP_DIR_ = 'DANZEM/data/LoadGenerationPrice'
LGP_URL_PREFIX_ = 'https://www.emi.ea.govt.nz/Wholesale/Datasets/Final_pricing/Load_Generation_Price'
LGP_SUFFIXES_ = ['.csv', '_F.csv', 'x_F.csv', 'x.csv', '_I.csv', 'x_I.csv']
LGP_START_ = (2013, 1, 1)

def DownloadLGP(lgp_files):
    for year, month, day in lgp_files:
        file_path = '%s/%d%02d%02d.csv' % (LGP_DIR_, year, month, day)
        found = False
        for suffix in LGP_SUFFIXES_:
            url = '%s/%d/%d%02d%02d_Load_Generation_Price%s' % (
                    LGP_URL_PREFIX_, year, year, month, day, suffix)
            if DownloadCSV(url, file_path, warn=False):
                found = True
                break
        if not found:
            print('**** Warning: could not find any LGP file for %d/%02d/%02d' % (
                year, month, day))


def UpdateLGP():
    day_range = DayRange(LGP_START_, Today())
    all_days = set(day_range[:-2])
    current_days = set(GetCurrent(LGP_DIR_))
    to_download = sorted(list(all_days - current_days))
    print('Downloading %d new LoadGenerationPrice files' % len(to_download))
    DownloadLGP(to_download)


OFFERS_DIR_ = 'DANZEM/data/Offers'
OFFERS_URL_PREFIX_ = 'https://www.emi.ea.govt.nz/Wholesale/Datasets/BidsAndOffers/Offers'
OFFERS_START_ = (2013, 1, 1)

def DownloadOffers(offer_files):
    for year, month, day in offer_files:
        url = '%s/%d/%d%02d%02d_Offers.csv' % (
                OFFERS_URL_PREFIX_, year, year, month, day)
        file_path = '%s/%d%02d%02d.csv' % (OFFERS_DIR_, year, month, day)
        DownloadCSV(url, file_path)


def UpdateOffers():
    day_range = DayRange(OFFERS_START_, Today())
    all_days = set(day_range[:-2])
    current_days = set(GetCurrent(OFFERS_DIR_))
    to_download = sorted(list(all_days - current_days))
    print('Downloading %d new offer files' % len(to_download))
    DownloadOffers(to_download)


def Update():
    UpdateBids()
    UpdateOffers()
    UpdateLGP()
    UpdateFinalPrices()


if __name__ == '__main__':
    Update()

