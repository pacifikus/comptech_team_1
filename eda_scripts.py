# importing the libraries 
import pandas as pd
import numpy as np
import math
import json
# 

# function to get df from csv file
def get_data(path):
    '''
    The function receives the path to csv file with orders data 
    (either train or test) and returns the dataframe for further processing
    '''
    df = pd.read_csv(path, parse_dates=['date'])
    
    return df


def get_season(mo):
    '''
    The function receives month as a number and returns a season
    '''
    if (mo > 11 or mo < 3):
        return 1
    elif (mo == 3 or mo <= 5):
        return 2
    elif (mo >=6 and mo <= 8):
        return 3
    else:
        return 4
    
def get_holidays(path='./data/calendar_2021.json'):
    # https://github.com/d10xa/holidays-calendar/blob/master/json/consultant2021.json
    # open file, read it and close
    with open(path, 'r') as f:
        jsonText = f.read()     
    # text to dictionary
    jsonObj = json.loads(jsonText)
    return jsonObj

def get_days_before_nearest_day(row, calendar):
    current_doy = row['doy']
    days = min(calendar[calendar['doy'] >= current_doy]['doy']) - current_doy
    return days

def get_days_after_nearest_day(row, calendar):
    current_doy = row['doy']
    days = max(calendar[calendar['doy'] <= current_doy]['doy']) - current_doy
    return days
    
def get_features(df):
    '''
    The function receives the dataframe and returns df with new features
    '''    
    df['year'] = df['date'].dt.year
    df['date_date'] = df['date'].dt.strftime('%Y-%m-%d') 
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['season'] = df['date'].dt.month.apply(lambda x: get_season(x))
    df['week'] = df['date'].dt.isocalendar().week
    df['dow'] = df['date'].dt.dayofweek
    df['doy'] = df['date'].dt.day_of_year # нужно для других функций
    df['hour'] = df['date'].dt.hour
    df['hour_cos'] = np.cos( 2 * math.pi * df['hour'] / 24)
    df['hour_sin'] = np.sin( 2 * math.pi * df['hour'] / 24)
    bank_holidays = get_holidays()['holidays'] + get_holidays()['preholidays']
    df['bank_holidays'] = np.where(df['date_date'].isin(bank_holidays), 1, 0)
    
    # before and after holidays
    
    ## dics
    #### of holidays
    official_holidays = pd.DataFrame(
                ['2021-01-01', '2021-01-02', '2021-01-03', 
                 '2021-01-04', '2021-01-05', '2021-01-06', 
                 '2021-01-07', '2021-01-08', '2021-02-23', 
                 '2021-03-08', '2021-05-01', '2021-05-09', 
                 '2021-06-12', '2021-11-04', '2021-12-31'],
                columns=['date'])
    official_holidays['date'] = pd.to_datetime(official_holidays['date'])
    official_holidays['doy'] = official_holidays['date'].dt.day_of_year
    #### celebrations
    celebrations = pd.DataFrame(
                ['2021-01-01', '2021-02-23', '2021-02-14', 
                '2021-03-08', '2021-09-01', '2021-12-31'],
                columns=['date'])
    celebrations['date'] = pd.to_datetime(celebrations['date'])
    celebrations['doy'] = celebrations['date'].dt.day_of_year    
    #### for faster performance
    doy_df = df[['date_date', 'doy']].copy()
    doy_df = doy_df.drop_duplicates()
    doy_df['days_before_of_holiday'] = doy_df.apply(lambda x: get_days_before_nearest_day(x, official_holidays), axis=1)
    doy_df['days_before_celebrations'] = doy_df.apply(lambda x: get_days_before_nearest_day(x, celebrations), axis=1)
    doy_df['days_after_of_holiday'] = doy_df.apply(lambda x: get_days_after_nearest_day(x, official_holidays), axis=1)
    doy_df['days_after_celebrations'] = doy_df.apply(lambda x: get_days_after_nearest_day(x, celebrations), axis=1)
    df = df.merge(doy_df, on=['doy', 'date_date'], how='left')
    
    return df
