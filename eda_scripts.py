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
    df['hour'] = df['date'].dt.hour
    df['hour_cos'] = np.cos( 2 * math.pi * df['hour'] / 24)
    df['hour_sin'] = np.sin( 2 * math.pi * df['hour'] / 24)
    holidays = get_holidays()['holidays'] + get_holidays()['preholidays']
    df['bank_holidays'] = np.where(df['date_date'].isin(holidays), 1, 0)    
    return df
