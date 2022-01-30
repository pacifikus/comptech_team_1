import os

import logging
import holidays
import pandas as pd
from tqdm import tqdm

from src.utils import read_config, get_test_data, timeseries_train_test_split, save_all_dfs


logging.basicConfig(level=logging.DEBUG)

config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
config = read_config(config_path)
tqdm.pandas()


def get_orders():
    return pd.read_csv(config['preprocessing']['orders_path'], parse_dates=['date'])


def get_dayoffs():
    return [pd.to_datetime(day) for day in config['preprocessing']['dayoffs']]


def get_holidays():
    holidays_dict = holidays.RU(years=2021)
    df_holidays = pd.DataFrame.from_dict(holidays_dict, orient='index').reset_index()
    df_holidays.columns = ['ds', 'holiday']
    df_holidays['ds'] = pd.to_datetime(df_holidays['ds'])
    df_holidays['doy'] = df_holidays['ds'].dt.dayofyear
    return df_holidays


def split_date_time(data):
    logging.info('Getting date features')
    data['dttm'] = pd.to_datetime(data['dttm'])
    data['year'] = data['dttm'].dt.year
    data['month'] = data['dttm'].dt.month
    data['week'] = data['dttm'].dt.week
    data['dayofweek'] = data['dttm'].dt.dayofweek
    data['doy'] = data['dttm'].dt.dayofyear
    return data


def group_by_date(data):
    data['date'] = data['date'].dt.date
    data = data.groupby(['delivery_area_id', 'date']).sum()['orders_cnt'].reset_index()
    return data


def get_prev_next_features(data):
    logging.info('Getting base features from past and future')
    data = data.sort_values(by=['delivery_area_id', 'date'])

    for i in range(1, config['preprocessing']['prev_cols_cnt'] + 1):
        data['prev_' + str(i)] = data.groupby(['delivery_area_id', 'date'], as_index=False) \
            .mean().shift(i).rolling(1).sum()['orders_cnt']

    data["future_1"] = data['orders_cnt'].copy()
    for i in range(2, config['preprocessing']['next_cols_cnt'] + 1):
        data["future_" + str(i)] = data.groupby(['delivery_area_id', 'date'], as_index=False) \
            .mean().shift(-i + 1).rolling(1).mean()['orders_cnt']
    return data


def get_holidays_features(data, df_holidays, dayoffs):
    logging.info('Getting holidays features')
    data['is_holiday'] = data.progress_apply(lambda x: x['date'] in df_holidays['ds'].values, axis=1)
    data['is_dayoff'] = data.progress_apply(lambda x: x['date'] in dayoffs, axis=1)
    data['is_weekend'] = data.progress_apply(lambda x: x['dayofweek'] in [5, 6], axis=1)
    return data


def get_celebrations():
    celebrations = pd.DataFrame(
        config['preprocessing']['celebrations'],
        columns=['date']
    )
    celebrations['date'] = pd.to_datetime(celebrations['date'])
    celebrations['doy'] = celebrations['date'].dt.dayofyear
    return celebrations


def get_days_before_nearest_day(row, calendar):
    current_doy = row['doy']
    days_after_current = calendar[calendar['doy'] >= current_doy]['doy']
    if len(days_after_current) > 0:
        days = min(days_after_current) - current_doy
        return days
    else:
        return 0


def get_days_after_nearest_day(row, calendar):
    current_doy = row['doy']
    days = abs(max(calendar[calendar['doy'] <= current_doy]['doy']) - current_doy)
    return days


def get_days_to_holidays(data, celebrations, df_holidays):
    logging.info('Getting days before and after holidays and celebrations')
    data['days_before_of_holiday'] = data.apply(lambda x: get_days_before_nearest_day(x, df_holidays), axis=1)
    data['days_before_celebrations'] = data.apply(lambda x: get_days_before_nearest_day(x, celebrations), axis=1)
    data['days_after_of_holiday'] = data.apply(lambda x: get_days_after_nearest_day(x, df_holidays), axis=1)
    data['days_after_celebrations'] = data.apply(lambda x: get_days_after_nearest_day(x, celebrations), axis=1)
    return data


def norm_by_1_week_median(data):
    logging.info('Normalize by median')
    cols_to_norm = ["prev_" + str(i) for i in range(1, config['preprocessing']['prev_cols_cnt'] + 1)] + \
                   ["future_" + str(i) for i in range(1, config['preprocessing']['next_cols_cnt'] + 1)]
    for col in cols_to_norm:
        data[col] = data[col] / (data['1_week_median_all'] + config['preprocessing']['eps'])
    return data


def get_aggregations(data, period, cols_range_left, cols_range_right):
    logging.info('Getting aggregations features')
    prev_cols = \
        ["prev_" + str(i) for i in range(1, config['preprocessing']['prev_cols_cnt'] + 1)][cols_range_left:cols_range_right]

    data[f'{period}_median'] = data[prev_cols].median(1)
    data[f'{period}_mean'] = data[prev_cols].mean(1)
    data[f'{period}_std'] = data[prev_cols].std(1)
    data[f'{period}_min'] = data[prev_cols].min(1)
    data[f'{period}_max'] = data[prev_cols].max(1)
    data[f'{period}_sum'] = data[prev_cols].sum(1)
    return data


def get_stat_changes(data, first_period='1_week', last_period='2_week'):
    for stat in ['median', 'mean', 'std', 'min', 'max', 'sum']:
        data[f'week_{stat}_change'] = (data[f'{first_period}_{stat}'] - data[f'{last_period}_{stat}']) \
                                      / (data[f'{last_period}_{stat}'] + config['preprocessing']['eps'])
    return data


def get_statistics(data):
    logging.info('Getting stats by weeks and month')
    data = get_aggregations(data, '1_week', cols_range_left=0, cols_range_right=7)
    data = get_aggregations(data, '2_week', cols_range_left=7, cols_range_right=14)
    data = get_aggregations(data, '1_month', cols_range_left=0, cols_range_right=30)

    data = get_stat_changes(data)
    return data


def get_features_for_chunk(data, df_holidays, dayoffs, celebrations):
    prev_cols = ["prev_" + str(i) for i in range(1, config['preprocessing']['prev_cols_cnt'] + 1)][:7]

    data = get_holidays_features(data, df_holidays, dayoffs)
    data = get_days_to_holidays(data, celebrations, df_holidays)
    data['1_week_median_all'] = data[prev_cols].median(1)
    data = norm_by_1_week_median(data)
    data = get_statistics(data)

    return data


def preprocess_orders(is_training=True):
    logging.info('Getting initial data')
    df_holidays = get_holidays()
    dayoffs = get_dayoffs()
    celebrations = get_celebrations()
    orders = get_orders()

    X_test_data = orders[orders['date'] > pd.to_datetime(config['preprocessing']['date_test_split'])]
    orders = group_by_date(orders)
    orders['dttm'] = orders['date']
    orders = split_date_time(orders)
    orders = get_prev_next_features(orders)

    logging.info('Data split generation')
    X_train, X_test = get_test_data(orders)
    X_train, X_val = timeseries_train_test_split(X_train, test_size=config['preprocessing']['test_size'])

    if is_training:
        logging.info('Getting features for all splits')
        X_train = get_features_for_chunk(X_train, df_holidays, dayoffs, celebrations)
        X_val = get_features_for_chunk(X_val, df_holidays, dayoffs, celebrations)
        X_test = get_features_for_chunk(X_test, df_holidays, dayoffs, celebrations)
        save_all_dfs(X_train, X_val, X_test, name_suffix='preprocessed')
        return X_train, X_val, X_test
    else:
        logging.info('Getting features for the test split')
        return get_features_for_chunk(X_test, df_holidays, dayoffs, celebrations), X_test_data
