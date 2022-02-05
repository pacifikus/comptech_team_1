import logging
import os

import holidays
import pandas as pd
from tqdm import tqdm

from src.utils import read_config, get_test_data, timeseries_train_test_split, save_all_dfs
from src.bootstrap import get_bootstrap_ci

logging.basicConfig(level=logging.DEBUG)

config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
config = read_config(config_path)
tqdm.pandas()


def get_source_data(data_path, date_col='date'):
    """Get source data from .csv file.

    Args:
        data_path (str): path to source data. Defaults path stored in config file.
        date_col (str): name of the date column.
    Returns:
        DataFrame with orders information.
    """
    return pd.read_csv(data_path, parse_dates=[date_col])


def get_dayoffs():
    """Get predefined dayoffs dates from .txt file. Filepath is stored in config.

    Returns:
        List with dayoffs dates.
    """
    with open(config['data']['dayoffs_path']) as f:
        dayoffs = f.read().split(',')
    return [pd.to_datetime(day) for day in dayoffs]


def get_celebrations():
    """Get predefined celebrations dates from .txt file.Filepath is stored in config.

    Returns:
        DataFrame with celebrations dates.
    """
    with open(config['data']['celebrations_path']) as f:
        celebrations = f.read().split(',')

    celebrations = pd.DataFrame(celebrations, columns=['date'])
    celebrations['date'] = pd.to_datetime(celebrations['date'])
    celebrations['doy'] = celebrations['date'].dt.dayofyear
    return celebrations


def get_holidays():
    """Get holidays dates with holidays package.

    Returns:
        DataFrame with holidays dates.
    """
    holidays_dict = holidays.RU(years=2021)
    df_holidays = pd.DataFrame.from_dict(holidays_dict, orient='index').reset_index()
    df_holidays.columns = ['ds', 'holiday']
    df_holidays['ds'] = pd.to_datetime(df_holidays['ds'])
    df_holidays['doy'] = df_holidays['ds'].dt.dayofyear
    return df_holidays


def split_date_time(data):
    """Get features from date split. Computed features: dayofweek, dayofyear.

    Args:
        data (DataFrame): data to getting features.
    Returns:
        Initial data with computed features.
    """
    logging.info('Getting date features')
    data['dttm'] = pd.to_datetime(data['dttm'])
    data['dayofweek'] = data['dttm'].dt.dayofweek
    data['doy'] = data['dttm'].dt.dayofyear
    return data


def group_by_date(data):
    """Get orders count for grouped by delivery_area_id and date rows.

    Args:
        data (DataFrame): data to getting features.
    Returns:
        Initial data with computed features.
    """
    data['date'] = data['date'].dt.date
    data = data.groupby(['delivery_area_id', 'date']).sum()['orders_cnt'].reset_index()
    return data


def get_prev_next_features(data):
    """Get sliding window previous and future features for the data grouped by area.

    Args:
        data (DataFrame): data to getting features.
    Returns:
        Initial data with computed features.
    """
    logging.info('Getting base features from past and future')
    data = data.sort_values(by=['delivery_area_id', 'date']).reset_index(drop=True)

    for i in range(1, config['preprocessing']['prev_cols_cnt'] + 1):
        data['prev_' + str(i)] = data.groupby(['delivery_area_id'])["orders_cnt"] \
            .shift(i).rolling(1).sum()

    data["future_1"] = data['orders_cnt'].copy()
    for i in range(2, config['preprocessing']['next_cols_cnt'] + 1):
        data["future_" + str(i)] = data.groupby(['delivery_area_id'])['orders_cnt'] \
            .shift(-i + 1).rolling(1).mean()
    return data


def get_holidays_features(data, df_holidays, dayoffs):
    """Define if dates are holidays, dayoffs, weekends.

    Args:
        data (DataFrame): data to getting features.
        df_holidays (DatFrame): holidays dates.
        dayoffs (DataFrame): dayoffs dates.
    Returns:
        Initial data with computed features.
    """
    logging.info('Getting holidays features')
    data['is_holiday'] = data['date'].isin(df_holidays['ds'].values)
    data['is_dayoff'] = data['date'].isin(dayoffs)
    data['is_weekend'] = data['dayofweek'].isin({5, 6})
    return data


def get_days_before_nearest_day(row, calendar):
    """Get days count before nearest day from the gotten calendar.

    Args:
        row (Series): single pd.DataFrame row with current day.
        calendar (DataFrame): set of dates to compute days count.
    Returns:
        Days count from nearest day to current day.
    """
    current_doy = row['doy']
    days_after_current = calendar[calendar['doy'] >= current_doy]['doy']
    if len(days_after_current) > 0:
        days = min(days_after_current) - current_doy
        return days
    else:
        return 0


def get_days_after_nearest_day(row, calendar):
    """Get days count after nearest day from the gotten calendar.

    Args:
        row (Series): single pd.DataFrame row with current day.
        calendar (DataFrame): set of dates to compute days count.
    Returns:
        Days count from nearest day to current day.
    """
    current_doy = row['doy']
    days = abs(max(calendar[calendar['doy'] <= current_doy]['doy']) - current_doy)
    return days


def get_days_to_holidays(data, celebrations, df_holidays):
    """Get days before and after celebrations and holidays for each date.

    Args:
        data (DataFrame): data to getting features.
        celebrations (DataFrame): celebrations dates.
        df_holidays (DataFrame): holidays dates.
    Returns:
        Initial data with computed features.
    """
    logging.info('Getting days before and after holidays and celebrations')
    data['days_before_of_holiday'] = data.apply(lambda x: get_days_before_nearest_day(x, df_holidays), axis=1)
    data['days_before_celebrations'] = data.apply(lambda x: get_days_before_nearest_day(x, celebrations), axis=1)
    data['days_after_of_holiday'] = data.apply(lambda x: get_days_after_nearest_day(x, df_holidays), axis=1)
    data['days_after_celebrations'] = data.apply(lambda x: get_days_after_nearest_day(x, celebrations), axis=1)
    return data


def norm_by_1_week_median(data):
    """Get prev_ and future_ features normalized by a week median.

    Args:
        data (DataFrame): data to getting features.
    Returns:
        Initial data with normalized features.
    """
    logging.info('Normalize by median')
    prev_cols = ["prev_" + str(i) for i in range(1, config['preprocessing']['prev_cols_cnt'] + 1)][:7]
    data['1_week_median_all'] = data[prev_cols].median(1)

    cols_to_norm = ["prev_" + str(i) for i in range(1, config['preprocessing']['prev_cols_cnt'] + 1)] + \
                   ["future_" + str(i) for i in range(1, config['preprocessing']['next_cols_cnt'] + 1)]
    for col in cols_to_norm:
        data[col] = data[col] / (data['1_week_median_all'] + config['preprocessing']['eps'])
    return data


def get_aggregations(data, period, cols_range_left, cols_range_right):
    """Get aggregated statistics by period. Aggregation functions: median, mean, std, min, max, sum.

    Args:
        data (DataFrame): data to getting features.
        period (str): period and column prefix for new feature columns.
        cols_range_left (int): left index of features range to extract.
        cols_range_right (int):  right index of features range to extract.
    Returns:
        Initial data with computed aggregated statistics.
    """
    logging.info('Getting aggregations features')
    prev_cols = \
        ["prev_" + str(i) for i in range(1, config['preprocessing']['prev_cols_cnt'] + 1)] \
            [cols_range_left:cols_range_right]

    data[f'{period}_median'] = data[prev_cols].median(1)
    data[f'{period}_mean'] = data[prev_cols].mean(1)
    data[f'{period}_std'] = data[prev_cols].std(1)
    data[f'{period}_min'] = data[prev_cols].min(1)
    data[f'{period}_max'] = data[prev_cols].max(1)
    data[f'{period}_sum'] = data[prev_cols].sum(1)
    return data


def get_stat_changes(data, first_period='1_week', last_period='2_week'):
    """Get changes between statistics by two periods.

    Args:
        data (DataFrame): data to getting features.
        first_period (str): column name of the first period.
        last_period (str): column name of the second period.
    Returns:
        Initial data with computed statistics changes.
    """
    for stat in ['median', 'mean', 'std', 'min', 'max', 'sum']:
        data[f'week_{stat}_change'] = (data[f'{first_period}_{stat}'] - data[f'{last_period}_{stat}']) \
                                      / (data[f'{last_period}_{stat}'] + config['preprocessing']['eps'])
    return data


def get_statistics(data):
    """Get aggregated statistics from the data by 1-2 weeks, 1-month, and changes between periods.

    Args:
        data (DataFrame): data to getting features.
    Returns:
        Initial data with computed statistics.
    """
    logging.info('Getting stats by weeks and month')
    data = get_aggregations(data, '1_week', cols_range_left=0, cols_range_right=7)
    data = get_aggregations(data, '2_week', cols_range_left=7, cols_range_right=14)
    data = get_aggregations(data, '1_month', cols_range_left=0, cols_range_right=30)

    data = get_stat_changes(data)
    return data


def extract_features(orders):
    """Run features extraction from a given data.

    Args:
        orders (pd.DataFrame): data to extract features.
    Returns:
        Set of features from the data.
    """
    logging.info('Start features extraction')
    df_holidays = get_holidays()
    dayoffs = get_dayoffs()
    celebrations = get_celebrations()

    orders = group_by_date(orders)
    orders['dttm'] = orders['date']
    orders = split_date_time(orders)
    orders = get_prev_next_features(orders)
    orders = get_holidays_features(orders, df_holidays, dayoffs)
    orders = get_days_to_holidays(orders, celebrations, df_holidays)
    orders = norm_by_1_week_median(orders)
    orders = get_statistics(orders)
    return orders


def compute_percentage_by_hours(orders):
    """Orders distribution by hours in percentage.

    Args:
        orders (pd.DataFrame): orders data.
    Returns:
        Dictionary with percentage of orders for the each hour.
    """
    orders['hour'] = orders['date'].dt.hour
    orders = orders[~orders['hour'].isin([7, 8, 9, 21, 22, 23])]
    grouped_by_hour = orders.groupby('hour').mean()['orders_cnt']
    grouped_by_hour = grouped_by_hour.apply(lambda x: 100 * x / float(grouped_by_hour.sum()))
    return grouped_by_hour.to_dict()


def compute_delay_median(delays, orders):
    """Run features extraction from a given data.

    Args:
        delays (pd.DataFrame): delays data.
        orders (pd.DataFrame): orders data.
    Returns:
        Median courier per order value optimizing delay.
    """
    logging.info('Compute courier median')
    delays.columns = ['delivery_area_id', 'date', 'partners_cnt', 'delay_rate']
    df = pd.merge(orders, delays, on=['date', 'delivery_area_id'], how='inner')
    df['order_p_partner'] = df['orders_cnt'] / df['partners_cnt']
    delay_threshold = config['delay_threshold']
    _, median = get_bootstrap_ci(df[df['delay_rate'] < delay_threshold]['order_p_partner'])
    return round(median, 3)


def preprocess_orders(orders_path, delays_path, date, is_training=True):
    """Run preprocessing pipeline for the data.

    Args:
        orders_path (str): path to file with orders data.
        delays_path (str): path to file with delays data.
        date (pd.DateTime): date to predict from.
        is_training (boolean): flag of training mode. If True, returns train, valid, test splits.
        If False, returns test split only. Defaults to False.
    Returns:
        Splits of dataset depending on the mode.
    """
    logging.info('Getting initial data')
    orders = get_source_data(orders_path)
    orders['date'] = pd.to_datetime(orders['date'])

    if not is_training:
        cut_date = date - pd.DateOffset(days=config['days_to_feature_calc'] + 1)
        orders = orders[orders['date'] >= cut_date]
        orders = orders[orders['date'] <= date + pd.DateOffset(days=1)]

        delays = get_source_data(delays_path, 'dttm')
        delay_median = compute_delay_median(delays, orders)

        percentages = compute_percentage_by_hours(orders)
        orders = extract_features(orders)
        orders = orders[orders['date'] == date]
        return orders, percentages, delay_median

    orders = extract_features(orders)
    logging.info('Data split generation')
    X_train, X_test = get_test_data(orders)
    X_train, X_val = timeseries_train_test_split(X_train, test_size=config['preprocessing']['test_size'])
    save_all_dfs(X_train, X_val, X_test, name_suffix='preprocessed')
    return X_train, X_val, X_test
