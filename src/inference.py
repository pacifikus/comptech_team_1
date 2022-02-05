import os

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from src.utils import load_model
from src.utils import read_config
from src.linprog import get_best_shifts

model = load_model()
config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
config = read_config(config_path)


def predict_on_chunk(data, print_mape=False):
    """Predict orders count for each day from the current day.

    Args:
        data (DataFrame): precomputed features for the current day.
        print_mape (boolean): flag to calculate MAPE metric if needed. Defaults to False.
    Returns:
        Dataframe with prediction.
    """
    if print_mape:
        data = data.dropna()
        res = model.predict(data)
        for i in range(7):
            res['pred_' + str(i)] = res['pred_' + str(i)] * data['1_week_median_all'].values
            print(
                mean_absolute_percentage_error(
                    data[f'future_{i + 1}'] * data['1_week_median_all'].values,
                    res['pred_' + str(i)]
                )
            )
    else:
        res = model.predict(data)
        next_days = config['preprocessing']['next_cols_cnt']
        for i in range(next_days):
            data['pred_' + str(i)] = (res['pred_' + str(i)] * data['1_week_median_all'].values).apply(np.ceil).values
    return data[['delivery_area_id', 'date'] + ['pred_' + str(i) for i in range(next_days)]]


def get_hours_distribution(predictions, percentage_distribution, date):
    """Compute orders distribution by hours for each day.

    Args:
        predictions (DataFrame): precomputed orders count distribution for each day.
        percentage_distribution (dict): orders distribution by hours in percentage.
        date: prediction start date.
    Returns:
        Dataframe with computed hours distribution.
    """
    hours = [item for item in list(percentage_distribution.keys())]
    result = []
    for i, row in predictions.iterrows():
        for day in range(7):
            col = f'pred_{day}'
            for hour in hours:
                new_row = {
                    'delivery_area_id': row['delivery_area_id'],
                    'date': date + pd.Timedelta(days=day + 1, hours=hour),
                    'n_orders': np.ceil(row[col] * percentage_distribution[hour] / 100)
                }
                result.append(new_row)
    return pd.DataFrame(result)


def get_courier_distribution(hour_distribution, courier_median):
    """Compute orders distribution by hours for each day.

    Args:
        hour_distribution (DataFrame): orders distribution by hours.
        courier_median (float): median courier by order rate value.
    Returns:
        Dataframe with computed courier by hours distribution.
    """
    return (hour_distribution['n_orders'] * courier_median).apply(np.ceil)


def get_shifts(hours_distribution):
    """Compute courier shifts for each delivery area for each day.

    Args:
        hours_distribution (DataFrame): couriers distribution by hours.
    Returns:
        Dataframe with computed couriers shifts.
    """
    shifts = []
    hours_distribution['day'] = hours_distribution['date'].dt.date
    for day in hours_distribution['day'].unique():
        for area in hours_distribution['delivery_area_id'].unique():
            courier_distribution = hours_distribution[
                (hours_distribution['day'] == day) & (hours_distribution['delivery_area_id'] == area)
            ]['courier_cnt'].values
            shifts.append(get_best_shifts(courier_distribution, day, area))
    shifts = pd.concat(shifts)
    shifts['shift_id'] = shifts.reset_index().index
    return shifts
