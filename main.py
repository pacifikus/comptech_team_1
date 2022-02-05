import argparse
import os
import warnings

import pandas as pd

from src.inference import predict_on_chunk, get_hours_distribution, get_courier_distribution, get_shifts
from src.preprocessing import preprocess_orders
from src.utils import read_config

warnings.simplefilter('ignore')

config_path = os.path.join(os.path.dirname(__file__), 'config/config.yaml')
config = read_config(config_path)


def get_predictions(date, orders_path, delays_path):
    """Predict hours distribution for a week from the current day.

    Args:
        date (str): date to predict from.
        orders_path (str): path to file with orders data.
        delays_path (str): path to file with delays data.
    Returns:
        Dataframe with computed hours distribution.
    """
    date = pd.to_datetime(date)
    test_features, percentage_by_hours, courier_median = preprocess_orders(orders_path, delays_path, date, is_training=False)
    predictions = predict_on_chunk(test_features)
    hours_distribution = get_hours_distribution(predictions, percentage_by_hours, date)
    hours_distribution['courier_cnt'] = get_courier_distribution(hours_distribution, courier_median)
    shifts = get_shifts(hours_distribution)
    return hours_distribution, shifts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        dest="date",
        type=str,
        default='2021-11-18',
        help="Current date to predict from"
    )
    parser.add_argument(
        "--orders",
        dest="orders_path",
        type=str,
        default=config['data']['orders_path'],
        help="Path to orders data "
    )
    parser.add_argument(
        "--delays",
        dest="delays_path",
        type=str,
        default=config['data']['delays_path'],
        help="Path to delays data "
    )

    args = parser.parse_args()
    orders_distribution, shifts = get_predictions(
        args.date,
        args.orders_path,
        args.delays_path,
    )
    orders_distribution.to_csv(f'orders_distribution_from_{args.date}.csv', index=False)
    shifts.to_csv(f'shifts_from_{args.date}.csv', index=False)
