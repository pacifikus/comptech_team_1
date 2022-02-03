import argparse
import os
import warnings

import numpy as np

from src.inference import predict_on_chunk, get_hours_distribution
from src.preprocessing import preprocess_orders
from src.utils import read_config

warnings.simplefilter('ignore')

config_path = os.path.join(os.path.dirname(__file__), 'config/config.yaml')
config = read_config(config_path)


def get_predictions(date):
    """Predict hours distribution for a week from the current day.

    Args:
        date (str): date to predict from.
    Returns:
        Dataframe with computed hours distribution.
    """
    test_features = preprocess_orders(is_training=False)
    predictions = predict_on_chunk(test_features)
    hours_distribution = get_hours_distribution(predictions[predictions['date'] == date])
    return hours_distribution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        dest="date",
        type=str,
        default='2021-11-18',
        help="Current date to predict from"
    )

    args = parser.parse_args()
    result = get_predictions(args.date)
    result.to_csv(f'orders_by_hours_from_{args.date}.csv', index=False)
    result = (result * config['courier_per_order']).apply(np.ceil)
    result.to_csv(f'couriers_by_hours_from_{args.date}.csv', index=False)
