import os

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from src.utils import load_model
from src.utils import read_config

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
        for i in range(7):
            res['pred_' + str(i)] = res['pred_' + str(i)] * data['1_week_median_all'].values
    res['date'] = data['date']
    return res


def get_hours_distribution(predictions):
    """Compute orders distribution by hours for each day.

    Args:
        predictions (DataFrame): precomputed orders count distribution for each day.
    Returns:
        Dataframe with computed hours distribution.
    """
    percentage_distribution = config['percentage_by_hours']
    hours = [item for item in list(percentage_distribution.keys())]
    result = []
    for col in [f'pred_{i}' for i in range(7)]:
        temp_result = []
        for hour in hours:
            temp_result.append(round(predictions[col].values[0] * percentage_distribution[hour] / 100))
        result.append(temp_result)
    return pd.DataFrame(result, columns=hours)
