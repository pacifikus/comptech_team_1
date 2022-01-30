import os

from sklearn.metrics import mean_absolute_percentage_error

from src.utils import load_model
from src.utils import read_config

model = load_model()
config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
config = read_config(config_path)


def predict_on_chunk(data, print_mape=False):
    res = model.predict(data)
    for i in range(7):
        res['pred_' + str(i)] = res['pred_' + str(i)] * data['1_week_median_all'].values
        if print_mape:
            print(
                mean_absolute_percentage_error(
                    data[f'future_{i + 1}'] * data['1_week_median_all'].values,
                    res['pred_' + str(i)]
                )
            )
    return res


def get_hours_distribution(test_features, test_data, predictions):
    test_features[[f'pred_{i}' for i in range(7)]] = predictions.to_numpy()
    percentage_distribution = config['percentage_by_hours']
    # TODO: распределение по часам
    return []
