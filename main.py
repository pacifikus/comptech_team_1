import warnings

from src.inference import predict_on_chunk, get_hours_distribution
from src.preprocessing import preprocess_orders

warnings.simplefilter('ignore')


def get_predictions():
    test_features, test_data = preprocess_orders(is_training=False)
    predictions = predict_on_chunk(test_features.drop(['date', 'dttm', 'orders_cnt'], axis=1))
    hours_distribution = get_hours_distribution(test_features, test_data, predictions)
    return predictions, hours_distribution


if __name__ == '__main__':
    get_predictions()
