import warnings

from src.inference import predict_on_chunk, get_hours_distribution
from src.preprocessing import preprocess_orders, get_orders

warnings.simplefilter('ignore')


def get_predictions(date):
    test_features = preprocess_orders(is_training=False)
    predictions = predict_on_chunk(test_features)
    hours_distribution = get_hours_distribution(predictions[predictions['date'] == date])
    return hours_distribution


if __name__ == '__main__':
    get_predictions('2021-11-18')
