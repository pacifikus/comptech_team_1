import os
import pickle

import pandas as pd
import yaml

from src.lgbm_model import LGBMWeekModelTopFeatures


class CustomUnpickler(pickle.Unpickler):
    """Custom class to unpickle complicated objects"""

    def find_class(self, module, name):
        """Override base class method"""
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


def read_config(cfg_path):
    """Read yaml config.

    Args:
        cfg_path (str): path to config file.
    Returns:
        Parsed config.
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        return cfg


def get_test_data(X):
    """Get test slice from initial data.

    Args:
        X (DataFrame): full dataset.
    Returns:
        Train, test splits of dataset.
    """
    date_test_split = pd.to_datetime(config['preprocessing']['date_test_split'])
    X_train = X[X['date'] <= date_test_split]
    X_test = X[X['date'] > date_test_split]
    return X_train, X_test


def timeseries_train_test_split(X, y=None, test_size=0.3):
    """Consecutive train/test split implementation for time series.

    Args:
        X (DataFrame): features part of the filename.
        y (DataFrame): target part of the dataset.
        test_size (float): proportion of the dataset to include in the test split.
    Returns:
        Train, test splits of dataset.
    """
    X = X.sort_values(by='date')
    test_index = int(len(X) * (1 - test_size))
    X_train = X.iloc[:test_index]
    X_test = X.iloc[test_index:]
    if y is not None:
        y_train = y.iloc[:test_index]
        y_test = y.iloc[test_index:]
        return X_train, X_test, y_train, y_test
    return X_train, X_test


def save_all_dfs(X_train, X_val, X_test, name_suffix):
    """Save all dataset splits.

    Args:
        X_train (DataFrame): train split of the dataset.
        X_val (str): valid split of the dataset.
        X_test (str): test split of the dataset.
        name_suffix (str): last part of the filename.
    """
    X_train.to_csv(f'X_train_{name_suffix}.csv', index=False)
    X_val.to_csv(f'X_val_{name_suffix}.csv', index=False)
    X_test.to_csv(f'X_test_{name_suffix}.csv', index=False)
    print('Данные сохранены')


def load_all(name_suffix):
    """Load all dataset splits saved earlier.

    Args:
        name_suffix (str): last part of the filename.
    Returns:
        Loaded train, valid, and test splits of dataset.
    """
    X_train = pd.read_csv(f'X_train_{name_suffix}.csv')
    X_val = pd.read_csv(f'X_val_{name_suffix}.csv')
    X_test = pd.read_csv(f'X_test_{name_suffix}.csv')
    return X_train, X_val, X_test


def load_model():
    """Load serialized model from file.

    Returns:
        Loaded model.
    """
    with open(config['model_path'], "rb") as f:
        model = CustomUnpickler(f).load()
        return model


config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
config = read_config(config_path)
