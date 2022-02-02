import lightgbm as lgb
import pandas as pd
import tqdm
from sklearn.base import BaseEstimator


class LGBMWeekModelTopFeatures(BaseEstimator):
    """Custom LGBM model based on a set of LGBM regressors trained on top features."""

    def __init__(self):
        self.models = []

    def fit(self, data, params_list, f_importances):
        """Learn from data on top features with optimized hyperparameters.

        Args:
            data (DataFrame): data to learn from.
            params_list (list of dict): list of optimized hyperparameters for each base model.
            f_importances (list of DataFrame): computed feature importances for each model.
        Returns:
            Fitted model.
        """
        self.f_importances = f_importances
        for i in tqdm(range(7)):
            params = params_list[i]
            train_data = data.dropna(subset=[f'future_{i + 1}'])
            train_data = train_data.drop(train_data[train_data[f'future_{i + 1}'] == 0].index)
            need_columns = self.f_importances[i].sort_values(by="importance", ascending=False)[0:30]['cols'].values
            train_data = lgb.Dataset(
                train_data[need_columns],
                label=train_data[f'future_{i + 1}']
            )
            model = lgb.train(
                params=params,
                train_set=train_data,
                valid_sets=[train_data],
                num_boost_round=100,
                verbose_eval=1000
            )
            self.models.append(model)
        return self

    def predict(self, data):
        """Predict orders count for each unseen day.

        Args:
            data (DataFrame): data to predict.
        Returns:
            DataFrame with predicted orders count.
        """
        preds_df = pd.DataFrame()
        for i in range(7):
            need_columns = self.f_importances[i].sort_values(by="importance", ascending=False)[0:30]['cols'].values
            preds_df['pred_' + str(i)] = self.models[i].predict(data[need_columns])
        return preds_df
