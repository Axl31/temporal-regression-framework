"""Base classes and shared utilities for model wrappers."""

import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
# Linear and nonlinear regression models
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn import linear_model

# Support Vector Regression
from sklearn.svm import SVR

# Tree-based models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Polynomial regression tools
from sklearn.pipeline import make_pipeline

from itertools import combinations


REGRESSORS = [RandomForestRegressor(random_state=1),
              LinearRegression(),
              Ridge(alpha=1.0),
              linear_model.Lasso(alpha=0.1, max_iter=10000),
              ElasticNet(random_state=0),
              DecisionTreeRegressor(random_state=0),
              make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))]

LABELS = ['RandomForest',
         'Linear',
         'Ridge',
         'Lasso',
         'ElasticNet',
         'DecisionTree',
         'SVR']

# Function to generate all possible combinations of base regressors
def reg_comb():
    """Return a list of all possible combinations of base regressors."""
    regressor_list = REGRESSORS
    all_combos = []
    for r in range(1, len(regressor_list)+1):
        combos_r = combinations(regressor_list, r)
        for combo in combos_r:
            all_combos.append(list(combo))
    return all_combos

class BaseModelWrapper:
    """Lightweight base class to store data and a trained model."""
    def __init__(self, X_train, y_train, X_test, y_test):
        # Store training and test sets
        self.X = X_train
        self.y = y_train
        self.testX = X_test
        self.testY = y_test
        self.model = None

    def evaluate_rmse(self, preds):
        """Return RMSE between preds and stored testY."""
        return sqrt(mean_squared_error(self.testY, preds))

    def walk_forward_validation(self, retraining=True):
        # simple walk-forward implementation
        preds, actuals, errors = [], [], []
        X, y = self.X.copy(), self.y.copy()
        for i in range(len(self.testX) - 1):
            if retraining:
                self.model.fit(X, y)
            p = self.model.predict(self.testX[i:i + 1])
            preds.append(p)
            actuals.append(self.testY[i:i + 1])
            errors.append(sqrt(mean_squared_error(self.testY[i:i + 1], p)))
            if retraining:
                X = pd.concat([X, self.testX.iloc[i:i + 1]], ignore_index=True)
                y = pd.concat([y, pd.Series(self.testY.iloc[i:i + 1]).reset_index(drop=True)], ignore_index=True)
        return preds, actuals, errors
