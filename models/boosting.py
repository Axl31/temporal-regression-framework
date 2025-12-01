"""
Boosting wrapper using AdaBoostRegressor over different base learners.
"""

from sklearn.ensemble import AdaBoostRegressor
from .base_models import BaseModelWrapper, REGRESSORS, LABELS
import time
import numpy as np

class Boosting(BaseModelWrapper):
    """
    Boosting ensemble using AdaBoostRegressor on multiple base learners.
    Inherits from BaseModelWrapper.
    """

    def tuning(self):
        """
        Fit AdaBoostRegressor for each base learner, compute RMSE, and track time.

        Returns:
            models (list): list of fitted AdaBoost models
            predictions (list): list of predictions on test set
            errors (list): RMSE for each model
            times (list): computation time for each model
        """
        models = []
        predictions = []
        errors = []
        times = []

        for i, base_model in enumerate(REGRESSORS):
            start_time = time.time()

            # Initialize AdaBoost with base estimator
            model = AdaBoostRegressor(estimator=base_model, random_state=1)
            model.fit(self.X, self.y)

            # Predict on test set
            pred = model.predict(self.testX)

            # Record predictions and errors
            predictions.append(pred)
            rmse = self.evaluate_rmse(pred)
            errors.append(rmse)
            models.append(model)

            elapsed_time = time.time() - start_time
            times.append(elapsed_time)

            print(f"Time taken for {LABELS[i]}: {elapsed_time:.6f} seconds, RMSE: {rmse:.6f}")

        # Select best model
        best_idx = int(np.argmin(errors))
        self.model = models[best_idx]
        print(f"The best model is: {LABELS[best_idx]} with RMSE {errors[best_idx]:.6f}")

        return models, predictions, errors, times
