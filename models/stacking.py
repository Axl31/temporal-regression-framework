"""
Stacking wrapper using sklearn StackingRegressor over combinations of base learners.
"""

from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from .base_models import BaseModelWrapper, LABELS, reg_comb
import time
import numpy as np

class Stacking(BaseModelWrapper):
    """
    Stacking ensemble using all possible combinations of base regressors.
    """

    def tuning(self):
        """
        Fit StackingRegressor for all combinations of base learners, compute RMSE, and track time.

        Returns:
            models (list): list of fitted StackingRegressor models
            predictions (list): list of predictions on test set
            errors (list): RMSE for each ensemble
            times (list): computation time for each ensemble
        """
        all_combinations = reg_comb()
        models = []
        predictions = []
        errors = []
        times = []

        for idx, combo in enumerate(all_combinations):
            start_time = time.time()

            # Pair regressor labels with combo
            combo_labels = LABELS[:len(combo)]
            ensemble_estimators = list(zip(combo_labels, combo))

            stacking_model = StackingRegressor(
                estimators=ensemble_estimators,
                final_estimator=RandomForestRegressor(n_estimators=len(combo), random_state=42)
            )
            stacking_model.fit(self.X, self.y)

            # Predict and compute RMSE
            pred = stacking_model.predict(self.testX)
            rmse = self.evaluate_rmse(pred)

            predictions.append(pred)
            errors.append(rmse)
            models.append(stacking_model)

            elapsed_time = time.time() - start_time
            times.append(elapsed_time)

            print(f"Time taken for ensemble {idx+1} (size {len(combo)}): {elapsed_time:.6f} sec, RMSE: {rmse:.6f}")

        # Select the best ensemble
        best_idx = int(np.argmin(errors))
        self.model = models[best_idx]
        print(f"The best stacking ensemble is combination {best_idx+1} with RMSE {errors[best_idx]:.6f}")

        return models, predictions, errors, times
