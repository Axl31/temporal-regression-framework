import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from .stepwise import stepwise_selection
from .base_models import BaseModelWrapper

class Regressors(BaseModelWrapper):
    """
    Train and evaluate multiple regression models including
    Linear, Polynomial, Ridge, Lasso, ElasticNet, Decision Tree, SVR, and Stepwise regression.
    """

    def tuning(self):
        """
        Train all regression models, evaluate them on the test set, and select the best one based on RMSE.

        Returns:
            labels (list): Names of the models
            predictions (list): Predictions of each model
            errors (list): RMSE of each model
        """
        labels = ['Linear', 'Polynomial', 'Ridge', 'Lasso', 'Elastic', 'Tree', 'SVR', 'Stepwise']

        # 1. Linear Regression
        linear = LinearRegression()
        linear.fit(self.X, self.y)

        # 2. Polynomial Regression (interaction only)
        poly = PolynomialFeatures(interaction_only=True)
        X_poly = poly.fit_transform(self.X)
        polynomial = LinearRegression()
        polynomial.fit(X_poly, self.y)

        # 3. Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(self.X, self.y)

        # 4. Lasso Regression
        lasso = Lasso(alpha=0.1)
        lasso.fit(self.X, self.y)

        # 5. ElasticNet Regression
        elastic = ElasticNet(random_state=0)
        elastic.fit(self.X, self.y)

        # 6. Decision Tree
        tree = DecisionTreeRegressor(random_state=0)
        tree.fit(self.X, self.y)

        # 7. Support Vector Regression (with StandardScaler pipeline)
        svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        svr.fit(self.X, self.y)

        # 8. Stepwise Regression (with fallback in case of error)
        try:
            pred_stepwise, selected_features = stepwise_selection(self.X, self.y, self.testX, self.testY)
            print("Selected features (Stepwise Regression):", selected_features)
        except Exception as e:
            print("Stepwise Regression Error:", e)
            pred_stepwise = svr.predict(self.testX)

        # Generate predictions
        predictions = [
            linear.predict(self.testX),
            polynomial.predict(poly.transform(self.testX)),
            ridge.predict(self.testX),
            lasso.predict(self.testX),
            elastic.predict(self.testX),
            tree.predict(self.testX),
            svr.predict(self.testX),
            np.array(pred_stepwise)
        ]

        models = [linear, polynomial, ridge, lasso, elastic, tree, svr, linear]  # store best later

        # Compute RMSE
        errors = [self.evaluate_rmse(p) for p in predictions]

        # Select the best model
        best_index = np.argmin(errors)
        self.model = models[best_index]
        print("The best model is:", labels[best_index], "with RMSE:", errors[best_index])

        return labels, predictions, errors
