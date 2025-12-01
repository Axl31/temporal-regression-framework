import pandas as pd
import statsmodels.api as sm

def stepwise_selection(X_train, y_train, X_test, y_test,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out=0.05,
                       verbose=True):
    """
    Perform a stepwise feature selection using forward-backward approach
    based on p-values from OLS regression.

    Args:
        X_train (DataFrame): Training features
        y_train (Series/DataFrame): Training target
        X_test (DataFrame): Test features
        y_test (Series/DataFrame): Test target
        initial_list (list): Initial list of features to include
        threshold_in (float): P-value threshold for adding features
        threshold_out (float): P-value threshold for removing features
        verbose (bool): If True, print progress

    Returns:
        predictions (array): Predictions on test set using selected features
        included (list): List of selected features
    """
    included = list(initial_list)
    while True:
        changed = False

        # --- Forward step ---
        excluded = list(set(X_train.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y_train, sm.add_constant(X_train[included + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add {best_feature:30} with p-value {best_pval:.6f}')

        # --- Backward step ---
        model = sm.OLS(y_train, sm.add_constant(X_train[included])).fit()
        pvalues = model.pvalues.iloc[1:]  # exclude intercept
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f'Drop {worst_feature:30} with p-value {worst_pval:.6f}')

        if not changed:
            break

    # --- Fit final model on selected features ---
    X_train_selected = X_train[included]
    X_test_selected = X_test[included]
    final_model = sm.OLS(y_train, sm.add_constant(X_train_selected)).fit()
    predictions = final_model.predict(sm.add_constant(X_test_selected))

    return predictions, included
