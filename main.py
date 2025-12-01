from utils.data_loader import load_data
from models.boosting import Boosting
from models.bagging import Bagging
from models.voting import Voting
from models.stacking import Stacking
from evaluation.plotting import plot_predictions
import warnings
warnings.filterwarnings('ignore')

# --- Load dataset ---
X_train, X_test, y_train, y_test, mapping = load_data(
    'data/infy_stock.csv',
    date_col='Date',
    target_col='Close',
    columns_to_exclude=['Symbol', 'Series', 'Prev Close', 'Open', 'High', 'Low', 'Last', 'VWAP'],
    split=True,
    test_size=0.2
)

# --- Initialize models ---
models_list = [
    Boosting(X_train, y_train, X_test, y_test),
    Bagging(X_train, y_train, X_test, y_test),
    Voting(X_train, y_train, X_test, y_test),
    Stacking(X_train, y_train, X_test, y_test)
]

best_model = None
best_error = float('inf')

# --- Train & evaluate models ---
for model_obj in models_list:
    models, predictions, errors, times = model_obj.tuning()
    min_error_idx = errors.index(min(errors))
    if errors[min_error_idx] < best_error:
        best_error = errors[min_error_idx]
        best_model = model_obj
        best_predictions = predictions[min_error_idx]

# --- Plot best model predictions ---
plot_predictions(best_predictions, y_test, label_pred='Best Model', label_true='Actual')
