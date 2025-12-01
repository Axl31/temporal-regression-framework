import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(pred, y_test, label_pred='Prediction', label_true='Actual', fontsize=18):
    """
    Plot predicted vs actual values.

    Args:
        pred (array-like): Predictions
        y (array-like): True values
        label_pred (str): Label for prediction curve
        label_true (str): Label for true curve
        fontsize (int): Font size for labels and ticks
    """
    y = np.array(y_test)
    plt.figure(figsize=(20, 10))
    plt.plot(pred, label=label_pred, linewidth=3)
    plt.plot(y, label=label_true, linewidth=3)
    plt.xlabel('Time', fontsize=fontsize)
    plt.ylabel('Value', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.show()
