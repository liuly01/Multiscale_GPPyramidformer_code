"""
Evaluation metrics for daily GPP estimation.

Only the four primary metrics used for model comparison are included here:
R2, RMSE, MAE, and KGE.
"""

import numpy as np


def _prepare_arrays(y_true, y_pred):
    """Return finite paired observations and predictions."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def r2_score_np(y_true, y_pred):
    """Coefficient of determination."""
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    if y_true.size == 0:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def rmse_score(y_true, y_pred):
    """Root mean squared error."""
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    if y_true.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_score(y_true, y_pred):
    """Mean absolute error."""
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    if y_true.size == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))


def kge_score(y_true, y_pred):
    """Kling-Gupta efficiency."""
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    if y_true.size < 2:
        return np.nan
    obs_std = np.std(y_true)
    pred_std = np.std(y_pred)
    obs_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    if obs_std == 0 or obs_mean == 0:
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = pred_std / obs_std
    beta = pred_mean / obs_mean
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def regression_metrics(y_true, y_pred):
    """Compute the four primary evaluation metrics."""
    return {
        "R2": r2_score_np(y_true, y_pred),
        "RMSE": rmse_score(y_true, y_pred),
        "MAE": mae_score(y_true, y_pred),
        "KGE": kge_score(y_true, y_pred),
    }
