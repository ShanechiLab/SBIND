import copy
import numpy as np

import torch

import sklearn.metrics as metrics
from skimage.metrics import structural_similarity as ssim




def eval_prediction(true_value, prediction, measure, missing_marker=None, mask=None):
    """
    Evaluates the prediction of data.
    """

    is_categorical = (true_value.dtype == 'int' or true_value.dtype == 'bool')

    if mask is not None:
        mask = mask.squeeze().astype(bool)
        true_value = true_value[mask]
        prediction = prediction[mask]

    if missing_marker is not None:
        if np.isnan(missing_marker):
            is_ok = np.all(~np.isnan(prediction), axis=1)
        else:
            is_ok = np.all(prediction != missing_marker, axis=1)

        true_value = copy.deepcopy(true_value)[is_ok, :]
        prediction = copy.deepcopy(prediction)[is_ok, :]

    if true_value.ndim > 2:
        n_samples, *image_shape = true_value.shape
        n_dims = np.prod(image_shape)

        true_value_flat = true_value.reshape(n_samples, n_dims)
        prediction_flat = prediction.reshape(n_samples, n_dims)
    else:
        n_samples, n_dims = true_value.shape
        true_value_flat = true_value
        prediction_flat = prediction

    if n_samples == 0:
        return np.nan * np.ones((n_dims,))

    if is_categorical:
        perf = np.zeros(n_dims)

        if measure in ['angular_error', 'accuracy_8', 'cm_8']:
            label_to_xy = {
                0: (1, 1),
                1: (0, 1),
                2: (2, 1),
                3: (1, 0),
                4: (2, 0),
                5: (1, 2),
                6: (0, 2),
                7: (2, 2),
                8: (0, 0)
            }
            xy_to_label = {v: k for k, v in label_to_xy.items()}

            if len(np.unique(true_value_flat)) <= 2 or true_value_flat.shape[1] != 2:
                return np.nan

            else:
                if measure == 'accuracy_8':
                    pred_dim_updown = np.argmax(prediction_flat[:, 1, :], axis=1)
                    pred_dim_rightleft = np.argmax(prediction_flat[:, 0, :], axis=1)

                    true_dim_updown = true_value_flat[:, 1]
                    true_dim_rightleft = true_value_flat[:, 0]

                    pred_dim_combined = (pred_dim_updown - 1) * 3 + pred_dim_rightleft

                    true_dim_combined = (true_dim_updown - 1) * 3 + true_dim_rightleft

                    return metrics.accuracy_score(true_dim_combined, pred_dim_combined)

                if measure == 'cm_8':
                    pred_dim_updown = np.argmax(prediction_flat[:, 1, :], axis=1)
                    pred_dim_rightleft = np.argmax(prediction_flat[:, 0, :], axis=1)

                    true_dim_updown = true_value_flat[:, 1]
                    true_dim_rightleft = true_value_flat[:, 0]

                    pred_dim_combined = np.array([xy_to_label[(x, y)] for x, y in zip(pred_dim_rightleft, pred_dim_updown)])

                    true_dim_combined = np.array([xy_to_label[(x, y)] for x, y in zip(true_dim_rightleft, true_dim_updown)])

                    return metrics.confusion_matrix(true_dim_combined, pred_dim_combined)

                else:
                    prediction_mapped = np.where(np.argmax(prediction_flat, axis=2) == 2, 1,
                                                 np.where(np.argmax(prediction_flat, axis=2) == 1, -1, 0))

                    true_value_mapped = np.where(true_value_flat == 2, 1,
                                                 np.where(true_value_flat == 1, -1, 0))

                    return np.mean(calculate_angular_error(prediction_mapped, true_value_mapped))

        for dim in range(n_dims):
            num_classes = max(max(np.unique(true_value_flat)) + 1, len(np.unique(true_value_flat)))

            true_dim = true_value_flat[:, dim]
            pred_dim = np.argmax(prediction_flat[:, dim, :], axis=1)

            if num_classes > 2:
                avg_method = 'macro'
            else:
                avg_method = 'binary'

            if measure == 'accuracy':
                perf[dim] = metrics.accuracy_score(true_dim, pred_dim)

            elif measure == 'precision':
                perf[dim] =  metrics.precision_score(true_dim, pred_dim, average=avg_method,
                                                 zero_division=0)
            elif measure == 'recall':
                perf[dim] = metrics.recall_score(true_dim, pred_dim, average=avg_method, zero_division=0)
            elif measure == 'f1':
                perf[dim] = metrics.f1_score(true_dim, pred_dim, average=avg_method, zero_division=0)

            elif measure == 'auc':
                if len(np.unique(true_dim)) == 2:
                    perf[dim] = metrics.roc_auc_score(true_dim, prediction_flat[:, dim, 1])
                elif len(np.unique(true_dim)) > 2:
                    perf[dim] = metrics.roc_auc_score(true_dim, prediction_flat[:, dim, :], multi_class='ovr')
                else:
                    perf[dim] = np.nan
            else:
                raise ValueError(f"Unknown measure for categorical data: {measure}")

        return perf

    is_flat = (np.max(true_value_flat, axis=0) - np.min(true_value_flat, axis=0)) == 0

    if measure == 'CC':
        if n_samples < 2:
            return np.nan * np.ones((n_dims,))

        mean_true = np.mean(true_value_flat, axis=0)
        mean_pred = np.mean(prediction_flat, axis=0)

        true_centered = true_value_flat - mean_true
        pred_centered = prediction_flat - mean_pred

        numerator = np.sum(true_centered * pred_centered, axis=0)
        denominator = np.sqrt(np.sum(true_centered ** 2, axis=0) * np.sum(pred_centered ** 2, axis=0))

        with np.errstate(divide='ignore', invalid='ignore'):
            perf = np.where(denominator != 0, numerator / denominator, np.nan)

        perf[np.isnan(denominator)] = np.nan
        if true_value.ndim > 2:
            perf = perf.reshape(image_shape)

    elif measure == 'SSIM':
        perf = np.array([ssim(true_value[i], prediction[i], data_range=true_value[i].max() - true_value[i].min())
                         for i in range(n_samples)])

    elif measure == 'MSE':
        perf = metrics.mean_squared_error(true_value_flat, prediction_flat, multioutput='raw_values')
    elif measure == 'RMSE':
        mse = eval_prediction(true_value_flat, prediction_flat, 'MSE')
        perf = np.sqrt(mse)
    elif measure == 'NRMSE':
        rmse = eval_prediction(true_value_flat, prediction_flat, 'RMSE')
        std = np.std(true_value_flat, axis=0)
        perf = np.full(rmse.size, np.nan)
        perf[~is_flat] = rmse[~is_flat] / std[~is_flat]
    elif measure == 'MAE':
        perf = metrics.mean_absolute_error(true_value_flat, prediction_flat, multioutput='raw_values')
    elif measure == 'NMAE':
        mae = eval_prediction(true_value_flat, prediction_flat, 'MAE')
        denom = metrics.mean_absolute_error(true_value_flat - np.mean(true_value_flat, axis=0),
                                            np.zeros_like(prediction_flat), multioutput='raw_values')
        perf = np.full(mae.size, np.nan)
        perf[~is_flat] = mae[~is_flat] / denom[~is_flat]
    elif measure == 'EV':
        perf = metrics.explained_variance_score(true_value_flat, prediction_flat, multioutput='raw_values')
        perf[is_flat] = 0
    elif measure == 'R2':
        if n_samples < 2:
            return np.nan * np.ones((n_dims,))
        perf = metrics.r2_score(true_value_flat, prediction_flat, multioutput='raw_values')
        perf[is_flat] = 0
    else:
        raise ValueError(f"Unknown measure: {measure}")

    return perf


def calculate_angular_error(predictions, true):
    """
    Calculate the angular error between predicted and true movement vectors.

    Args:
        predictions (ndarray): Predicted (x, y) coordinates, shape (n, 2).
        true (ndarray): True (x, y) coordinates, shape (n, 2).

    Returns:
        ndarray: Angular error for each sample, shape (n,).
    """
    angles_pred = np.arctan2(predictions[:, 1], predictions[:, 0])
    angles_true = np.arctan2(true[:, 1], true[:, 0])

    angular_diff = np.abs(angles_pred - angles_true)

    angular_error = np.minimum(angular_diff, 2 * np.pi - angular_diff)

    angular_error_deg = np.degrees(angular_error)

    return angular_error_deg


def move_optimizer_state_to_cpu(state_dict):
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            state_dict[key] = value.cpu()
        elif isinstance(value, dict):
             move_optimizer_state_to_cpu(value)
        elif isinstance(value, list):
            state_dict[key] = [v.cpu() if isinstance(v, torch.Tensor) else v for v in value]
    return state_dict




class DataSplitter:
    def __init__(self, y, z, temp_mask, z_temp_mask=None, val_train_ratio=0.2):
        """
        Initializes the DataSplitter.
        """
        self.y = y
        self.z = z
        self.temporal_mask = temp_mask
        self.z_temporal_mask = z_temp_mask
        self.val_train_ratio = val_train_ratio
        self.means = {}
        self.stds = {}

    def split_data_for_fold(self, folds, ifold):
        """
        Splits data into train, validation, and test sets for a given fold.
        """
        fold_size = self.y.shape[0] // folds

        test_start = ifold * fold_size
        test_end = test_start + fold_size
        test_indices = np.arange(test_start, test_end)

        train_indices = np.setdiff1d(np.arange(self.y.shape[0]), test_indices)

        val_size = int(len(train_indices) * self.val_train_ratio)
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]

        def normalize(data, indices):
            subset = data[indices]
            mean, std = subset.mean(axis=0), subset.std() + 1e-7
            return (subset - mean) / std, mean, std

        yTrain, y_mean, y_std = normalize(self.y, train_indices)
        self.means[ifold], self.stds[ifold] = y_mean, y_std

        yVal = (self.y[val_indices] - y_mean) / y_std
        yTest = (self.y[test_indices] - y_mean) / y_std

        zTrain, zVal, zTest = self.z[train_indices], self.z[val_indices], self.z[test_indices]

        maskTrain, maskVal, maskTest = (
            self.temporal_mask[train_indices],
            self.temporal_mask[val_indices],
            self.temporal_mask[test_indices],
        )

        if self.z_temporal_mask is not None:
            zmaskTrain, zmaskVal, zmaskTest = (
                self.z_temporal_mask[train_indices],
                self.z_temporal_mask[val_indices],
                self.z_temporal_mask[test_indices],
            )
        else:
            zmaskTrain, zmaskVal, zmaskTest = maskTrain, maskVal, maskTest

        return {
            'yTrain': yTrain, 'yVal': yVal, 'yTest': yTest,
            'zTrain': zTrain, 'zVal': zVal, 'zTest': zTest,
            'maskTrain': maskTrain, 'maskVal': maskVal, 'maskTest': maskTest,
            'zmaskTrain': zmaskTrain, 'zmaskVal': zmaskVal, 'zmaskTest': zmaskTest,
        }