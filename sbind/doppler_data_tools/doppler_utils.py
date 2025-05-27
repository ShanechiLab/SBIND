import numpy as np

class DataSplitter:
    def __init__(self, y, z, trial_len=30, start_z_mask=20, val_train_ratio=None, zscore_ew=False, zscore=True):
        self.y = y
        self.z = z
        self.trial_len = trial_len
        self.val_train_ratio = val_train_ratio
        self.zscore_ew = zscore_ew
        self.zscore = zscore

        if isinstance(start_z_mask, int):
            self.start_z_mask = [start_z_mask]
        else:
            self.start_z_mask = start_z_mask

        self.means = {}
        self.stds = {}

    def split_data_for_fold(self, folds, ifold):
        data_size = len(self.y)
        all_inds = np.arange(data_size).reshape(-1, self.trial_len)

        test_trials = np.arange(ifold, all_inds.shape[0], folds)
        test_inds = all_inds[test_trials].flatten()

        if self.val_train_ratio is not None:
            num_val_folds = max(1, int(folds * self.val_train_ratio))
        else:
            num_val_folds = 1

        val_trials = []
        for i in range(1, num_val_folds + 1):
            val_fold = (ifold + i) % folds
            val_trials.extend(np.arange(val_fold, all_inds.shape[0], folds))
        val_trials = np.unique(val_trials)
        val_inds = all_inds[val_trials].flatten()

        train_inds = np.setdiff1d(np.arange(data_size), np.concatenate((test_inds, val_inds)))

        yTrain, yVal, yTest = self.y[train_inds, np.newaxis], self.y[val_inds, np.newaxis], self.y[
            test_inds, np.newaxis]

        y_mean = yTrain.mean(axis=0)
        if self.zscore:
            y_std = yTrain.std() if not self.zscore_ew else yTrain.std(axis=0)
        else:
            y_std = 1
        self.means[ifold], self.stds[ifold] = y_mean, y_std

        yTrain = np.where(y_std != 0, (yTrain - y_mean) / y_std, yTrain - y_mean)
        yVal = np.where(y_std != 0, (yVal - y_mean) / y_std, yVal - y_mean)
        yTest = np.where(y_std != 0, (yTest - y_mean) / y_std, yTest - y_mean)

        zTrain, zVal, zTest = self.z[train_inds], self.z[val_inds], self.z[test_inds]

        maskTrain, maskVal, maskTest = np.ones((zTrain.shape[0], 1)), np.ones((zVal.shape[0], 1)), np.ones((zTest.shape[0], 1))
        maskTrain[0::self.trial_len], maskVal[0::self.trial_len], maskTest[0::self.trial_len] = 0, 0, 0

        zmaskTrain, zmaskVal, zmaskTest = np.zeros((zTrain.shape[0], 1)), np.zeros((zVal.shape[0], 1)), np.zeros((zTest.shape[0], 1))
        for start in self.start_z_mask:
            zmaskTrain[start - 1::self.trial_len] = 1
            zmaskVal[start - 1::self.trial_len] = 1
            zmaskTest[start - 1::self.trial_len] = 1

        return {
            'yTrain': yTrain, 'yVal': yVal, 'yTest': yTest,
            'zTrain': zTrain, 'zVal': zVal, 'zTest': zTest,
            'maskTrain': maskTrain, 'maskVal': maskVal, 'maskTest': maskTest,
            'zmaskTrain': zmaskTrain, 'zmaskVal': zmaskVal, 'zmaskTest': zmaskTest
        }