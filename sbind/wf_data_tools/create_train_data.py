import numpy as np
import os
import  pickle

_LOAD_PATH = 'PATH_TO_DATA'

def load_data(load_path=None, file_name=None, variable_trial_length=False):
    if load_path is None:
        load_path = _LOAD_PATH
    if file_name is None:
        file_name = 'DEFAULT_FILE_NAME'

    if file_name.endswith('.npz'):
        data = np.load(os.path.join(load_path, file_name))['data']
    elif file_name.endswith('.pkl'):
        with open(os.path.join(load_path, file_name), 'rb') as file:
            data = pickle.load(file)
        data = data['data']
    if variable_trial_length:
        load_path = os.path.join(load_path, 'VARIABLE_TRIAL_LENGTH_FILE.pkl')
    else:
        load_path = os.path.join(load_path, 'FIXED_TRIAL_LENGTH_FILE.pkl')

    with open(load_path, 'rb') as file:
        loaded_dict = pickle.load(file)

    return data, loaded_dict

def preprocess_data(data, mask_file):
    mask_ds = np.load(mask_file)['mask_ds']
    mask_ds = 1 - mask_ds
    data = data * mask_ds
    return data, mask_ds

def prepare_data(data, loaded_dict, train_indices, val_indices, ifold, zscore=True):
    yTrain = np.copy(data[train_indices, np.newaxis])

    if val_indices.size > 0:
        yVal = np.copy(data[val_indices, np.newaxis])
    else:
        yVal = np.empty((0, *data.shape[1:]))

    test_indices = loaded_dict['val_fixed_indices_org'][ifold]
    if test_indices.size > 0:
        yTest = np.copy(data[test_indices, np.newaxis])
    else:
        yTest = np.empty((0, *data.shape[1:]))

    if zscore:
        y_mean = yTrain.mean(axis=0)
        y_std = yTrain.std()

        y_std = y_std if y_std > 0 else 1.0

        yTrain = (yTrain - y_mean) / y_std
        if yVal.size > 0:
            yVal = (yVal - y_mean) / y_std
        if yTest.size > 0:
            yTest = (yTest - y_mean) / y_std

    return yTrain, yVal, yTest

def prepare_behavior_data(loaded_dict, train_indices_z, val_indices_z, ifold, zscore=True):
    if train_indices_z.size > 0:
        zTrain = np.concatenate((loaded_dict['svd_behavior'][ifold]['train'],
                                 loaded_dict['svd_behavior_motion'][ifold]['train']), axis=1)[train_indices_z]
    else:
        zTrain = np.empty((0, loaded_dict['svd_behavior'][ifold]['train'].shape[1] +
                           loaded_dict['svd_behavior_motion'][ifold]['train'].shape[1]))

    if val_indices_z.size > 0:
        zVal = np.concatenate((loaded_dict['svd_behavior'][ifold]['train'],
                               loaded_dict['svd_behavior_motion'][ifold]['train']), axis=1)[val_indices_z]
    else:
        zVal = np.empty((0, loaded_dict['svd_behavior'][ifold]['train'].shape[1] +
                         loaded_dict['svd_behavior_motion'][ifold]['train'].shape[1]))

    test_indices = loaded_dict['val_fixed_indices_folds'][ifold]
    if test_indices.size > 0:
        zTest = np.concatenate((loaded_dict['svd_behavior'][ifold]['test'],
                                loaded_dict['svd_behavior_motion'][ifold]['test']), axis=1)[test_indices]
    else:
        zTest = np.empty((0, loaded_dict['svd_behavior'][ifold]['test'].shape[1] +
                          loaded_dict['svd_behavior_motion'][ifold]['test'].shape[1]))

    if zscore and zTrain.size > 0:
        z_mean = zTrain.mean(axis=0)
        z_std = zTrain.std(axis=0)

        z_std[z_std == 0] = 1.0

        zTrain = (zTrain - z_mean) / z_std
        if zVal.size > 0:
            zVal = (zVal - z_mean) / z_std
        if zTest.size > 0:
            zTest = (zTest - z_mean) / z_std

    return zTrain, zVal, zTest

def prepare_fold_data_classify(ifold, data, loaded_dict, zz, split_ratio=0.2, variable_trial_length=False, zscore=True):
    if variable_trial_length:
        trial_lengths = loaded_dict['trial_length']
        val_trials = np.arange(ifold, len(trial_lengths), 5)
        train_trials = np.setdiff1d(np.arange(len(trial_lengths)), val_trials)
        train_trial_lengths = trial_lengths[train_trials]
        cum_train_trial_lengths = np.cumsum(
            train_trial_lengths[train_trial_lengths > 0])
        test_test_lengths = trial_lengths[val_trials]
        cum_test_trial_lengths = np.cumsum(
            test_test_lengths[test_test_lengths > 0])
        val_train_trials = np.arange(0, len(cum_train_trial_lengths), int(1 / split_ratio))

        val_train_indices = []
        for i in range(len(val_train_trials)):
            this_trial_end = cum_train_trial_lengths[val_train_trials[i]]
            this_trial_start = 0 if i == 0 else cum_train_trial_lengths[val_train_trials[i] - 1]
            val_train_indices.extend(np.arange(this_trial_start, this_trial_end))

        this_val_indices_z = np.array(val_train_indices)
        this_train_indices_z = np.setdiff1d(loaded_dict['train_fixed_indices_folds'][ifold], val_train_indices)

        this_val_indices_y = loaded_dict['train_fixed_indices_org'][ifold][this_val_indices_z]
        this_train_indices_y = loaded_dict['train_fixed_indices_org'][ifold][this_train_indices_z]

    else:
        trial_len = 170
        this_train_inds_y = loaded_dict['train_fixed_indices_org'][ifold].reshape(-1, trial_len)

        val_trials_portion = split_ratio
        this_val_trials = np.arange(2, this_train_inds_y.shape[1], int(1 / val_trials_portion))
        this_train_trials = np.setdiff1d(np.arange(this_train_inds_y.shape[1]), this_val_trials)

        this_train_indices_y = this_train_inds_y[:, this_train_trials].reshape(-1)
        this_val_indices_y = this_train_inds_y[:, this_val_trials].reshape(-1)

        this_train_inds_z = loaded_dict['train_fixed_indices_folds'][ifold].reshape(-1, trial_len)
        this_train_indices_z = this_train_inds_z[:, this_train_trials].reshape(-1)
        this_val_indices_z = this_train_inds_z[:, this_val_trials].reshape(-1)

    yTrain, yVal, yTest = prepare_data(data, loaded_dict, this_train_indices_y, this_val_indices_y, ifold, zscore=zscore)
    zTrain, zVal, zTest = zz[this_train_indices_y], zz[this_val_indices_y], zz[loaded_dict['val_fixed_indices_org'][ifold]]

    if variable_trial_length:
        maskTrainVal = np.ones(cum_train_trial_lengths[-1], )
        maskTrainVal[cum_train_trial_lengths[:-1]] = 0
        maskTrainVal[0] = 0
        maskTrain = maskTrainVal[this_train_indices_z]
        maskVal = maskTrainVal[this_val_indices_z]
        maskTest = np.ones((zTest.shape[0],))
        maskTest[cum_test_trial_lengths[:-1]] = 0
        maskTest[0] = 0

        zmaskTrainVal = np.zeros(cum_train_trial_lengths[-1], )
        zmaskTrainVal[cum_train_trial_lengths-1] = 1
        zmaskTrain = zmaskTrainVal[this_train_indices_z]
        zmaskVal = zmaskTrainVal[this_val_indices_z]
        zmaskTest = np.zeros((zTest.shape[0],))
        zmaskTest[cum_test_trial_lengths-1] = 1
    else:
        maskTrain = np.ones((zTrain.shape[0],))
        maskTrain[0::170] = 0.0
        maskVal = np.ones((zVal.shape[0],))
        maskVal[0::170] = 0.0
        maskTest = np.ones((zTest.shape[0],))
        maskTest[0::170] = 0.0
        zmaskTrain = np.zeros((zTrain.shape[0],))
        zmaskTrain[169::170] = 1.0
        zmaskTest = np.zeros((zTest.shape[0],))
        zmaskTest[169::170] = 1.0
        zmaskVal = np.zeros((zVal.shape[0],))
        zmaskVal[169::170] = 1

    return yTrain, yVal, yTest, zTrain, zVal, zTest, maskTrain, maskVal, maskTest, zmaskTrain, zmaskVal, zmaskTest

def prepare_fold_data(ifold, data, loaded_dict, split_ratio=0.2, variable_trial_length=False, zscore=True):
    if variable_trial_length:
        trial_lengths = loaded_dict['trial_length']
        val_trials = np.arange(ifold, len(trial_lengths), 5)
        train_trials = np.setdiff1d(np.arange(len(trial_lengths)), val_trials)
        train_trial_lengths = trial_lengths[train_trials]
        cum_train_trial_lengths = np.cumsum(
            train_trial_lengths[train_trial_lengths > 0])
        test_test_lengths = trial_lengths[val_trials]
        cum_test_trial_lengths = np.cumsum(
            test_test_lengths[test_test_lengths > 0])

        if split_ratio > 0.0:
            val_train_trials = np.arange(0, len(cum_train_trial_lengths), int(1 / split_ratio))

            val_train_indices = []
            for i in range(len(val_train_trials)):
                this_trial_end = cum_train_trial_lengths[val_train_trials[i]]
                this_trial_start = 0 if i == 0 else cum_train_trial_lengths[val_train_trials[i] - 1]
                val_train_indices.extend(np.arange(this_trial_start, this_trial_end))

            this_val_indices_z = np.array(val_train_indices)
            this_train_indices_z = np.setdiff1d(loaded_dict['train_fixed_indices_folds'][ifold], val_train_indices)

            this_val_indices_y = loaded_dict['train_fixed_indices_org'][ifold][this_val_indices_z]
            this_train_indices_y = loaded_dict['train_fixed_indices_org'][ifold][this_train_indices_z]

        else:
            this_train_indices_z = loaded_dict['train_fixed_indices_folds'][ifold]
            this_train_indices_y = loaded_dict['train_fixed_indices_org'][ifold]
            this_val_indices_z = np.array([])
            this_val_indices_y = np.array([])

    else:
        trial_len = 170
        this_train_inds_y = loaded_dict['train_fixed_indices_org'][ifold].reshape(-1, trial_len)

        if split_ratio > 0.0:
            val_trials_portion = split_ratio
            this_val_trials = np.arange(2, this_train_inds_y.shape[1], int(1 / val_trials_portion))
            this_train_trials = np.setdiff1d(np.arange(this_train_inds_y.shape[1]), this_val_trials)

            this_train_indices_y = this_train_inds_y[:, this_train_trials].reshape(-1)
            this_val_indices_y = this_train_inds_y[:, this_val_trials].reshape(-1)

            this_train_inds_z = loaded_dict['train_fixed_indices_folds'][ifold].reshape(-1, trial_len)
            this_train_indices_z = this_train_inds_z[:, this_train_trials].reshape(-1)
            this_val_indices_z = this_train_inds_z[:, this_val_trials].reshape(-1)
        else:
            this_train_indices_y = this_train_inds_y.reshape(-1)
            this_val_indices_y = np.array([])

            this_train_inds_z = loaded_dict['train_fixed_indices_folds'][ifold].reshape(-1, trial_len)
            this_train_indices_z = this_train_inds_z.reshape(-1)
            this_val_indices_z = np.array([])

    yTrain, yVal, yTest = prepare_data(data, loaded_dict, this_train_indices_y, this_val_indices_y, ifold, zscore=zscore)
    zTrain, zVal, zTest = prepare_behavior_data(loaded_dict, this_train_indices_z, this_val_indices_z, ifold, zscore=zscore)

    if variable_trial_length:
        maskTrainVal = np.ones(cum_train_trial_lengths[-1], )
        maskTrainVal[cum_train_trial_lengths[:-1]] = 0
        maskTrainVal[0] = 0
        maskTrain = maskTrainVal[this_train_indices_z]
        maskVal = maskTrainVal[this_val_indices_z] if this_val_indices_z.size > 0 else np.empty((0,))
        maskTest = np.ones((zTest.shape[0],))
        maskTest[cum_test_trial_lengths[:-1]] = 0
        maskTest[0] = 0
    else:
        maskTrain = np.ones((zTrain.shape[0],))
        maskTrain[0::170] = 0.0
        maskVal = np.ones((zVal.shape[0],))
        maskVal[0::170] = 0.0
        maskTest = np.ones((zTest.shape[0],))
        maskTest[0::170] = 0.0

    return yTrain, yVal, yTest, zTrain, zVal, zTest, maskTrain, maskVal, maskTest