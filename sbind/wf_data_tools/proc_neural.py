import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt
from scipy.ndimage import convolve, gaussian_filter
from skimage.morphology import disk
from scipy.signal import convolve2d


def butter_highpass(cutoff, fs=30, order=2, output='ba'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    if output == 'sos':
        sos = butter(order, normal_cutoff, btype='high', analog=False, output=output)
        return sos
    elif output == 'ba':
        b, a = butter(order, normal_cutoff, btype='high', analog=False, output=output)
    return b, a


def highpass_filter(data, cutoff, fs=30, order=2, output='ba'):
    b, a = butter_highpass(cutoff, fs, order=order, output=output)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def highpass_filter_filtfilt(data, cutoff, fs=30, order=2):
    sos = butter_highpass(cutoff, fs, order=order, output='sos')
    filtered_data = sosfiltfilt(sos, data, axis=0)
    return filtered_data


def im_spatial_filter(data_in, filter_options):
    _, _, n_windows = data_in.shape
    data_out = np.full_like(data_in, np.nan)

    if filter_options[0] == 'disk':
        h = disk(filter_options[1])
    elif filter_options[0] == 'gaussian':
        h = None
    else:
        raise ValueError('This filter type has not been implemented')

    for window in range(n_windows):
        if filter_options[0] == 'disk':
            data_out[:, :, window] = convolve(data_in[:, :, window], h, mode='constant')
        elif filter_options[0] == 'gaussian':
            data_out[:, :, window] = gaussian_filter(data_in[:, :, window], sigma=filter_options[2])

    return data_out


def preprocess_data(data, trial_length, temporal_filter='causal', spatial_filter=None,
                    filt_options={'cutoff': 0.1, 'fs': 30, 'order': 2},
                    zscore=True, zscore_opts={'mean_ew': True, 'std_ew': False}, num_zscore_trials=107):
    num_data = data.shape[0]

    trial_inds = np.pad(np.cumsum(trial_length, dtype=int), (1, 0))
    train_indices = np.arange(trial_inds[num_zscore_trials])
    validation_indices = np.arange(trial_inds[num_zscore_trials], num_data)

    if zscore:
        if zscore_opts['mean_ew'] and zscore_opts['std_ew']:
            window_size = 60
            kernel = np.ones(window_size) / window_size

            data_reshaped = data.reshape(data.shape[0], -1)

            rolling_mean = convolve(data_reshaped, kernel[:, None], mode='nearest')
            squared_data = convolve(data_reshaped ** 2, kernel[:, None], mode='nearest')
            rolling_std = np.sqrt(squared_data - rolling_mean ** 2)

            data_reshaped = (data_reshaped - rolling_mean) / (rolling_std + 1e-8)

            data = data_reshaped.reshape(data.shape)

        elif zscore_opts['mean_ew'] and not zscore_opts['std_ew']:
            data = data - np.mean(data, axis=0, keepdims=True)
            data = data / (np.std(data, axis=(0, 1, 2), keepdims=True) + 1e-8)
        elif not zscore_opts['mean_ew'] and zscore_opts['std_ew']:
            data = (data - np.mean(data, axis=(0, 1, 2), keepdims=True))
            data = data / (np.std(data, axis=0, keepdims=True) + 1e-8)
        else:
            images_mean = np.mean(data[train_indices])
            images_std = np.std(data[train_indices]) + 1e-8
            data = (data - images_mean) / images_std
    else:
        images_mean = None
        images_std = None

    cutoff = filt_options.get('cutoff', 0.02)
    fs = filt_options.get('fs', 2)
    order = filt_options.get('order', 2)

    original_shape = data.shape
    reshaped_data = data.reshape(data.shape[0], data.shape[1], -1)

    if temporal_filter == 'causal':
        filtered_data = np.zeros_like(reshaped_data)
        for channel in range(reshaped_data.shape[1]):
            filtered_data[:, channel] = highpass_filter(reshaped_data[:, channel], cutoff=cutoff, fs=fs, order=order)
    elif temporal_filter == 'filtfilt':
        filtered_data = np.zeros_like(reshaped_data)
        for channel in range(reshaped_data.shape[1]):
            filtered_data[:, channel] = highpass_filter_filtfilt(reshaped_data[:, channel], cutoff=cutoff, fs=fs,
                                                                 order=order)
    elif temporal_filter == None:
        filtered_data = reshaped_data
    else:
        raise ValueError("Invalid temporal_filter argument. Choose 'causal' or 'filtfilt'.")

    filtered_data = filtered_data.reshape(original_shape)

    if spatial_filter:
        filtered_data = im_spatial_filter(filtered_data, filter_options=spatial_filter)

    if filtered_data.shape[2] > 128:
        filtered_data = filtered_data[:, :, :128]
    else:
        pad_width = 128 - filtered_data.shape[2]
        filtered_data = np.pad(filtered_data, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)

    filtered_data = filtered_data.transpose(0, 2, 1)

    result_dict = {
        'preprocessed_data': filtered_data,
        'zscore_mean': images_mean,
        'zscore_std': images_std,
        'temporal_filter': temporal_filter,
        'filtering_options': filt_options,
        'train_indices': train_indices,
        'validation_indices': validation_indices
    }

    return filtered_data, result_dict


def widefield_hemocorrect(data, hemodata, kernel_size=1, pxl_size_per_op=200):
    '''
        pixelwise widefield hemodynamic correction:
        assuming both intrinsic and neural channels
        are demeaned and normalized
    '''
    T, H, W = data.shape

    # Smooth hemo data
    # if smoothFact
    if kernel_size > 1:
        averaging_kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
        for i in range(T):
            hemodata[i] = convolve2d(hemodata[i], averaging_kernel, mode='same', boundary='symm', fillvalue=0)

    data = data.reshape((T, -1))
    hemodata = hemodata.reshape((T, -1))

    theta = np.zeros(data.shape[1])

    inds = np.arange(0, 1 + pxl_size_per_op * np.ceil(data.shape[1] / pxl_size_per_op), pxl_size_per_op, dtype=int)
    print(inds)
    for i in range(inds.shape[0] - 1):
        a = data[:, inds[i]:inds[i + 1]]
        b = hemodata[:, inds[i]:inds[i + 1]]
        # print((a * b).sum(axis=0))
        # print((b * b))
        temp_theta = (a * b).sum(axis=0) / (b * b).sum(axis=0)
        temp_theta[np.isnan(temp_theta)] = 0
        theta[inds[i]:inds[i + 1]] = temp_theta

    data = data - hemodata * theta  # subtract scaled hemoChannel from data
    theta = theta.reshape((H, W))
    data = data.reshape((T, H, W))

    return data, theta

