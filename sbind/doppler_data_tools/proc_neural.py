import numpy as np
from scipy.signal import butter, lfilter
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk

def butter_highpass(cutoff, fs=2, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs=2, order=2):
    b, a = butter_highpass(cutoff, fs, order=order)
    filtered_data = lfilter(b, a, data, axis=0)
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

def preprocess_data(data, zscore=True, temporal_filter=None, spatial_filter=None, fix_dims = True, mask=None,
                    zscore_opts={'mean_ew':True, 'std_ew':False}, filt_options={'cut_off':0.02, 'fs':2, 'order':2},
                    filter_options=['disk', 2, 0]):
    if len(data.shape) != 3:
        raise ValueError('Data must be 3D')

    if temporal_filter:
        data_shape = data.shape
        data = data.reshape(data.shape[0], -1)
        data = highpass_filter(data, **filt_options)
        data = data.reshape(data_shape)

    if zscore:
        if zscore_opts['mean_ew'] and not zscore_opts['std_ew']:
            data = data - np.mean(data, axis=0, keepdims=True)
            data = data / (np.std(data, axis=(0,1,2), keepdims=True) + 1e-8)
        elif not zscore_opts['mean_ew'] and zscore_opts['std_ew']:
            data = (data - np.mean(data, axis=(0,1,2), keepdims=True))
            data = data / (np.std(data, axis=0, keepdims=True) + 1e-8)

    if spatial_filter:
        data = im_spatial_filter(data, filter_options=filter_options)

    if fix_dims:
        if data.shape[2] > 128:
            data = data[:, :, :128]
        else:
            pad_width = 128 - data.shape[2]
            data = np.pad(data, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)

        data = data.transpose(0, 2, 1)

        if mask is not None:
            data = data * mask

    return data