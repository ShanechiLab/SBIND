
import cv2
from . import matlab_to_python as mat_to_py
import numpy as np
import os

def load_and_resize_data(base_path, num_trials, x_pixel, y_pixel,
                         x_start=30, x_end=570, y_start=0, y_end=540):
    datalist = list()
    trial_length = np.zeros((num_trials,))

    # Set default y_end to the full height if not provided

    for i in range(1, num_trials + 1):
        blue_file = os.path.join(base_path, f'blueData_Trial{i}.mat')
        hemo_file = os.path.join(base_path, f'hemoData_Trial{i}.mat')

        if os.path.exists(hemo_file):
            bluedata = mat_to_py.loadmat(blue_file)
            num_blue_frames = bluedata['bData'].shape[2]
            hemodata = mat_to_py.loadmat(hemo_file)
            num_hemo_frames = hemodata['vData'].shape[2]
            trial_length[i - 1] = num_blue_frames

            # Create temporary array to hold resized data
            temp = np.zeros((num_blue_frames, 2, y_pixel, x_pixel), dtype=np.float32)

            # Resize each frame for both blue and hemo data based on the clipping coordinates
            for j in range(num_blue_frames):
                temp[j, 0] = cv2.resize(bluedata['bData'][y_start:y_end, x_start:x_end, j],
                                        (y_pixel, x_pixel), interpolation=cv2.INTER_AREA)
                temp[j, 1] = cv2.resize(hemodata['vData'][y_start:y_end, x_start:x_end, j],
                                        (y_pixel, x_pixel), interpolation=cv2.INTER_AREA)

            # Append the resized data to the list
            datalist.append(temp)
            if i%10 == 0:
                print(f"Processed trial {i}")

    # Concatenate all trials into one array
    datalist = np.concatenate(datalist, axis=0)
    return datalist, trial_length
