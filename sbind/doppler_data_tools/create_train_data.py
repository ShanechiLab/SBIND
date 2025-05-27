import numpy as np
from .proc_behavior import parse_behavior_time

def map_labels(label):
    if np.isnan(label):
        return -1
    elif label in [1, 2, 3, 4]:
        return label - 1
    elif 6 <= label <= 9:
        return label - 2
    else:
        return label

def label_samples_valid_unitarget2(timestamps, behavior, actual_labels,
                                   target_start_key='memory', target_end_key='reward',
                                   target_acquire_key='target_acquire', num_clf_samples=1):
    target_start = parse_behavior_time(behavior, target_start_key)
    target_end = parse_behavior_time(behavior, target_end_key)
    target_acquire = parse_behavior_time(behavior, target_acquire_key)

    valid_trials = np.where(~np.isnan(target_acquire))[0]
    valid_indices = []
    labels = np.full_like(actual_labels, np.nan)

    for trial_idx in valid_trials:
        target_indices = np.where((timestamps >= target_start[trial_idx]) & (timestamps <= target_end[trial_idx]))[0]
        target_indices = target_indices[-num_clf_samples:]
        labels[target_indices] = actual_labels[target_indices]
        valid_indices.append(target_indices)
    return labels, valid_indices, valid_trials

def get_categorical_labels(data, target_start_key='target_acquire', target_end_key='reward',
                           target_acquire_key='target_acquire', num_clf_samples=1):
    labels, train_inds, valid_trials = label_samples_valid_unitarget2(data['timestamps'], data['behavior'],
                                                                      data['actual_labels'], target_start_key,
                                                                      target_end_key, target_acquire_key, num_clf_samples=num_clf_samples)

    mapped_z = np.vectorize(map_labels)(labels).astype('int64')

    mapped_z = mapped_z[:, np.newaxis]
    z_mask = (mapped_z > -1)

    return mapped_z, z_mask