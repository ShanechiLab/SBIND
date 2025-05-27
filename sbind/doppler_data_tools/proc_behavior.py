import numpy as np

def parse_behavior_time(behavior_struct, columnTitle):
    temp_cell = behavior_struct[columnTitle].flatten()

    empty_ind = [not isinstance(x, str) for x in temp_cell]
    negative_ind = [False] * len(temp_cell)
    for i, val in enumerate(temp_cell):
        if not empty_ind[i]:
            negative_ind[i] = '-' in val

    number_chars = len(temp_cell[empty_ind.index(False)])

    chartime = np.empty((len(temp_cell), number_chars), dtype='U1')

    for i, val in enumerate(temp_cell):
        if not empty_ind[i]:
            if negative_ind[i]:
                chartime[i, :] = list(val)
            else:
                chartime[i, -len(val):] = list(val)

    numTimeArray = np.zeros((chartime.shape[0], 3))

    for i in range(chartime.shape[0]):
        if not empty_ind[i]:
            temp_chartime = ''.join(chartime[i, :]).strip()
            colon_inds = [pos for pos, char in enumerate(temp_chartime) if char == ':']
            comma_ind = temp_chartime.find(',')

            numTimeArray[i, 0] = int(temp_chartime[:colon_inds[0]])
            numTimeArray[i, 2] = int(temp_chartime[colon_inds[1] + 1:comma_ind])

    numTime = np.full(len(temp_cell), np.nan)

    non_empty_idx = ~np.array(empty_ind)
    numTime[non_empty_idx] = (numTimeArray[non_empty_idx, 0] * 3600 +
                              numTimeArray[non_empty_idx, 1] * 60 +
                              numTimeArray[non_empty_idx, 2])

    numTime[negative_ind] *= -1

    return numTime