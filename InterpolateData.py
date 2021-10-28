import matplotlib.pyplot as plt
import numpy as np
import math


def BinaryPartition(arr, key) -> int:
    """
    Returns the index of the right most element in the left partition.
    Left partition is defined as arr <= key.
    :param arr:
    :param key:
    :return: index of right most element in left partition, whether element at index equals key.
    """
    lo = 0
    hi = len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2

        if arr[mid] == key:
            return (mid, True)
        elif arr[mid] < key:
            lo = mid + 1
        else:  # arr[mid] > key
            hi = mid - 1

        # Ensure left partition is returned
        if arr[mid] > key:
            mid -= 1

    return mid, False

def GetAveragingWeights(x, x_key, left_part, right_part):
    assert x[left_part] <= x_key, "GetAveragingWeights. Element in left partition is greater than middle element"
    assert x_key <= x[right_part], "GetAveragingWeights. Element in right partition is less than middle element"

    delta_x = x[right_part] - x[left_part]
    w_left = (x[right_part] - x_key) / delta_x
    w_right = (x_key - x[left_part]) / delta_x
    return w_left, w_right, delta_x

WEIGHT_THRESHOLD = 0.80

def WeightedMean(x_arr, x_key, left_part, right_part, y_arr, max_delta_x = None):
    """
    :param x_key: value around which to partition predictor variable.
    :param left_part: last predictor value in left partition.
    :param right_part: first predictor value in right partition.
    :param y_arr: response array.
    :return:
    """
    w_left, w_right, delta_x = GetAveragingWeights(x=x_arr, x_key=x_key, left_part=left_part, right_part=right_part)

    if max_delta_x != None and delta_x > max_delta_x and not (w_left > WEIGHT_THRESHOLD or w_right > WEIGHT_THRESHOLD):
        return np.nan

    return w_left * y_arr[left_part] + w_right * y_arr[right_part]

def Interpolate(x, y, x_interp, max_delta_x):
    y_interp = np.empty(x_interp.shape)
    y_interp[:] = np.nan

    # sort x, y -> O(n)
    i_x = np.argsort(x)
    x = x[i_x]
    y = y[i_x]

    for i in range(len(x_interp)):
        x_key = x_interp[i]
        left_part, found = BinaryPartition(arr=x, key=x_key)    # O(log n)
        # print(x_key, left_part, found)
        if found:
            if math.isnan(y[left_part]):
                if left_part > 0:
                    right_part = left_part + 1
                    left_part -= 1
                    y_interp[i] = WeightedMean(x, x_key, left_part, right_part, y, max_delta_x)
            else:
                y_interp[i] = y[left_part]
        else:  # not found in array
            if left_part >= 0 and left_part < len(x) - 1:
                right_part = left_part + 1
                y_interp[i] = WeightedMean(x, x_key, left_part, right_part, y, max_delta_x)

    return y_interp

def Main():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([1, 2, np.nan, 3.5, 4, 4.5, 2, 1])
    max_delta_x = 1.0
    x_interp = np.arange(0, 8.5, 0.5 / 4)
    y_interp = Interpolate(x=x, y=y, x_interp=x_interp, max_delta_x = max_delta_x)

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, y)
    plt.xlim(0, 8)
    plt.ylim(0, 6)

    plt.figure()
    plt.scatter(x_interp, y_interp)
    plt.plot(x_interp, y_interp)
    plt.xlim(0, 8)
    plt.ylim(0, 6)

    plt.show()

# Main()



