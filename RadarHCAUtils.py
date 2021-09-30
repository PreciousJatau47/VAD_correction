import sys
import numpy as np
import pandas as pd


def read_info_from_radar_name(radar_file):
    """
    Finds the corresponding hca PPI name given a radar PPI name.
    :param radar_file:
    :return:
    """
    radar_name = radar_file[:4]
    year = radar_file[4:8]
    month = radar_file[8:10]
    day = radar_file[10:12]
    hh = radar_file[13:15]
    mm = radar_file[15:17]
    ss = radar_file[17:19]
    return radar_name, year, month, day, hh, mm, ss

def find_hca_file_name_from_radar(radar_file):
    """
    Finds the corresponding hca PPI name given a radar PPI name.
    :param radar_file:
    :return:
    """
    radar_name = radar_file[:4]
    year = radar_file[4:8]
    month = radar_file[8:10]
    day = radar_file[10:12]
    hhmm = radar_file[13:17]
    # KOUN_...
    # KHUN ...
    # KOHX
    return "KOUN_SDUS84_N0H{}_{}{}{}{}".format(radar_name[1:], year, month, day, hhmm)

def ReadRadarSlice(radar, radar_products_slice, slice_idx):
    """
    :param radar:
    :param radar_products_slice:
    :param slice_idx:
    :return:
    """
    if slice_idx > 1 or slice_idx < 0:
        sys.exit("ReadRadarCutAsTable can only process slice 0 and slice 1.")

    radar_range = radar.range['data'] / 1000  # in km
    sweep_ind = radar.get_slice(slice_idx)
    radar_az_deg = radar.azimuth['data'][sweep_ind]  # in degrees

    if slice_idx == 0:
        radar_mask = radar.fields["differential_reflectivity"]['data'][sweep_ind].mask
    elif slice_idx == 1:
        radar_mask = radar.fields["velocity"]['data'][sweep_ind].mask

    data_slice = []
    for radar_variable in radar_products_slice[slice_idx]:
        print(radar_variable)
        curr_data = radar.fields[radar_variable]['data'][sweep_ind].data  # .reshape(-1, 1)
        data_slice.append(curr_data)
    print("ReadRadarSlice ", data_slice[0].shape)

    return radar_range, radar_az_deg, data_slice.copy(), radar_mask

def MatchGates(arr, key_arr):
    """
    :param arr:
    :param key_arr:
    :return:
    """
    match_idxs = [np.nan for _ in range(len(key_arr))]
    for i in range(len(key_arr)):
        diff = list(np.abs(arr - key_arr[i]))
        match_idxs[i] = diff.index(min(diff))
        # print(diff[i])
        # print(i, ",", match_idxs[i])
        # print(arr[match_idxs[i]], "-", key_arr[i])
    return match_idxs

def MergeRadarAndHCA(radar, radar_products_slice, hca, maxRange):
    """
    :param radar:
    :param radar_products_slice:
    :param hca:
    :param maxRange:
    :return:
    """
    range_dp, az_dp, data_slice_dp, radar_mask_dp = ReadRadarSlice(radar, radar_products_slice, 0)  # dp
    range_sp, az_sp, data_slice_sp, radar_mask_sp = ReadRadarSlice(radar, radar_products_slice, 1)  # sp

    print(hca.fields['radar_echo_classification']['options'])
    hca_az_deg = hca.azimuth['data']  # in degrees
    hca_range = hca.range['data'] / 1000  # in km
    hca_mask = hca.fields['radar_echo_classification']['data'].mask
    hca_data = hca.fields['radar_echo_classification']['data'].data
    print(hca_data.shape)

    range_common = np.arange(10, maxRange, 0.25)
    az_common = np.arange(0, 360, 0.5)

    # Match dual pol variables to common grid.
    range_idxs_dp = MatchGates(range_dp, range_common)
    az_idxs_dp = MatchGates(az_dp, az_common)
    print(data_slice_dp[0].shape)
    for i in range(len(data_slice_dp)):
        data_slice_dp[i] = data_slice_dp[i][np.ix_(az_idxs_dp, range_idxs_dp)]
        data_slice_dp[i] = data_slice_dp[i].reshape(-1,1)
    print(data_slice_dp[0].shape)
    radar_mask_dp = radar_mask_dp[np.ix_(az_idxs_dp, range_idxs_dp)]
    radar_mask_dp = np.logical_not(radar_mask_dp.reshape(-1,1))

    # Match single pol variables to common grid.
    range_idxs_sp = MatchGates(range_sp, range_common)
    az_idxs_sp = MatchGates(az_sp, az_common)
    print(data_slice_sp[0].shape)
    for i in range(len(data_slice_sp)):
        data_slice_sp[i] =  data_slice_sp[i][np.ix_(az_idxs_sp, range_idxs_sp)]
        data_slice_sp[i] = data_slice_sp[i].reshape(-1,1)
    print(data_slice_sp[0].shape)
    radar_mask_sp = radar_mask_sp[np.ix_(az_idxs_sp, range_idxs_sp)]
    radar_mask_sp = np.logical_not(radar_mask_sp.reshape(-1,1))

    range_idxs_hca = MatchGates(hca_range, range_common)
    az_idxs_hca = MatchGates(hca_az_deg, az_common)
    hca_data =  hca_data[np.ix_(az_idxs_hca, range_idxs_hca)]
    hca_data = hca_data.reshape(-1,1)
    hca_mask = hca_mask[np.ix_(az_idxs_hca, range_idxs_hca)]
    hca_mask = np.logical_not(hca_mask.reshape(-1,1))

    # combine variables
    data_slice = data_slice_dp
    data_slice.extend(data_slice_sp)
    columns = radar_products_slice[0].copy()
    columns.extend(radar_products_slice[1].copy())

    data_slice.append(hca_data)
    data_slice.append(hca_mask)
    columns.extend(["hca","hca_mask"])

    data_slice.append(radar_mask_dp)
    data_slice.append(radar_mask_sp)
    columns.extend(["mask_dp", "mask_sp"])

    range_common, az_common = np.meshgrid(range_common, az_common)
    az_common = az_common.reshape(-1, 1)
    range_common = range_common.reshape(-1, 1)
    data_slice.append(az_common)
    data_slice.append(range_common)
    columns.extend(["azimuth", "range"])

    data_table = np.concatenate(data_slice, axis=1)
    data_table = pd.DataFrame(data_table, columns=columns)
    return data_table

def ReadRadarCutAsTable(radar, radar_products_slice, slice_idx):
    """
    Assumes elevation is 0.5 degrees.
    :param radar:
    :param radar_products_slice:
    :param slice_idx:
    :return:
    """
    radar_range, radar_az_deg, data_slice, radar_mask = ReadRadarSlice(radar, radar_products_slice, slice_idx)

    for i in range(len(data_slice)):
        data_slice[i] = data_slice[i].reshape(-1, 1)
    print("ReadRadarCutAsTable ", data_slice[0].shape)

    columns = radar_products_slice[slice_idx].copy()
    columns.extend(["mask", "range", "azimuth"])
    data_slice.append(radar_mask.reshape(-1, 1))
    range_grid, az_grid = np.meshgrid(radar_range, radar_az_deg)
    range_grid = range_grid.reshape(-1, 1)
    data_slice.append(range_grid)
    az_grid = az_grid.reshape(-1, 1)
    data_slice.append(az_grid)

    radar_data_table = np.concatenate(data_slice, axis=1)
    radar_data_table = pd.DataFrame(radar_data_table, columns=columns)

    return radar_data_table
