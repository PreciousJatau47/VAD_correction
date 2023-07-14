import numpy as np
import random
import math
import sys
import pickle
import GeneralUtils as gu
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from VADMaskEnum import *
from WindUtils import Polar2CartesianComponentsDf, Cartesian2PolarComponentsDf

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 12}
plt.rc('font', **font)

signal_func = lambda x, t: x[0] * np.sin(2 * np.pi * (1 / 360) * t + x[1])
MAX_FLOAT = sys.float_info.max
MIN_FLOAT = sys.float_info.min

class Missingness(enum.Enum):
    random = 0
    sector = 1

def GenerateVAD(speed, dirn, dirn_grid):
    phase = (90 - dirn) * np.pi / 180
    return signal_func([speed, phase], dirn_grid)

def AddVADMissingness(y_data, x_grid, miss_type, miss_prop=0.0, start_az=0):
    y_data = y_data.copy()
    if miss_prop == 0:
        return y_data

    if miss_type == Missingness.random:
        len_data = len(y_data)
        option_idxs = [i for i in range(len_data)]
        num_miss = int(miss_prop * len_data)
        miss_idxs = list(np.random.choice(a=option_idxs, size=num_miss, replace=False))
    elif miss_type == Missingness.sector:
        sector_width = miss_prop * 360
        stop_az = start_az + sector_width
        miss_sectors = [(start_az, 360), (0, stop_az % 360)] if stop_az > 360 else [(start_az, stop_az)]

        miss_idxs = False
        for start, stop in miss_sectors:
            idx = np.logical_and(x_grid >= start, x_grid < stop)
            miss_idxs = np.logical_or(miss_idxs, idx)

    y_data[miss_idxs] = -64.5
    return y_data


def GenerateNoiseNormal(mu=0, sdev=1, num_samples=100):
    return sdev * np.random.randn(num_samples) + mu


def MixVADs(vad_main, vad_sec, a, mix_r):
    """
    Mix two VADs according to the distribution specified by a and mix_r.
    :param vad_main:
    :param vad_sec:
    :param a: the proportion of the mixed vad that contain pure samples from vad_main
    :param mix_r: the proportion of the mixed vad that contains an average between vad_main and vad_sec
    :return:
    """
    assert a + mix_r <= 1
    N = len(vad_main)
    num_mix = int(mix_r * N)
    num_a = int(a * N)

    # Partition indices.
    idxs = [i for i in range(N)]
    mix_idxs = np.random.choice(a=idxs, size=num_mix, replace=False)
    rem_idxs_set = set(idxs) - set(mix_idxs)
    rem_idxs = list(rem_idxs_set)
    a_idxs = np.random.choice(a=rem_idxs, size=num_a, replace=False)
    b_idxs = list(rem_idxs_set - set(a_idxs))

    # Mix VADs
    vad_mix = vad_main.copy()
    vad_mix[b_idxs] = vad_sec[b_idxs]
    weights_mix = np.random.rand(num_mix, )
    vad_mix[mix_idxs] = weights_mix * vad_mix[mix_idxs] + (1 - weights_mix) * vad_sec[mix_idxs]

    # Probabilies. Gives the proportion of vad_main at some point i
    is_main_vad = np.ones((N,))
    is_main_vad[b_idxs] = 0
    is_main_vad[mix_idxs] = weights_mix

    return vad_mix, is_main_vad, a_idxs, mix_idxs, b_idxs, weights_mix


def fitVAD(pred_var, resp_var, signal_func, showDebugPlot, description, weights=1, min_required_nsamples=720):
    """
    :param pred_var: Should have shape (N,).
    :param resp_var: Should have shape (N,).
    :param signal_func:
    :param showDebugPlot:
    :return:
    """
    if len(pred_var) < min_required_nsamples:
        return np.nan, np.nan, None

    # Cost function.
    cost_func = lambda x: (x[0] * np.sin(2 * np.pi * (1 / 360) * pred_var + x[1]) - resp_var) * weights
    x0 = [random.uniform(0, 20), random.uniform(0, 2 * np.pi)]

    # Minimize cost function.
    x_hat = least_squares(cost_func, x0)
    soln = x_hat.x
    optimized_signal = signal_func(x_hat.x, pred_var)
    residue = resp_var - optimized_signal

    # Find the direction that maximizes the wind.
    wind_cost_func = lambda phi: -soln[0] * np.sin(2 * np.pi * (1 / 360) * phi + soln[1])
    wind_phi0 = [0]
    wind_dir_pred = minimize(wind_cost_func, wind_phi0)

    wind_speed = abs(wind_cost_func(wind_dir_pred.x[0]))
    wind_dir = wind_dir_pred.x[0] % 360

    if showDebugPlot:
        print("estimated wind speed :", wind_speed, "mps.\nDirection: ", wind_dir, " degrees.")

        fig, ax = plt.subplots()
        ax.plot(pred_var, resp_var, label="$V_r$", color="blue")
        ax.plot(pred_var, optimized_signal, label="$\hat{V}_{VAD}$", color="red", alpha=1.0)
        ax.axvline(x=wind_dir, linestyle='dotted', color='red')
        ax.axhline(y=wind_speed, linestyle='dotted', color='red')
        ax.set_xlim(0, 360)
        ax.set_xlabel("$\phi$ ($^{\circ}$)")
        ax.set_ylabel("$V_r$ (m/s)")
        ax.set_title("$\hat{V}_{VAD}$")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        fig, ax = plt.subplots(2, 2, figsize=(8.0, 6.4))
        ax[0, 0].plot(pred_var, resp_var, label="data", color="blue")
        ax[0, 0].plot(pred_var, optimized_signal, label="VAD fit", color="red", alpha=1.0)
        ax[0, 0].set_xlabel("Azimuth ($^{\circ}$)")
        ax[0, 0].set_ylabel("Velocity (mps)")
        ax[0, 0].set_title("VAD fit")
        ax[0, 0].legend()

        ax[0, 1].hist(pred_var, bins=4)
        ax[0, 1].set_title("Azimuth distribution.")
        ax[0, 1].set_xlabel("Azimuth ($^{\circ}$)")
        ax[0, 1].set_ylabel("Bin count (no unit)")

        ax[1, 0].plot(pred_var, residue, c='r')
        ax[1, 0].set_title("Residue")
        ax[1, 0].set_xlabel("Azimuth ($^{\circ}$)")
        ax[1, 0].set_ylabel("Res (mps)")

        ax[1, 1].hist(residue, bins=50)
        ax[1, 1].set_title("Residue")
        ax[1, 1].set_xlabel("Residue (mps)")
        ax[1, 1].set_ylabel("Bin count (no unit)")

        plt.tight_layout()
        plt.suptitle(description)
        plt.show()
    return wind_speed, wind_dir, optimized_signal


def IsVADDistributionValid(az_cut, wind_dir, echo_type):
    if echo_type == VADMask.weather:
        return True

    az_bin_size = 90
    az_cut_bins = (az_cut - wind_dir) % 360 // az_bin_size
    az_bins_id = np.unique(az_cut_bins)
    az_bins_id = az_bins_id.astype('int32')

    az_dist = np.zeros((360 // az_bin_size,))
    for bin_idx in az_bins_id:
        az_dist[bin_idx] = np.sum(az_cut_bins == bin_idx)
    az_dist /= max(az_dist)

    invalid_bins = az_dist < 0.4  # TODO define threshold
    total_invalid_bins = np.sum(invalid_bins)

    vad_valid_status = False
    if total_invalid_bins >= 3:
        vad_valid_status = False
    elif total_invalid_bins == 2:
        # Get indices of low sample bins.
        invalid_bins_idx = np.where(invalid_bins)
        curr_idx = invalid_bins_idx[0][0]
        next_idx = invalid_bins_idx[0][1]
        # Invalidate VAD if low sample bins are neighbours.
        if (next_idx == (curr_idx - 1) % len(invalid_bins)) or (next_idx == (curr_idx + 1) % len(invalid_bins)):
            vad_valid_status = False
        else:
            vad_valid_status = True
    else:
        vad_valid_status = True

    return vad_valid_status


def GetVADMask(data_table, echo_type, clf_purity_threshold=0.5):
    if echo_type == VADMask.biological:
        vad_mask_arr = np.logical_and(data_table["mask_differential_reflectivity"], data_table["hca_bio"])
        vad_mask_arr = np.logical_and(vad_mask_arr, data_table["mask_velocity"])
    elif echo_type == VADMask.insects:
        vad_mask_arr = np.logical_and(data_table["mask_differential_reflectivity"], data_table["hca_bio"])
        vad_mask_arr = np.logical_and(vad_mask_arr, data_table["BIProb"] < clf_purity_threshold)
        vad_mask_arr = np.logical_and(vad_mask_arr, data_table["mask_velocity"])
    elif echo_type == VADMask.birds:
        vad_mask_arr = np.logical_and(data_table["mask_differential_reflectivity"], data_table["hca_bio"])
        vad_mask_arr = np.logical_and(vad_mask_arr, data_table["BIProb"] >= (1 - clf_purity_threshold))
        vad_mask_arr = np.logical_and(vad_mask_arr, data_table["mask_velocity"])
    elif echo_type == VADMask.weather:
        vad_mask_arr = np.logical_and(data_table["hca_weather"], data_table["mask_velocity"])
    else:
        vad_mask_arr = data_table["mask_velocity"]
    return vad_mask_arr


def MinMaxNormalization(x, min_guard=MAX_FLOAT, max_guard=MIN_FLOAT):
    x_min = min(min_guard, np.nanmin(x))
    x_max = max(max_guard, np.nanmax(x))
    return (x - x_min) / (x_max - x_min)


def GetVADWeights(bi_scores_cut, echo_type, to_normalize=False):
    if len(bi_scores_cut) == 0:
        return 0

    if echo_type == VADMask.weather:
        return 1

    if to_normalize:
        if echo_type == VADMask.birds:
            return MinMaxNormalization(x=bi_scores_cut, min_guard=0.5, max_guard=1)
        if echo_type == VADMask.insects:
            return MinMaxNormalization(x=1 - bi_scores_cut, min_guard=0.5, max_guard=1)
        if echo_type == VADMask.biological:
            return MinMaxNormalization(x=1 - bi_scores_cut, min_guard=0, max_guard=1)
    else:
        if echo_type == VADMask.birds:
            return bi_scores_cut
        if echo_type == VADMask.insects:
            return 1 - bi_scores_cut
        if echo_type == VADMask.biological:
            return 1 - bi_scores_cut

    return None


def VADWindProfile(signal_func, vad_ranges, echo_type, radar_sp_table, showDebugPlot, use_weights=False,
                   clf_purity_threshold=0.5,min_required_nsamples=720):
    """
    :param signal_func:
    :param vad_ranges:
    :param echo_type:
    :param radar_sp_table:
    :param clf_purity_threshold:
    :param showDebugPlot:
    :param use_weights:
    :param min_required_nsamples:
    :return:
    """
    vad_mask = GetVADMask(radar_sp_table, echo_type, clf_purity_threshold=clf_purity_threshold)

    wind_profile_vad = []
    for height_vad in vad_ranges:
        range_diff = np.abs(radar_sp_table['height_bin_meters'] - height_vad)
        idx_height_vad = range_diff.idxmin(skipna=False)
        height_vad = np.array(radar_sp_table['height_bin_meters'])[idx_height_vad]

        idx_cut = radar_sp_table['height_bin_meters'] == height_vad
        idx_cut = np.logical_and(idx_cut, vad_mask)
        velocity_cut = radar_sp_table['velocity'][idx_cut]
        az_cut = radar_sp_table['azimuth'][idx_cut]
        bi_scores_cut = radar_sp_table['BIProb'][idx_cut]
        weights_cut = 1

        if use_weights:
            weights_cut = GetVADWeights(bi_scores_cut=bi_scores_cut,echo_type=echo_type)

        # Mean statistics per height bin.
        # TODO (test pending).
        idx_ref = np.logical_and(idx_cut, radar_sp_table["reflectivity"] > -32.5)
        ref_cut = radar_sp_table['reflectivity'][idx_ref]
        mean_ref = np.nanmean(ref_cut)
        mean_prob = np.nanmean(bi_scores_cut)

        description = "{}, {}m".format(GetVADMaskDescription(echo_type), height_vad)
        wind_speed, wind_dir, fitted_points = fitVAD(az_cut, velocity_cut, signal_func, showDebugPlot, description,
                                                     weights=weights_cut, min_required_nsamples=min_required_nsamples)

        if math.isnan(wind_dir):
            vad_valid = False
        else:
            vad_valid = IsVADDistributionValid(az_cut, wind_dir, echo_type)

        if not vad_valid:
            wind_speed, wind_dir, fitted_points = np.nan, np.nan, None

        windU, windV = Polar2CartesianComponentsDf(wind_speed, wind_dir)
        wind_profile_vad.append(
            [wind_speed, wind_dir, windU, windV, height_vad, len(velocity_cut), mean_ref, mean_prob])

    wind_profile_vad = pd.DataFrame(wind_profile_vad,
                                    columns=["wind_speed", "wind_direction", "wind_U", "wind_V", "height",
                                             "num_samples", "mean_ref", "mean_prob"])
    wind_profile_vad = wind_profile_vad.drop_duplicates(subset='height', keep='last', ignore_index=True)

    tmp = False
    if tmp:  # TODO remove or define proper flag.
        plt.figure()
        plt.plot(wind_profile_vad['wind_U'], wind_profile_vad['height'], color='blue', label="windU")
        plt.plot(wind_profile_vad['wind_V'], wind_profile_vad['height'], color='red', label="windV")
        plt.xlim(-25, 25)
        plt.ylim(0, 5000)
        plt.xlabel("wind component [m/s]")
        plt.ylabel("height [m]")
        plt.title("VAD wind")
        plt.legend()
        # plt.show()

        plt.figure()
        plt.plot(wind_profile_vad['num_samples'], wind_profile_vad['height'], color='blue', label="num_samples")
        # plt.xlim( 0, 720)
        plt.ylim(0, 1000)
        plt.title("Number of samples")
        plt.xlabel("number of samples [no units]")
        plt.ylabel("height [m]")

    return wind_profile_vad


def Main():
    showDebugPlot = True
    N = 720
    t = np.linspace(0, 360, N)

    true_wind_speed = random.uniform(0, 20)
    true_wind_dir = random.uniform(0, 360)
    print("true wind speed :", true_wind_speed, "mps.\nDirection: ", true_wind_dir, " degrees.")

    data = GenerateVAD(speed=true_wind_speed, dirn=true_wind_dir, dirn_grid=t) + GenerateNoiseNormal(num_samples=N)

    wind_speed, wind_dir, vad_fit = fitVAD(t, data, signal_func, showDebugPlot, description='')


if __name__ == "__main__":
    Main()
