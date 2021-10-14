import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize

MIN_FRACTION_SAMPLES_REQUIRED = 1.5


def fitVAD(pred_var, resp_var, signal_func, showDebugPlot):
    """
    :param pred_var: Should have shape (N,).
    :param resp_var: Should have shape (N,).
    :param signal_func:
    :param showDebugPlot:
    :return:
    """
    if (len(pred_var) / 720) < MIN_FRACTION_SAMPLES_REQUIRED:
        return np.nan, np.nan, None

    # Cost function.
    cost_func = lambda x: x[0] * np.sin(2 * np.pi * (1 / 360) * pred_var + x[1]) - resp_var
    x0 = [random.uniform(0, 20), random.uniform(0, 2 * np.pi)]

    # Minimize cost function.
    x_hat = least_squares(cost_func, x0)
    soln = x_hat.x
    optimized_signal = signal_func(x_hat.x, pred_var)

    # Find the direction that maximizes the wind.
    wind_cost_func = lambda phi: -soln[0] * np.sin(2 * np.pi * (1 / 360) * phi + soln[1])
    wind_phi0 = [0]
    wind_dir_pred = minimize(wind_cost_func, wind_phi0)

    wind_speed = abs(wind_cost_func(wind_dir_pred.x[0]))
    wind_dir = wind_dir_pred.x[0] % 360
    # print("estimated wind speed :", wind_speed, "mps.\nDirection: ", wind_dir, " degrees.")

    if showDebugPlot:
        plt.figure()
        plt.plot(pred_var, resp_var, label="data", color="blue")
        plt.plot(pred_var, optimized_signal, label="VAD fit", color="red", alpha=1.0)
        plt.legend()
        plt.show()
    return wind_speed, wind_dir, optimized_signal


def VADWindProfile(signal_func, vad_ranges, vad_mask, radar_sp_table, showDebugPlot):
    """
    :param signal_func:
    :param vad_ranges:
    :param radar_sp_table:
    :return:
    """
    wind_profile_vad = []
    for height_vad in vad_ranges:
        range_diff = np.abs(radar_sp_table['height_bin_meters'] - height_vad)
        idx_height_vad = range_diff.idxmin(skipna=False)
        height_vad = np.array(radar_sp_table['height_bin_meters'])[idx_height_vad]

        idx_cut = radar_sp_table['height_bin_meters'] == height_vad
        idx_cut = np.logical_and(idx_cut, vad_mask)
        velocity_cut = radar_sp_table['velocity'][idx_cut]
        az_cut = radar_sp_table['azimuth'][idx_cut]

        # Mean reflectivity per height bin.
        # TODO use reflectivity mask.
        idx_ref = np.logical_and(idx_cut, radar_sp_table["reflectivity"] != -33.0)
        mean_ref = np.mean(radar_sp_table['reflectivity'][idx_ref])

        wind_speed, wind_dir, fitted_points = fitVAD(az_cut, velocity_cut, signal_func, False)
        wind_U = wind_speed * np.sin(wind_dir * np.pi / 180)
        wind_V = wind_speed * np.cos(wind_dir * np.pi / 180)
        wind_profile_vad.append([wind_speed, wind_dir, wind_U, wind_V, height_vad, len(velocity_cut), mean_ref])

    wind_profile_vad = pd.DataFrame(wind_profile_vad,
                                    columns=["wind_speed", "wind_direction", "wind_U", "wind_V", "height",
                                             "num_samples","mean_ref"])

    if showDebugPlot:
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
    showDebugPlot = False
    N = 360
    t = np.arange(0, N)

    x = [random.uniform(0, 20), random.uniform(0, 2 * np.pi)]
    true_wind_speed = x[0]
    true_wind_dir = np.rad2deg(np.pi / 2 - x[1]) % 360
    print("true wind speed :", true_wind_speed, "mps.\nDirection: ", true_wind_dir, " degrees.")

    signal_func = lambda x, t: x[0] * np.sin(2 * np.pi * (1 / 360) * t + x[1])
    data = signal_func(x, t) + random.uniform(0, 2) * np.random.randn(N)

    wind_speed, wind_dir, vad_fit = fitVAD(t, data, signal_func, showDebugPlot)

# Main()
