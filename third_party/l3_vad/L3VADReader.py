from __future__ import print_function
import numpy as np
import os
from .L3VAD import VADFile
from .L3VADUtils import *

import pandas as pd
import matplotlib.pyplot as plt

from WindUtils import Polar2CartesianComponentsDf

KM_2_KFT = 3.2808
KNOT_T_MPS = 0.514444


def ReadL3VAD(radar_id, time, local_path, print_vad=False, plot_wind_barbs=False, figure_dir='./'):
    plot_time = parse_time(time)
    iname = build_has_name(radar_id, plot_time)
    vad_file_path = "%s/%s" % (local_path, iname)

    vad = VADFile(open(vad_file_path, 'rb'))
    l3_vad = pd.DataFrame(vad._data)

    # Print out VAD
    if print_vad:
        for block in vad._text_message:
            for line in block:
                print(line)

    if plot_wind_barbs:
        dirn = (l3_vad['wind_dir'] + 180) % 360
        u, v = Polar2CartesianComponentsDf(spd=l3_vad['wind_spd'], dirn=dirn)
        y = l3_vad['altitude']
        y_kft = y * KM_2_KFT
        x = np.array([0 for i in range(len(y))])

        wind_ppi_id = time
        wind_ppi_id = wind_ppi_id.replace('/', '')
        wind_ppi_id = wind_ppi_id.replace('-', '')
        wind_ppi_id = ''.join([radar_id, wind_ppi_id])
        figure_path = os.path.join(figure_dir, wind_ppi_id + '_barbs.png')

        wind_time = time.split('/')[-1]
        title = ','.join([radar_id, time])
        xticks = [-1, 0, 1]
        xtick_lab = ['', wind_time, '']

        plt.figure()
        plt.barbs(x, y_kft, u, v, length=6)
        plt.xlim(-1, 1)
        plt.ylim(-1, 10)
        plt.xticks(xticks, xtick_lab)
        plt.grid(True)
        plt.xlabel("Time (UTC)")
        plt.ylabel("ALT KFT")
        plt.title(title)
        plt.savefig(figure_path, dpi=200)
        # plt.show()

    return l3_vad


def GetL3VADWindProfile(local_path, radar_id, time, height_grid_m, height_binsize_m, print_vad=False,
                        plot_wind=False, figure_dir='./'):
    l3_vad = ReadL3VAD(radar_id=radar_id, time=time, local_path=local_path, print_vad=print_vad,
                       plot_wind_barbs=plot_wind, figure_dir=figure_dir)

    # Match variables to local VAD profiles
    rename_guide = {'wind_spd': 'wind_speed', 'wind_dir': 'wind_direction'}
    l3_vad.rename(columns=rename_guide, inplace=True)
    l3_vad['num_samples'] = np.nan
    l3_vad['mean_ref'] = np.nan
    l3_vad['mean_prob'] = np.nan
    l3_vad['coverage_perc'] = np.nan
    l3_vad['num_samples_50'] = np.nan

    # Match units/reference to local VAD profiles.
    l3_vad['wind_direction'] = (l3_vad['wind_direction'] + 180) % 360  # Convert to wind destination.
    l3_vad['wind_speed'] *= KNOT_T_MPS  # Convert to mps.
    l3_vad["height_m"] = l3_vad["slant_range"] * np.sin(l3_vad["elev_angle"] * np.pi / 180) * 1e3
    l3_vad["height"] = (l3_vad["height_m"] // height_binsize_m * height_binsize_m + height_binsize_m / 2)

    # Convert wind to cartesian components.
    windU, windV = Polar2CartesianComponentsDf(l3_vad['wind_speed'], l3_vad['wind_direction'])
    l3_vad['wind_U'] = windU
    l3_vad['wind_V'] = windV

    # Match height grid.
    height_grid_m = pd.DataFrame({'height': height_grid_m})
    l3_vad = l3_vad.groupby(["height"], as_index=False).mean()
    l3_vad = pd.merge(height_grid_m, l3_vad, on="height", how="left")

    if plot_wind:
        # Output path
        wind_ppi_id = time
        wind_ppi_id = wind_ppi_id.replace('/', '')
        wind_ppi_id = wind_ppi_id.replace('-', '')
        wind_ppi_id = ''.join([radar_id, wind_ppi_id])
        figure_path = os.path.join(figure_dir, wind_ppi_id + '.png')

        # Plots
        valid_idx = l3_vad['height_m'] < 2000
        valid_idx = np.logical_and(valid_idx, np.isfinite(l3_vad['height_m']))
        fig, ax = plt.subplots(ncols=2)

        # Wind direction
        ax[0].scatter(l3_vad["wind_direction"][valid_idx], l3_vad["height"][valid_idx])
        ax[0].plot(l3_vad["wind_direction"][valid_idx], l3_vad["height"][valid_idx])
        ax[0].set_xlim(0, 360)
        ax[0].grid(True)
        ax[0].set_xlabel(r"Direction ($^\circ$)")
        ax[0].set_ylabel(r"Height (m)")

        # Wind speed
        ax[1].scatter(l3_vad["wind_speed"][valid_idx], l3_vad["height"][valid_idx])
        ax[1].plot(l3_vad["wind_speed"][valid_idx], l3_vad["height"][valid_idx])
        ax[1].set_xlim(0, 20)
        ax[1].grid(True)
        ax[1].set_xlabel(r"Speed (m/s)")
        ax[1].set_ylabel(r"Height (m)")
        plt.suptitle(wind_ppi_id)
        plt.savefig(figure_path, dpi=200)
        plt.show()

    return l3_vad


def Main():
    local_path = './level3_VAD_wind_profiles/KOHX_20180503_test_data/20180503'  # './HAS012373834/0002/'
    radar_id = 'KOHX'
    time = '2018-05-03/2206'  # '2018-05-03/1803'
    figure_dir = './'

    height_binsize = 40
    height_grid_m = np.arange(100, 2400, height_binsize)

    try:
        l3_vad = GetL3VADWindProfile(local_path=local_path, radar_id=radar_id, time=time, height_grid_m=height_grid_m,
                                     height_binsize_m=height_binsize, figure_dir=figure_dir, plot_wind=True)
    except Exception as e:
        print("An error occured in GetL3VADWindProfile(): ", str(e))

    print()


if __name__ == "__main__":
    Main()
