import os
import pickle
import numpy as np
from VADMaskEnum import VADMask
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
from TrueWindEnum import *
from NexradUtils import GetTimeHourUTC
import sys
from AirspeedAnalysisUtils import *
import GeneralUtils as gu
from PreciousFunctions import PreciousCmap


font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 11}
plt.rc('font', **font)


def ImposeConstraints(df, constraints, idx=True):
    """
    :param df:
    :param constraints: list of constraints to be imposed on df. Structured as [(column name, lowerbound, upperbound),....]
    :return: index of df that satisfies constraints.
    """

    if not constraints:
        return pd.Series([True for _ in range(df.shape[0])])

    for constraint in constraints:
        if constraint[0] == 'file_name':
            idx_files = False
            for file_name in constraint[1]:
                idx_files = np.logical_or(idx_files, df["file_name"] == file_name)

            idx_files = np.logical_not(idx_files)
            idx = np.logical_and(idx, idx_files)
        else:
            if len(constraint) == 3:
                idx = np.logical_and(idx,
                                     np.logical_and(df[constraint[0]] >= constraint[1], df[constraint[0]] <= constraint[2]))
    return idx


def FilterFlightspeeds(wind_error, constraints):
    target_idx = ImposeConstraints(wind_error, constraints)
    return wind_error[target_idx]

def prepare_pcolor_grid_from_series(x, y, z, uniqueX = None, uniqueY = None):
    if uniqueX is None:
        uniqueX = np.unique(x)
    if uniqueY is None:
        uniqueY = np.unique(y)
    zGrid = np.full((len(uniqueX), len(uniqueY)), np.nan)

    for ix in range(len(uniqueX)):
        xIdx = x == uniqueX[ix]
        for iy in range(len(uniqueY)):
            yIdx = y == uniqueY[iy]
            idx = gu.logical_and(xIdx, yIdx)
            if np.sum(idx) == 0:
                continue
            zGrid[ix][iy] = z[idx]

    return uniqueX, uniqueY, zGrid

def prepare_weekly_data_for_pcolor_plot(key_col, x_col_name, y_col_name, in_data, month, noon_s_midnight, uniqueX = None, uniqueY = None):
    assert "week" in in_data.columns, "DataFrame must contain a week column."
    data_grids = []
    x_labels = []

    for week in range(5):
        x_label = ["{}/{}".format(month, week * 7 + curr_time // 24 + 1) if curr_time % 24 == 0 else "12 UTC"
                       for curr_time in noon_s_midnight]

        week_idx = in_data["week"] == week
        if np.sum(week_idx) <= 0:
            continue

        uniqueX, uniqueY, data_grid = prepare_pcolor_grid_from_series(
            in_data[x_col_name][week_idx], in_data[y_col_name][week_idx],
            in_data[key_col][week_idx], uniqueX=uniqueX, uniqueY=uniqueY)

        data_grids.append(data_grid)
        x_labels.append(x_label)

    return uniqueX, uniqueY, data_grids, x_labels


def VisualizeFlightspeeds(wind_error, constraints, color_info, c_group, save_plots, figure_summary_dir,
                          plot_title_suffix, out_name_suffix, max_airspeed=None, show_plots=True,
                          generate_weekly_month_profiles=True):

    delta_insect_prop = 5

    idx_constraints = ImposeConstraints(wind_error, constraints)
    idx_valid = np.isfinite(wind_error['airspeed_birds'])
    idx_valid = np.logical_and(idx_valid, np.isfinite(wind_error['airspeed_insects']))
    idx_valid = np.logical_and(idx_valid, idx_constraints)
    print("Number of airspeed samples: ", np.sum(idx_valid))

    if not max_airspeed:
        tmp = np.nanmax(wind_error['airspeed_insects'])
        max_airspeed = max(tmp, np.nanmax(wind_error['airspeed_birds']))
        max_airspeed = max_airspeed * 1.05

    # Plot 2. Bird x insect flightspeed.
    plt.figure(figsize=(6.4 * 1.2, 4.8 * 1.2))
    plt.plot([0, max_airspeed], [0, max_airspeed], linestyle='dashed', alpha=0.6)
    plt.scatter(x=wind_error['airspeed_insects'][idx_valid],
                y=wind_error['airspeed_birds'][idx_valid], s=80, c=color_info[c_group][0][idx_valid], alpha=0.3,
                cmap=color_info[c_group][1], vmin=0,
                vmax=100)

    plt.xlim(0, max_airspeed)
    plt.ylim(0, max_airspeed)
    plt.grid(True)
    cbar = plt.colorbar(ticks=color_info[c_group][2])
    cbar.ax.set_yticklabels(color_info[c_group][3])
    plt.xlabel("Flight speed insects [m/s]")
    plt.ylabel("Flight speed birds [m/s]")
    plt.title("".join(["Birds, insects flight speed comparison. ", plot_title_suffix]))
    if save_plots:
        plt.savefig(
            os.path.join(figure_summary_dir, "".join(["comparison_", out_name_suffix, ".png"])),
            dpi=200)

    # Plot 3. Height x insect prop x flightspeed.
    insect_prop_part = {0: (0, 33.33), 1: (33.33, 66.66), 2: (66.66, 100)}
    height_part = {0: (0, 350), 1: (350, 700), 2: (700, 1000)}

    fig, ax = plt.subplots(nrows=len(height_part), ncols=len(insect_prop_part), figsize=(6.4 * 2.8, 4.8 * 2.0))

    for i_prop_part in insect_prop_part.keys():
        for i_height_part in height_part.keys():
            idx_prop_part = np.logical_and(wind_error['insect_prop_bio'] >= insect_prop_part[i_prop_part][0],
                                           wind_error['insect_prop_bio'] < insect_prop_part[i_prop_part][1])
            idx_height_part = np.logical_and(wind_error['height_m'] >= height_part[i_height_part][0],
                                             wind_error['height_m'] < height_part[i_height_part][1])
            idx_part = np.logical_and(idx_prop_part, idx_height_part)
            idx_part = np.logical_and(idx_part, idx_valid)

            ax[i_height_part, i_prop_part].plot([0, max_airspeed], [0, max_airspeed], linestyle='dashed', alpha=0.6)
            ax[i_height_part, i_prop_part].scatter(x=wind_error['airspeed_insects'][idx_part],
                                                   y=wind_error['airspeed_birds'][idx_part], s=80,
                                                   c=color_info[c_group][0][idx_part],
                                                   alpha=0.3, cmap=color_info[c_group][1], vmin=0,
                                                   vmax=100)

            ax[i_height_part, i_prop_part].set_xlim(0, max_airspeed)
            ax[i_height_part, i_prop_part].set_ylim(0, max_airspeed)
            ax[i_height_part, i_prop_part].set_title(
                "Height, {} - {}m. Insect prop, {} - {}%".format(height_part[i_height_part][0],
                                                                 height_part[i_height_part][1],
                                                                 insect_prop_part[i_prop_part][0],
                                                                 insect_prop_part[i_prop_part][1]))
            ax[i_height_part, i_prop_part].grid(True)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Flight speed insects [m/s]")
    plt.ylabel("Flight speed birds [m/s]")
    plt.tight_layout()

    if save_plots:
        plt.savefig(os.path.join(figure_summary_dir, "".join(
            ["comparison_height_prop_", out_name_suffix, ".png"])), dpi=200)

    # Plot 4. Pure birds vs pure insects.
    impurity_threshold = 30
    idx_all_birds = wind_error['insect_prop_bio'] < impurity_threshold
    idx_all_insects = wind_error['insect_prop_bio'] > (100 - impurity_threshold)

    fig, ax = plt.subplots(3, 1, figsize=(6.4 * 1.14, 4.8 * 2))
    for height_bin in range(0, 3):
        idx_height_part = np.logical_and(wind_error['height_m'] >= height_part[height_bin][0],
                                         wind_error['height_m'] < height_part[height_bin][1])

        idx_pure_insects = np.logical_and(idx_height_part, idx_all_insects)
        idx_pure_birds = np.logical_and(idx_height_part, idx_all_birds)

        if np.sum(idx_pure_insects) > 0:
            ax[height_bin].hist(x=wind_error['airspeed_insects'][idx_pure_insects], color='red', label='insects',
                                alpha=0.3, density=True)

        if np.sum(idx_pure_birds) > 0:
            ax[height_bin].hist(x=wind_error['airspeed_birds'][idx_pure_birds], color='blue', label='birds', alpha=0.3,
                                density=True)
        ax[height_bin].grid(True)
        ax[height_bin].set_title(
            "Airspeed for single specie scans. {}% impurity. Height, {} - {} m".format(impurity_threshold,
                                                                                       height_part[height_bin][0],
                                                                                       height_part[height_bin][1]))
        ax[height_bin].legend()

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Airspeed (m/s)")
    plt.ylabel("Count (no units)")
    plt.tight_layout()

    if save_plots:
        plt.savefig(os.path.join(figure_summary_dir, "".join(["single_specie_airspeeds_", out_name_suffix, ".png"])),
                    dpi=200)

    # Plot 5. Histogram of data v insect proportion
    fig, ax = plt.subplots(nrows=len(height_part), ncols=1, figsize=(6.4 * 1.05, 4.8 * 2))
    for i_height_part in height_part.keys():
        idx_height_part = np.logical_and(wind_error['height_m'] >= height_part[i_height_part][0],
                                         wind_error['height_m'] < height_part[i_height_part][1])
        idx_height_part = np.logical_and(idx_height_part, idx_valid)

        ax[i_height_part].hist(wind_error["insect_prop_bio"][idx_height_part], bins=20)
        ax[i_height_part].set_xlim(0, 100)
        ax[i_height_part].set_title(
            "Height, {} - {}m".format(height_part[i_height_part][0], height_part[i_height_part][1]))
        ax[i_height_part].grid(True)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Insect proportion (relative to birds + insects)")
    plt.ylabel("Frequency of flight speed samples")
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(figure_summary_dir, "".join(["histogram_samples_", out_name_suffix, ".png"])),
                    dpi=200)

    # Plot 6. Histogram of bird, insect flightspeeds.
    plt.figure()
    plt.hist(x=wind_error["airspeed_insects"], color='red', alpha=0.5, label="insects")
    plt.hist(x=wind_error["airspeed_birds"], color='blue', alpha=0.5, label="birds")
    plt.xlabel("airspeed [m/s]")
    plt.ylabel("count [no units]")
    plt.title("".join(["Histogram of bird and insect airspeeds. ", plot_title_suffix]))
    plt.legend()
    if save_plots:
        plt.savefig(os.path.join(figure_summary_dir, "".join(["histogram_", out_name_suffix, ".png"])),
                    dpi=200)

    # TODO Refactor below code block.
    # Plot 7: Average flight speed vs % insects (birds).
    airspeed_df = wind_error[["insect_prop_bio", "airspeed_birds", "airspeed_insects", "height_m"]]
    airspeed_df["height_bins"] = airspeed_df["height_m"] // 350

    airspeed_df["insect_prop_bins"] = airspeed_df["insect_prop_bio"] // delta_insect_prop * delta_insect_prop + delta_insect_prop / 2
    airspeed_df = airspeed_df.groupby(["insect_prop_bins", "height_bins"], as_index=False).mean()
    max_airspeed_avg = max(np.nanmax(airspeed_df["airspeed_insects"]), np.nanmax(airspeed_df["airspeed_birds"]))
    max_airspeed_avg = np.ceil(max_airspeed_avg)

    fig, ax = plt.subplots(3, 1, figsize=(6.4 * 1.05, 4.8 * 2))
    for height_bin in range(0, 3):
        airspeed_height_idx = airspeed_df["height_bins"] == height_bin

        ax[height_bin].plot(airspeed_df["insect_prop_bins"][airspeed_height_idx],
                            airspeed_df["airspeed_insects"][airspeed_height_idx], color="red")
        ax[height_bin].plot(airspeed_df["insect_prop_bins"][airspeed_height_idx],
                            airspeed_df["airspeed_birds"][airspeed_height_idx], color="blue")
        ax[height_bin].set_ylim(0, max_airspeed_avg)
        ax[height_bin].grid(True)
        ax[height_bin].set_title("Height, {} - {}m".format(height_part[height_bin][0], height_part[height_bin][1]))

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("% insect echoes (no units)")
    plt.ylabel("mean airspeed (m/s)")
    plt.tight_layout()
    if save_plots:
        plt.savefig(
            os.path.join(figure_summary_dir, "".join(["mean_airspeed_v_insect_prop_", out_name_suffix, ".png"])),
            dpi=200)


    # # Plot 7: Average bird v insect airspeed height profile
    delta_height = 50
    airspeed_df = wind_error[["insect_prop_bio", "airspeed_birds", "airspeed_insects", "height_m"]]
    airspeed_df['height_bins'] = airspeed_df['height_m'] // delta_height * delta_height + delta_height / 2
    airspeed_df['insect_prop_bins'] = -1
    for part in insect_prop_part:
        idx = np.logical_and(airspeed_df['insect_prop_bio'] > insect_prop_part[part][0],
                             airspeed_df['insect_prop_bio'] < insect_prop_part[part][1])
        print(np.mean(idx))
        airspeed_df['insect_prop_bins'][idx] = part

    # Calculate the mean and standard deviation of airspeeds.
    averaged_airspeed_height = airspeed_df.groupby(['height_bins', 'insect_prop_bins'], as_index = False).mean()
    tmp_df = airspeed_df[["airspeed_birds", "airspeed_insects", "height_bins", "insect_prop_bins"]]
    std_df = tmp_df.groupby(['height_bins', 'insect_prop_bins'], as_index = False).apply(np.std)
    std_df = std_df.reset_index(drop = True)

    # Error bars.
    averaged_airspeed_height['lower_error_birds'] = averaged_airspeed_height['airspeed_birds'] - std_df['airspeed_birds']
    averaged_airspeed_height['upper_error_birds'] = averaged_airspeed_height['airspeed_birds'] + std_df['airspeed_birds']
    averaged_airspeed_height['lower_error_insects'] = averaged_airspeed_height['airspeed_insects'] - std_df['airspeed_insects']
    averaged_airspeed_height['upper_error_insects'] = averaged_airspeed_height['airspeed_insects'] + std_df['airspeed_insects']

    fig, ax = plt.subplots(1, 3, figsize = (2*6.4, 4.8))
    title_str = 'Height bin size: {}m'.format(delta_height)

    for part in insect_prop_part:
        idx = averaged_airspeed_height['insect_prop_bins'] == part

        ax[part].scatter(averaged_airspeed_height.loc[idx, 'airspeed_insects'], averaged_airspeed_height.loc[idx, 'height_bins'], color = 'red')
        ax[part].plot(averaged_airspeed_height.loc[idx, 'airspeed_insects'], averaged_airspeed_height.loc[idx, 'height_bins'], alpha = 0.6, color = 'red')
        ax[part].scatter(averaged_airspeed_height.loc[idx, 'airspeed_birds'], averaged_airspeed_height.loc[idx, 'height_bins'], color = 'blue')
        ax[part].plot(averaged_airspeed_height.loc[idx, 'airspeed_birds'], averaged_airspeed_height.loc[idx, 'height_bins'], alpha = 0.6, color = 'blue')

        ax[part].plot(averaged_airspeed_height.loc[idx, 'lower_error_insects'], averaged_airspeed_height.loc[idx, 'height_bins'], color = 'red', linestyle = 'dashed', alpha = 0.5)
        ax[part].plot(averaged_airspeed_height.loc[idx, 'upper_error_insects'], averaged_airspeed_height.loc[idx, 'height_bins'], color = 'red', linestyle = 'dashed', alpha = 0.5)
        ax[part].plot(averaged_airspeed_height.loc[idx, 'lower_error_birds'], averaged_airspeed_height.loc[idx, 'height_bins'], color = 'blue', linestyle = 'dashed', alpha = 0.5)
        ax[part].plot(averaged_airspeed_height.loc[idx, 'upper_error_birds'], averaged_airspeed_height.loc[idx, 'height_bins'], color = 'blue', linestyle = 'dashed', alpha = 0.5)

        ax[part].set_xlim(0,9)
        ax[part].set_ylim(0,1000)
        ax[part].grid(True)
        ax[part].set_title("Insect prop, {} - {}%".format(insect_prop_part[part][0],insect_prop_part[part][1]))

    plt.suptitle(title_str)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Flight speed insects [m/s]")
    plt.ylabel("Height [m]")
    plt.tight_layout()

    if save_plots:
        plt.savefig(
            os.path.join(figure_summary_dir, "airspeed_height_profile.png"),
            dpi=200)


    # Improvement analysis vs insect prop vs height
    height_ip_df = wind_error[["insect_prop_bio", "airspeed_birds", "airspeed_insects","airspeed_bio", "height_m", "prop_weather_scan"]]
    height_ip_df['airspeed_diff'] = height_ip_df['airspeed_birds'] - height_ip_df['airspeed_insects']
    height_ip_df['height_bins'] = height_ip_df['height_m'] // delta_height * delta_height + delta_height / 2
    height_ip_df["insect_prop_bins"] = height_ip_df["insect_prop_bio"] // delta_insect_prop * delta_insect_prop + delta_insect_prop / 2

    height_ip_df = height_ip_df.groupby(["height_bins","insect_prop_bins"], as_index=False).mean()

    unique_insect_prop_bins = np.arange(delta_insect_prop/2, 100, delta_insect_prop)
    unique_height_bins = np.arange(delta_height/2, 1000, delta_height)

    unique_height_bins, ins_prop_bins, diff_grid = prepare_pcolor_grid_from_series(height_ip_df['height_bins'],
                                                                                   height_ip_df['insect_prop_bins'],
                                                                                   height_ip_df['airspeed_diff'],
                                                                                   uniqueX=unique_height_bins,
                                                                                   uniqueY=unique_insect_prop_bins)

    # metrics
    # TODO(pjatau) maybe move threshold outside function.
    thresholds = (-0.5, 0.5)
    total_valid = np.sum(np.isfinite(height_ip_df['airspeed_diff']))
    pos_diff = np.sum(height_ip_df['airspeed_diff'] > thresholds[1]) / total_valid * 100
    zero_diff = np.sum(
        gu.logical_and(height_ip_df['airspeed_diff'] <= thresholds[1],
                       height_ip_df['airspeed_diff'] > thresholds[0])) / total_valid * 100
    neg_diff = np.sum(height_ip_df['airspeed_diff'] <= thresholds[0]) / total_valid * 100

    # Boundaries of airspeed diff
    max_diff = np.nanmax(height_ip_df['airspeed_diff'])
    min_diff = np.nanmin(height_ip_df['airspeed_diff'])
    max_amp = round(max(np.abs(max_diff), np.abs(min_diff)))
    max_amp = max(max_amp, 2)

    title_str = r"$bias_{birds} - bias_{insects}$"
    info_str = ">  {} m/s,  {}%\n<= {} m/s, {}%\nelse {}%".format(thresholds[1], round(pos_diff, 2), thresholds[0],
                                                               round(neg_diff, 2), round(zero_diff, 2))

    fig, ax = plt.subplots()
    cax = ax.pcolor(ins_prop_bins, unique_height_bins, diff_grid, vmin=-max_amp, vmax=max_amp, cmap='jet')
    ax.text(55, 800, info_str)
    cbar = fig.colorbar(cax)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1000)
    ax.set_xlabel('insect prop bio [%]')
    ax.set_ylabel('height [m]')
    ax.set_title(title_str)

    if save_plots:
        plt.savefig(
            os.path.join(figure_summary_dir, "airspeed_difference.png"),
            dpi=200)


    # Plot of height vs insect prop vs airspeed_bio
    max_airspeed_bio = np.max(np.abs(height_ip_df['airspeed_bio']))
    print("max_airspeed_bio: ", max_airspeed_bio)

    unique_height_bins, ins_prop_bins, airspeed_bio_grid = prepare_pcolor_grid_from_series(height_ip_df['height_bins'],
                                                                                   height_ip_df['insect_prop_bins'],
                                                                                   height_ip_df['airspeed_bio'],
                                                                                   uniqueX=unique_height_bins,
                                                                                   uniqueY=unique_insect_prop_bins)
    fig, ax = plt.subplots()
    cax = ax.pcolor(ins_prop_bins, unique_height_bins, airspeed_bio_grid, cmap='jet', vmin=0, vmax=max_airspeed_bio)
    cbar = fig.colorbar(cax)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1000)
    ax.set_xlabel('insect prop bio [%]')
    ax.set_ylabel('height [m]')
    ax.set_title(r"$airspeed_{bio}$")

    if save_plots:
        plt.savefig(
            os.path.join(figure_summary_dir, "airspeed_bio_height_insectprop.png"),
            dpi=200)


    # Plot of height vs insect prop vs airspeed_insects
    max_airspeed_ins = np.max(np.abs(height_ip_df['airspeed_insects']))
    print("max_airspeed_ins: ", max_airspeed_ins)

    unique_height_bins, ins_prop_bins, airspeed_ins_grid = prepare_pcolor_grid_from_series(height_ip_df['height_bins'],
                                                                                   height_ip_df['insect_prop_bins'],
                                                                                   height_ip_df['airspeed_insects'],
                                                                                   uniqueX=unique_height_bins,
                                                                                   uniqueY=unique_insect_prop_bins)
    fig, ax = plt.subplots()
    cax = ax.pcolor(ins_prop_bins, unique_height_bins, airspeed_ins_grid, cmap='jet', vmin=0, vmax=max_airspeed_bio)
    cbar = fig.colorbar(cax)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1000)
    ax.set_xlabel('insect prop bio [%]')
    ax.set_ylabel('height [m]')
    ax.set_title(r"$airspeed_{insects}$")
    plt.tight_layout()

    if save_plots:
        plt.savefig(
            os.path.join(figure_summary_dir, "airspeed_ins_height_insectprop.png"),
            dpi=200)


    # Plot: insect prop vs time of day vs height
    delta_time_hour = 1

    echo_profile_df = wind_error.loc[:,["month", "day" ,"time_hour", "insect_prop_bio", "height_m", "prop_weather_scan", "num_insects_height", "num_birds_height"]]
    echo_profile_df["height_bins"] = echo_profile_df['height_m'] // delta_height * delta_height + delta_height / 2
    echo_profile_df["time_hour_bins"] = echo_profile_df['time_hour'] // delta_time_hour * delta_time_hour + delta_time_hour /2

    # Plot: Insect profile per day
    # day_idx = echo_profile_df["day"] == 1
    ip_day = echo_profile_df[["day", "height_bins", "time_hour_bins", "insect_prop_bio"]]
    ip_day = ip_day.groupby(["day", "height_bins","time_hour_bins"], as_index=False).mean()

    unique_time_hr = np.arange(delta_time_hour/2, 24, delta_time_hour)

    nRows = 3
    nCols = 3
    fig, ax = plt.subplots(nRows, nCols, figsize=(6.4 *2.8, 4.8*2))
    echoProfileDays = [1, 2, 3, 8, 9, 13, 15, 20, 24]

    for i_day in range(len(echoProfileDays)):
        day = echoProfileDays[i_day]
        day_idx = ip_day["day"] == day
        if np.sum(day_idx) == 0:
            continue
        time_hr_day, heights_day, insect_prop_day = prepare_pcolor_grid_from_series(ip_day['time_hour_bins'][day_idx],
                                                                                    ip_day['height_bins'][day_idx],
                                                                                    ip_day['insect_prop_bio'][day_idx],
                                                                                    uniqueX=unique_time_hr,
                                                                                    uniqueY=unique_height_bins)
        im = ax[int((i_day) // nCols), int((i_day) % nCols)].pcolor(time_hr_day, heights_day,
                                                                    np.transpose(insect_prop_day), cmap='jet', vmin=0,
                                                                    vmax=100)
        ax[int((i_day) // nCols), int((i_day) % nCols)].set_xlim(0, 24)
        ax[int((i_day) // nCols), int((i_day) % nCols)].set_ylim(0, 1000)
        ax[int((i_day) // nCols), int((i_day) % nCols)].set_title(str(day))

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time [UTC]")
    plt.ylabel("Height [m]")
    plt.suptitle("Insect profile relative to biological echoes")
    plt.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8]) # (left, bottom, width, height)
    fig.colorbar(im, cax=cbar_ax)

    if save_plots:
        plt.savefig(
            os.path.join(figure_summary_dir, "insect_prop_height_timeday.png"),
            dpi=200)

    # Plot: Averaged insect profile
    height_time_df = echo_profile_df.loc[:,["height_bins", "time_hour_bins", "insect_prop_bio","num_insects_height", "num_birds_height"]]
    height_time_df = height_time_df.groupby(["height_bins","time_hour_bins"], as_index=False).mean()
    time_hr_bins, unique_height_bins, insect_prop_grid = prepare_pcolor_grid_from_series(
        height_time_df['time_hour_bins'],
        height_time_df['height_bins'],
        height_time_df['insect_prop_bio'], uniqueX=unique_time_hr,
        uniqueY=unique_height_bins)

    title_str = "Averaged insect profile relative to biological echoes"
    fig, ax = plt.subplots()
    cax = ax.pcolor(time_hr_bins, unique_height_bins, np.transpose(insect_prop_grid), cmap='jet', vmin = 0, vmax = 100)
    cbar = fig.colorbar(cax)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1000)
    ax.set_xlabel('Time [UTC]')
    ax.set_ylabel('Height [m]')
    ax.set_title(title_str)
    plt.tight_layout()

    if save_plots:
        plt.savefig(
            os.path.join(figure_summary_dir, "averaged_insect_prop_height_timeday.png"),
            dpi=200)

    # Plot: averaged number of birds, insects x height x time
    max_num = max(np.max(height_time_df["num_insects_height"]),np.max(height_time_df["num_birds_height"]))
    height_time_df["num_insects_height"] /= max_num
    height_time_df["num_birds_height"] /= max_num

    time_hr_bins, unique_height_bins, num_birds_grid = prepare_pcolor_grid_from_series(height_time_df['time_hour_bins'],
                                                                                       height_time_df['height_bins'],
                                                                                       height_time_df[
                                                                                           'num_birds_height'],
                                                                                       uniqueX=unique_time_hr,
                                                                                       uniqueY=unique_height_bins)
    time_hr_bins, unique_height_bins, num_insects_grid = prepare_pcolor_grid_from_series(
        height_time_df['time_hour_bins'],
        height_time_df['height_bins'],
        height_time_df['num_insects_height'], uniqueX=unique_time_hr,
        uniqueY=unique_height_bins)

    fig, ax = plt.subplots(1,2, figsize=(6.4*1.5, 4.8))
    cax = ax[0].pcolor(time_hr_bins, unique_height_bins, np.transpose(num_birds_grid), cmap='RdYlBu', vmin=0, vmax=1)
    ax[0].set_xlim(0, 24)
    ax[0].set_ylim(0, 1000)
    ax[0].set_xlabel('Time [UTC]')
    ax[0].set_ylabel('Height [m]')
    ax[0].set_title("num_birds")

    cax = ax[1].pcolor(time_hr_bins, unique_height_bins, np.transpose(num_insects_grid), cmap='RdYlBu', vmin=0, vmax=1)
    cbar = fig.colorbar(cax)
    ax[1].set_xlim(0, 24)
    ax[1].set_ylim(0, 1000)
    ax[1].set_xlabel('Time [UTC]')
    ax[1].set_ylabel('Height [m]')
    ax[1].set_title("num_insects")
    plt.tight_layout()

    if save_plots:
        plt.savefig(
            os.path.join(figure_summary_dir, "averaged_bird_insect_population_height_timeday.png"),
            dpi=200)


    # Plot: number of birds, insects profile for whole month
    if generate_weekly_month_profiles:

        population_df = wind_error.loc[:,
                        ["month", "day", "time_hour", "insect_prop_bio", "height_m", "prop_weather_scan",
                         "num_insects_height", "num_birds_height", "airspeed_bio"]]

        population_df["height_bins"] = population_df['height_m'] // delta_height * delta_height + delta_height / 2
        population_df["time_hour_bins"] = population_df[
                                              'time_hour'] // delta_time_hour * delta_time_hour + delta_time_hour / 2
        population_df["time_hour_week"] = (population_df["day"] - 1) % 7 * 24 + population_df["time_hour"]
        population_df["time_hour_week"] = population_df[
                                              'time_hour_week'] // delta_time_hour * delta_time_hour + delta_time_hour / 2
        population_df["week"] = (population_df["day"] - 1) // 7

        population_grouped_df = population_df.groupby(["week", "height_bins", "time_hour_week"], as_index=False).mean()

        max_num = max(np.max(population_grouped_df["num_insects_height"]),
                      np.max(population_grouped_df["num_birds_height"]))
        population_grouped_df["num_insects_height"] /= max_num
        population_grouped_df["num_birds_height"] /= max_num

        month = 5
        noon_s_midnight = np.arange(0, 24 * 7, 24)
        day_starts = np.arange(0, 24 * 7, 24)
        unique_time_week = np.arange(0.5, 168, 1)
        unique_height_bins = np.arange(delta_height / 2, 1000, delta_height)

        # Plot for number of bird gates.
        unique_time_week, unique_height_bins, weekly_data, xlabels = prepare_weekly_data_for_pcolor_plot(
            key_col='num_birds_height',
            x_col_name='time_hour_week',
            y_col_name='height_bins',
            in_data=population_grouped_df,
            month=month,
            noon_s_midnight=noon_s_midnight,
            uniqueX=unique_time_week,
            uniqueY=unique_height_bins)

        nWeeks = len(weekly_data)
        fig, ax = plt.subplots(nrows=nWeeks, ncols=1, figsize=(6.4 * 2.95, 4.8 * 2.0))
        for week in range(nWeeks):
            if weekly_data[week] is None:
                continue

            im = ax[week].pcolor(unique_time_week, unique_height_bins, np.transpose(weekly_data[week]), cmap='RdYlBu',
                                 vmin=0, vmax=1)
            ax[week].set_xticks(noon_s_midnight)
            ax[week].set_xticklabels(xlabels[week])
            ax[week].set_ylim(0, 1000)

            for day_start in day_starts:
                ax[week].axvline(x=day_start, color='k', linewidth=2, alpha=0.7)
                # if (day_start + 12) > 24*7:
                #     continue
                # ax[week].axvline(x=day_start + 12, color='c', linewidth = 2, alpha = 0.7)

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Time [UTC]")
        plt.ylabel("Height [m]")
        plt.suptitle("Number of range gates containing birds")
        plt.tight_layout()

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])  # (left, bottom, width, height)
        fig.colorbar(im, cax=cbar_ax)

        if save_plots:
            plt.savefig(
                os.path.join(figure_summary_dir, "bird_population_height_timeweek.png"),
                dpi=200)

        # Plot for number of insect gates
        unique_time_week, unique_height_bins, weekly_data, xlabels = prepare_weekly_data_for_pcolor_plot(
            key_col='num_insects_height',
            x_col_name='time_hour_week',
            y_col_name='height_bins',
            in_data=population_grouped_df,
            month=month,
            noon_s_midnight=noon_s_midnight,
            uniqueX=unique_time_week,
            uniqueY=unique_height_bins)

        nWeeks = len(weekly_data)
        fig, ax = plt.subplots(nrows=nWeeks, ncols=1, figsize=(6.4 * 2.95, 4.8 * 2.0))
        for week in range(nWeeks):
            if weekly_data[week] is None:
                continue

            im = ax[week].pcolor(unique_time_week, unique_height_bins, np.transpose(weekly_data[week]), cmap='RdYlBu',
                                 vmin=0, vmax=1)
            ax[week].set_xticks(noon_s_midnight)
            ax[week].set_xticklabels(xlabels[week])
            ax[week].set_ylim(0, 1000)

            for day_start in day_starts:
                ax[week].axvline(x=day_start, color='k', linewidth=2, alpha=0.7)
                # if (day_start + 12) > 24*7:
                #     continue
                # ax[week].axvline(x=day_start + 12, color='c', linewidth = 2, alpha = 0.7)

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Time [UTC]")
        plt.ylabel("Height [m]")
        plt.suptitle("Number of range gates containing insects")
        plt.tight_layout()

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])  # (left, bottom, width, height)
        fig.colorbar(im, cax=cbar_ax)

        if save_plots:
            plt.savefig(
                os.path.join(figure_summary_dir, "insect_population_height_timeweek.png"),
                dpi=200)


        # Plot for insect_prop_bio
        unique_time_week, unique_height_bins, weekly_data, xlabels = prepare_weekly_data_for_pcolor_plot(
            key_col='insect_prop_bio',
            x_col_name='time_hour_week',
            y_col_name='height_bins',
            in_data=population_grouped_df,
            month=month,
            noon_s_midnight=noon_s_midnight,
            uniqueX=unique_time_week,
            uniqueY=unique_height_bins)

        nWeeks = len(weekly_data)
        fig, ax = plt.subplots(nrows=nWeeks, ncols=1, figsize=(6.4 * 2.95, 4.8 * 2.0))
        for week in range(nWeeks):
            if weekly_data[week] is None:
                continue

            im = ax[week].pcolor(unique_time_week, unique_height_bins, np.transpose(weekly_data[week]), cmap='jet',
                                 vmin=0, vmax=100)
            ax[week].set_xticks(noon_s_midnight)
            ax[week].set_xticklabels(xlabels[week])
            ax[week].set_ylim(0, 1000)

            for day_start in day_starts:
                ax[week].axvline(x=day_start, color='k', linewidth=2, alpha=0.7)

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Time [UTC]")
        plt.ylabel("Height [m]")
        plt.suptitle("Relative proportion of insects")
        plt.tight_layout()

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])  # (left, bottom, width, height)
        fig.colorbar(im, cax=cbar_ax)

        if save_plots:
            plt.savefig(
                os.path.join(figure_summary_dir, "insect_prop_bio_height_timeweek.png"),
                dpi=200)

        # Plot for airspeed_bio height profile for whole month.
        unique_time_week, unique_height_bins, weekly_data, xlabels = prepare_weekly_data_for_pcolor_plot(
            key_col='airspeed_bio',
            x_col_name='time_hour_week',
            y_col_name='height_bins',
            in_data=population_grouped_df,
            month=month,
            noon_s_midnight=noon_s_midnight,
            uniqueX=unique_time_week,
            uniqueY=unique_height_bins)

        nWeeks = len(weekly_data)
        fig, ax = plt.subplots(nrows=nWeeks, ncols=1, figsize=(6.4 * 2.95, 4.8 * 2.0))
        for week in range(nWeeks):
            if weekly_data[week] is None:
                continue

            im = ax[week].pcolor(unique_time_week, unique_height_bins, np.transpose(weekly_data[week]), cmap='jet')
            ax[week].set_xticks(noon_s_midnight)
            ax[week].set_xticklabels(xlabels[week])
            ax[week].set_ylim(0, 1000)

            for day_start in day_starts:
                ax[week].axvline(x=day_start, color='k', linewidth=2, alpha=0.7)

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Time [UTC]")
        plt.ylabel("Height [m]")
        plt.suptitle("Airspeed of biological echoes")
        plt.tight_layout()

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])  # (left, bottom, width, height)
        fig.colorbar(im, cax=cbar_ax)

        if save_plots:
            plt.savefig(
                os.path.join(figure_summary_dir, "airspeed_bio_height_timeweek.png"),
                dpi=200)


    if show_plots:
        plt.show()

    print()


# TODO
# experiment id
# plot_title_suffix
# out_name_suffix
# check before and after when null constraints are used.
# apply contraints. check number of scans before and after weather correction.

def Main():
    # Inputs
    wind_dir = './vad_sounding_comparison_logs'
    experiment_id = "post_processing_default"  # "KENX_20180501_20180531_2hr_window"
    correct_hca_weather = True

    echo_count_dir = './analysis_output_logs'
    log_dir = "./post_processing_logs"

    figure_dir = "./figures"
    plot_title_suffix = "May 1 - 31, 2018"
    out_name_suffix = "May_1_31_2018"
    save_plots = True
    generate_weekly_month_profiles = True

    airspeed_log_dir = r'.\batch_analysis_logs'
    airspeed_files = ['KOHX_20180501_20180515_launched_2023118_16\KOHX_20180501_20180515.pkl',
                      'KOHX_20180516_20180531_launched_2023119_10\KOHX_20180516_20180531.pkl']
    # airspeed_files = [r'0_to_2pt5_el\KOHX_20180501_20180515_launched_2023121_21\KOHX_20180501_20180515.pkl',
    #                   r'0_to_2pt5_el\KOHX_20180516_20180531_launched_2023122_16\KOHX_20180516_20180531.pkl']
    # airspeed_files = [r'0_to_2pt5_el\KOHX_20180501_20180515_launched_2023121_21\KOHX_20180501_20180515.pkl',
    #                   r'0_to_2pt5_el\KOHX_20180516_20180531_launched_2023122_16\KOHX_20180516_20180531.pkl']
    # airspeed_files = [r'0_to_0pt5_el\KOHX_20180501_20180515_launched_2023124_11\KOHX_20180501_20180515.pkl',
    #                   r'0_to_0pt5_el\KOHX_20180516_20180531_launched_2023124_19\KOHX_20180516_20180531.pkl']

    use_ins_height_profile = True
    MAX_WEATHER_PROP = 10  # 10
    MAX_WEATHER_PROP_SCAN = 5 # 5  # 20 #15  # 20

    gt_wind_source = WindSource.rap_130
    wind_source_desc = GetWindSourceDescription(gt_wind_source)
    wind_source_desc = wind_source_desc.replace(' ', '_')

    # Load wind error
    wind_error = pd.DataFrame()
    for airspeed_file in airspeed_files:
        wind_error_path = os.path.join(airspeed_log_dir, airspeed_file)
        p_in = open(wind_error_path, 'rb')
        curr_wind_error, idx_last_log = pickle.load(p_in)
        p_in.close()
        wind_error = wind_error.append(curr_wind_error)

    # Calculate insect-bird height profile.
    if use_ins_height_profile:
        print('Using insect height profile. ')
        if 'num_insects_height' in wind_error.columns:
            wind_error['insect_prop_bio_height'] = 100 * wind_error['num_insects_height'] / (
                        wind_error['num_insects_height'] + wind_error['num_birds_height'])
            wind_error['insect_prop_bio'] = wind_error['insect_prop_bio_height']
        else:
            sys.exit('num_insects_height does not exist in wind_error.')

    # Filter wind error data.
    remove_cases_list = []
    constraints = [("prop_weather", 0, MAX_WEATHER_PROP), ("prop_weather_scan", 0, MAX_WEATHER_PROP_SCAN),
                   ('file_name', remove_cases_list)]
    idx_constraints = ImposeConstraints(wind_error, constraints)
    wind_error_constrained = wind_error[idx_constraints].reset_index(drop=True)
    # wind_error_constrained = wind_error # TODO EM

    # Visualize flight speeds
    color_weather = wind_error_constrained['prop_weather']
    color_weather = (color_weather / MAX_WEATHER_PROP) * 100
    ticks_weather = [0, 50, 100]
    ticklabels_weather = ["".join([str(round(tick * MAX_WEATHER_PROP / 100, 1)), '% WEA']) for tick in ticks_weather]

    color_info = {"insect_prop": (wind_error_constrained['insect_prop_bio'], "jet", [0, 25, 50, 75, 100],
                                  ['BIR', 'BIR MAJ', r'BIR $\approx$ INS', 'INS MAJ', 'INS']),
                  "weather": (color_weather, "jet", ticks_weather, ticklabels_weather)}
    # values are (colour, colour map, ticks, labels)

    c_group = "insect_prop"  # "insect_prop"
    out_name_suffix = "_".join(["color", c_group, out_name_suffix])

    if correct_hca_weather:
        figure_summary_dir = os.path.join(figure_dir, experiment_id, 'summary', 'hca_weather_corrected',
                                          ''.join(['airspeed_', wind_source_desc]))
    else:
        figure_summary_dir = os.path.join(figure_dir, experiment_id, 'summary', 'hca_default',
                                          ''.join(['airspeed_', wind_source_desc]))

    if not os.path.isdir(figure_summary_dir):
        os.makedirs(figure_summary_dir)

    VisualizeFlightspeeds(wind_error_constrained, constraints, color_info, c_group, save_plots, figure_summary_dir,
                          plot_title_suffix, out_name_suffix, max_airspeed=None, show_plots=True,
                          generate_weekly_month_profiles=generate_weekly_month_profiles)

    # Search for cases defined by constraints.
    # constraints = [("height_m", 700, 1000), ("insect_prop_bio", 0, 33.33), ("airspeed_insects", 0, 6),
    #                ("airspeed_birds", 7.5, 11), ("prop_weather", 0, MAX_WEATHER_PROP),
    #                ("prop_weather_scan", 0, MAX_WEATHER_PROP_SCAN), ('file_name', remove_cases_list)]
    # constraints = [("height_m", 700, 1000), ("insect_prop_bio", 33.33, 66.66), ("airspeed_insects", 12.5, 19),
    #                ("airspeed_birds", 12, 22), ("prop_weather", 0, MAX_WEATHER_PROP),
    #                ("prop_weather_scan", 0, MAX_WEATHER_PROP_SCAN)]

    constraints = [("height_m", 732, 734)]

    wind_error_filt = FilterFlightspeeds(wind_error_constrained, constraints)
    # wind_error_filt = wind_error_filt.sort_values(by=['prop_birds'])
    print(np.unique(wind_error_filt.file_name))
    print()

    # ['KOHX20180501_000411_V06_wind' 'KOHX20180501_135820_V06_wind'
    #  'KOHX20180501_234533_V06_wind' 'KOHX20180501_235518_V06_wind']


Main()
