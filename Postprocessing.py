import os
import sys
import pickle
import numpy as np
from VADMaskEnum import VADMask
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
from TrueWindEnum import *
from NexradUtils import GetTimeHourUTC
from scipy import stats
from AirspeedAnalysisUtils import *
import GeneralUtils as gu
from WindUtils import *
from PreciousFunctions import PreciousCmap

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 12}
plt.rc('font', **font)


class Constants:
    DELTA_INSECT_PROP = 5
    DELTA_HEIGHT = 50
    DELTA_TIME_HR = 1

    IMPURITY_TOLERANCE = 30

    DELTA_INSECT_PROP_PART = 33.33
    DELTA_HEIGHT_PART = 350

    BI_BIAS_DIFF_THRESHOLD = 0.5


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
                                     np.logical_and(df[constraint[0]] >= constraint[1],
                                                    df[constraint[0]] <= constraint[2]))
    return idx


def FilterFlightspeeds(wind_error, constraints):
    target_idx = ImposeConstraints(wind_error, constraints)
    return wind_error[target_idx]


def prepare_pcolor_grid_from_series(x, y, z, uniqueX=None, uniqueY=None):
    if uniqueX is None:
        uniqueX = np.unique(x)
    if uniqueY is None:
        uniqueY = np.unique(y)
    zGrid = {key: np.full((len(uniqueX), len(uniqueY)), np.nan) for key in z}

    for ix in range(len(uniqueX)):
        xIdx = x == uniqueX[ix]
        for iy in range(len(uniqueY)):
            yIdx = y == uniqueY[iy]
            idx = gu.logical_and(xIdx, yIdx)
            if np.sum(idx) == 0:
                continue

            for key in z:
                zGrid[key][ix][iy] = z[key][idx]

    return uniqueX, uniqueY, zGrid


def prepare_weekly_data_for_pcolor_plot(key_cols, x_col_name, y_col_name, in_data, month, noon_s_midnight, uniqueX=None,
                                        uniqueY=None):
    assert "week" in in_data.columns, "DataFrame must contain a week column."
    data_grids = {}
    x_labels = []

    for week in range(5):
        x_label = ["{}/{}".format(month, week * 7 + curr_time // 24 + 1) if curr_time % 24 == 0 else "12 UTC"
                   for curr_time in noon_s_midnight]

        week_idx = in_data["week"] == week
        if np.sum(week_idx) <= 0:
            continue
        # Gather all data for the current week
        z_dict = {key_col: in_data[key_col][week_idx] for key_col in key_cols}

        uniqueX, uniqueY, data_grid = prepare_pcolor_grid_from_series(
            x=in_data[x_col_name][week_idx], y=in_data[y_col_name][week_idx],
            z=z_dict, uniqueX=uniqueX, uniqueY=uniqueY)

        data_grids["week_{}".format(week)] = data_grid
        x_labels.append(x_label)

    return uniqueX, uniqueY, data_grids, x_labels


def plot_averages_pcolor(x, y, z, cmap, xlab, ylab, title_str, out_dir, out_name, min_z, max_z, xlim=None, ylim=None,
                         cbar_label=None, plot_txt=None, save_plot=False):
    fig, ax = plt.subplots()
    cax = ax.pcolor(x, y, z, vmin=min_z, vmax=max_z,
                    cmap=cmap)
    if plot_txt:
        ax.text(plot_txt[0], plot_txt[1], plot_txt[2])
    cbar = fig.colorbar(cax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title_str)

    if save_plot:
        plt.savefig(
            os.path.join(out_dir, out_name),
            dpi=200)
    return


def plot_averages_pcolor_with_vector_field(x, y, z, cmap, xlab, ylab, title_str, out_dir, out_name, min_z, max_z,
                                           vec_df, x_col, y_col, u_col, v_col, xlim=None, ylim=None, plot_txt=None,
                                           cbar_label=None, save_plot=False):
    fig, ax = plt.subplots()
    cax = ax.pcolor(x, y, z, vmin=min_z, vmax=max_z, cmap=cmap)
    ax.quiver(vec_df[x_col], vec_df[y_col], vec_df[u_col], vec_df[v_col])
    if plot_txt:
        ax.text(plot_txt[0], plot_txt[1], plot_txt[2])
    cbar = fig.colorbar(cax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title_str)

    if save_plot:
        plt.savefig(
            os.path.join(out_dir, out_name),
            dpi=200)
    return


def plot_weekly_averages(weekly_data, day_starts, noon_s_midnight, xtick_labs, key_col, x, y, cmap, xlab, ylab,
                         title_str, min_z, max_z, out_dir, out_name, xlim=None, ylim=None, cbar_label=None,
                         save_plot=False):
    nWeeks = len(weekly_data)
    fig, ax = plt.subplots(nrows=nWeeks, ncols=1, figsize=(6.4 * 2.95, 4.8 * 2.0))
    for week in range(nWeeks):
        z = weekly_data['week_{}'.format(week)][key_col]
        if z is None:
            continue

        im = ax[week].pcolor(x, y, np.transpose(z), cmap=cmap, vmin=min_z, vmax=max_z)
        ax[week].set_xticks(noon_s_midnight)
        ax[week].set_xticklabels(xtick_labs[week])
        if xlim:
            ax[week].set_xlim(xlim)
        if ylim:
            ax[week].set_ylim(ylim)

        for day_start in day_starts:
            ax[week].axvline(x=day_start, color='k', linewidth=2, alpha=0.7)
            # if (day_start + 12) > 24*7:
            #     continue
            # ax[week].axvline(x=day_start + 12, color='c', linewidth = 2, alpha = 0.7)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlab, labelpad=10)
    plt.ylabel(ylab, labelpad=15)
    plt.tight_layout()

    fig.subplots_adjust(right=0.9, top=0.95)
    cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])  # (left, bottom, width, height)
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    plt.suptitle(title_str)

    if save_plot:
        plt.savefig(
            os.path.join(out_dir, out_name),
            dpi=200)

    return


def plot_weekly_averages_with_vector_field(weekly_data, day_starts, noon_s_midnight, xtick_labs, key_col, x, y, vec_df,
                                           x_col, y_col, u_col, v_col, cmap, xlab, ylab, title_str, min_z, max_z,
                                           out_dir, out_name, xlim=None, ylim=None, save_plot=False):
    assert "week" in vec_df.columns, "DataFrame must contain a week column."
    nWeeks = len(weekly_data)
    fig, ax = plt.subplots(nrows=nWeeks, ncols=1, figsize=(6.4 * 2.95, 4.8 * 2.0))
    for week in range(nWeeks):
        z = weekly_data['week_{}'.format(week)][key_col]
        if z is None:
            continue

        week_idx = vec_df["week"] == week

        im = ax[week].pcolor(x, y, np.transpose(z), cmap=cmap, vmin=min_z, vmax=max_z)
        ax[week].quiver(vec_df.loc[week_idx, x_col], vec_df.loc[week_idx, y_col], vec_df.loc[week_idx, u_col],
                        vec_df.loc[week_idx, v_col], width=0.0008, scale=0.5, scale_units='x')
        ax[week].set_xticks(noon_s_midnight)
        ax[week].set_xticklabels(xtick_labs[week])
        if xlim:
            ax[week].set_xlim(xlim)
        if ylim:
            ax[week].set_ylim(ylim)

        for day_start in day_starts:
            ax[week].axvline(x=day_start, color='k', linewidth=2, alpha=0.7)
            # if (day_start + 12) > 24*7:
            #     continue
            # ax[week].axvline(x=day_start + 12, color='c', linewidth = 2, alpha = 0.7)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()

    fig.subplots_adjust(right=0.9, top=0.95)
    cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])  # (left, bottom, width, height)
    fig.colorbar(im, cax=cbar_ax)
    plt.suptitle(title_str)

    if save_plot:
        plt.savefig(
            os.path.join(out_dir, out_name),
            dpi=200)

    return


def LoadWindError(airspeed_log_dir, airspeed_files, target_echoes, use_ins_height_profile, MAX_WEATHER_PROP,
                  MAX_WEATHER_PROP_SCAN, remove_cases_list=[]):
    wind_error = pd.DataFrame()
    for airspeed_file in airspeed_files:
        wind_error_path = os.path.join(airspeed_log_dir, airspeed_file)
        p_in = open(wind_error_path, 'rb')
        curr_wind_error, idx_last_log = pickle.load(p_in)
        p_in.close()
        wind_error = wind_error.append(curr_wind_error)

    if 'airspeed_birds' not in wind_error.columns:
        for echo in target_echoes:
            airspeed_fname = 'airspeed_{}'.format(GetVADMaskDescription(echo))
            fdirn_fname = 'fdirn_{}'.format(GetVADMaskDescription(echo))
            migspeed_fname = '{}_speed'.format(GetVADMaskDescription(echo))
            migdirn_fname = '{}_direction'.format(GetVADMaskDescription(echo))

            # Calculate flight vector.
            wind_error[airspeed_fname], wind_error[fdirn_fname] = CalcPolarDiffVec(
                spd1=wind_error[migspeed_fname],
                dirn1=wind_error[migdirn_fname],
                spd2=wind_error['wind_speed'],
                dirn2=wind_error[
                    'wind_direction'])

            # Direction offset from the wind.
            wind_error['{}_wind_offset'.format(GetVADMaskDescription(echo))] = CalcSmallAngleDirDiffDf(
                wind_error['{}_direction'.format(
                    GetVADMaskDescription(echo))], wind_error['wind_direction'])

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
    constraints = [("prop_weather", 0, MAX_WEATHER_PROP), ("prop_weather_scan", 0, MAX_WEATHER_PROP_SCAN),
                   ('file_name', remove_cases_list)]
    idx_constraints = ImposeConstraints(wind_error, constraints)
    wind_error_constrained = wind_error[idx_constraints].reset_index(drop=True)

    return wind_error_constrained, constraints


def VisualizeFlightspeeds(wind_error, constraints, color_info, c_group, save_plots, figure_summary_dir,
                          plot_title_suffix, out_name_suffix, max_airspeed=None, show_plots=True,
                          generate_weekly_month_profiles=True):
    delta_insect_prop = Constants.DELTA_INSECT_PROP
    delta_height = Constants.DELTA_HEIGHT
    delta_time_hour = Constants.DELTA_TIME_HR
    impurity_tolerance = Constants.IMPURITY_TOLERANCE

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
    delta_insect_prop_part = Constants.DELTA_INSECT_PROP_PART
    delta_height_part = Constants.DELTA_HEIGHT_PART
    insect_prop_part = {i: (i * delta_insect_prop_part, (i + 1) * delta_insect_prop_part) for i in range(3)}
    height_part = {i: (i * delta_height_part, (i + 1) * delta_height_part) for i in range(3)}

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

    # Plot 4. Pure birds vs pure insects airspeeds.
    # impurity_tolerance = 30
    num_bins = 50
    idx_all_birds = wind_error['insect_prop_bio'] < impurity_tolerance
    idx_all_insects = wind_error['insect_prop_bio'] > (100 - impurity_tolerance)

    fig, ax = plt.subplots(3, 1, figsize=(6.4 * 1.14, 4.8 * 2))
    for height_bin in range(0, 3):
        idx_height_part = np.logical_and(wind_error['height_m'] >= height_part[height_bin][0],
                                         wind_error['height_m'] < height_part[height_bin][1])

        idx_pure_insects = np.logical_and(idx_height_part, idx_all_insects)
        idx_pure_birds = np.logical_and(idx_height_part, idx_all_birds)

        if np.sum(idx_pure_insects) > 0:
            ax[height_bin].hist(x=wind_error['airspeed_insects'][idx_pure_insects], color='red', label='insects',
                                alpha=0.3, density=True, bins=num_bins)
            print("median airspeed insects: {} mps".format(
                np.nanmedian(wind_error['airspeed_insects'][idx_pure_insects])))

        if np.sum(idx_pure_birds) > 0:
            ax[height_bin].hist(x=wind_error['airspeed_birds'][idx_pure_birds], color='blue', label='birds', alpha=0.3,
                                density=True, bins=num_bins)
            print("median airspeed birds: {} mps".format(np.nanmedian(wind_error['airspeed_birds'][idx_pure_birds])))
        ax[height_bin].grid(True)
        ax[height_bin].set_title(
            "Height, {} - {} m".format(height_part[height_bin][0],
                                       height_part[height_bin][1]))
        ax[height_bin].legend()

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Airspeed (m/s)")
    plt.ylabel("Count (no units)")
    # plt.suptitle("Airspeed for single specie scans. {}% impurity.".format(impurity_threshold))
    plt.tight_layout()

    plt.subplots_adjust(top=0.9)
    # plt.suptitle("Airspeed for single specie scans. {}% impurity.".format(impurity_threshold))
    plt.suptitle("Comparison of wind biases from mass bird and insect migration")

    if save_plots:
        plt.savefig(os.path.join(figure_summary_dir, "".join(["single_specie_airspeeds_", out_name_suffix, ".png"])),
                    dpi=200)

    ####################################################################################################################
    # Plot: Pure bird vs insect migration direction.
    # TODO(pjatau) Refactor?.
    fig, ax = plt.subplots(3, 1, figsize=(6.4 * 1.14, 4.8 * 2))
    for height_bin in range(0, 3):
        idx_height_part = np.logical_and(wind_error['height_m'] >= height_part[height_bin][0],
                                         wind_error['height_m'] < height_part[height_bin][1])

        idx_pure_insects = np.logical_and(idx_height_part, idx_all_insects)
        idx_pure_birds = np.logical_and(idx_height_part, idx_all_birds)

        ax[height_bin].hist(x=wind_error['wind_direction'], color='green', label='wind', alpha=0.6,
                            density=True, bins=num_bins)

        if np.sum(idx_pure_insects) > 0:
            ax[height_bin].hist(x=wind_error['insects_direction'][idx_pure_insects], color='red', label='insects',
                                alpha=0.3, density=True, bins=num_bins)

        if np.sum(idx_pure_birds) > 0:
            ax[height_bin].hist(x=wind_error['birds_direction'][idx_pure_birds], color='blue', label='birds', alpha=0.3,
                                density=True, bins=num_bins)

        ax[height_bin].set_xlim(0, 360)
        ax[height_bin].grid(True)
        ax[height_bin].set_title(
            r"Height, {} - {} m".format(
                height_part[height_bin][0],
                height_part[height_bin][1]))
        ax[height_bin].legend()

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Migration direction ($^\circ$)")
    plt.ylabel("Count (no units)")
    plt.tight_layout()

    plt.subplots_adjust(top=0.9)
    plt.suptitle(r"Migration direction $\alpha_r$. {}% impurity".format(impurity_tolerance))

    if save_plots:
        plt.savefig(os.path.join(figure_summary_dir, "".join(["single_specie_mig_dir_", out_name_suffix, ".png"])),
                    dpi=200)
    ####################################################################################################################

    ####################################################################################################################
    # Plot: Pure bird vs insect migration direction offset from the wind.
    # TODO(pjatau) Refactor?.
    fig, ax = plt.subplots(3, 1, figsize=(6.4 * 1.14, 4.8 * 2))
    for height_bin in range(0, 3):
        idx_height_part = np.logical_and(wind_error['height_m'] >= height_part[height_bin][0],
                                         wind_error['height_m'] < height_part[height_bin][1])

        idx_pure_insects = np.logical_and(idx_height_part, idx_all_insects)
        idx_pure_birds = np.logical_and(idx_height_part, idx_all_birds)

        if np.sum(idx_pure_insects) > 0:
            ax[height_bin].hist(x=wind_error['insects_wind_offset'][idx_pure_insects], color='red', label='insects',
                                alpha=0.3, density=True, bins=num_bins * 3)
            print("modal offset insects: {} mps".format(
                stats.mode(round(wind_error['insects_wind_offset'][idx_pure_insects]))))

        if np.sum(idx_pure_birds) > 0:
            ax[height_bin].hist(x=wind_error['birds_wind_offset'][idx_pure_birds], color='blue', label='birds',
                                alpha=0.3,
                                density=True, bins=num_bins * 3)
            print("modal offset birds: {} mps".format(
                stats.mode(round(wind_error['birds_wind_offset'][idx_pure_birds]))))
        ax[height_bin].set_xlim(-45, 45)
        ax[height_bin].grid(True)
        ax[height_bin].set_title(
            r"Height, {} - {} m".format(
                height_part[height_bin][0],
                height_part[height_bin][1]))
        ax[height_bin].legend()

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(r"$\alpha_r - \alpha_w$ ($^\circ$)")
    plt.tight_layout()

    plt.subplots_adjust(top=0.9)
    plt.suptitle(r"Direction offset $\alpha_r - \alpha_w$. {}% impurity".format(impurity_tolerance))

    if save_plots:
        plt.savefig(
            os.path.join(figure_summary_dir, "".join(["single_specie_offset_mig_dir_", out_name_suffix, ".png"])),
            dpi=200)
    ####################################################################################################################

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

    # TODO Refactor below code block.
    # Plot 7: Average flight speed vs % insects (birds).
    airspeed_df = wind_error[["insect_prop_bio", "airspeed_birds", "airspeed_insects", "height_m"]]
    airspeed_df["height_bins"] = airspeed_df["height_m"] // delta_height_part

    airspeed_df["insect_prop_bins"] = airspeed_df[
                                          "insect_prop_bio"] // delta_insect_prop * delta_insect_prop + delta_insect_prop / 2
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

    # Height vs insect prop vs data analysis.
    height_ip_df = wind_error[
        ["insect_prop_bio", "airspeed_birds", "airspeed_insects", "airspeed_biological", "height_m",
         "prop_weather_scan", "biological_wind_offset"]]
    height_ip_df["abs_bio_wind_offset"] = np.abs(height_ip_df["biological_wind_offset"])
    height_ip_df['airspeed_diff'] = height_ip_df['airspeed_birds'] - height_ip_df['airspeed_insects']
    height_ip_df['height_bins'] = height_ip_df['height_m'] // delta_height * delta_height + delta_height / 2
    height_ip_df["insect_prop_bins"] = height_ip_df[
                                           "insect_prop_bio"] // delta_insect_prop * delta_insect_prop + delta_insect_prop / 2

    # Bias bio vs % birds scatterplot.
    lb_height = 350
    height_ip_df["bird_prop_bins"] = 100 - height_ip_df["insect_prop_bins"]
    height_ip_df["bird_prop_bio"] = 100 - height_ip_df["insect_prop_bio"]

    bias_df = height_ip_df.loc[height_ip_df["height_m"] > lb_height, :]
    bias_df = height_ip_df

    plt.figure(figsize=(6.4 * 1.2, 4.8 * 1.2))
    plt.grid(True)
    plt.scatter(bias_df['bird_prop_bio'], bias_df['airspeed_biological'], s=4, alpha=0.2)
    plt.xlabel("Proportion of predicted birds [%]")
    plt.ylabel("Biases from bio VAD [m/s]")
    plt.title("Biases from bio VAD. Height > {} m".format(lb_height))
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(figure_summary_dir, "bias_bio_scatter.png"), dpi=200)
    ####################################################################################################################

    height_ip_df = height_ip_df.groupby(["height_bins", "bird_prop_bins"], as_index=False).mean()

    unique_insect_prop_bins = np.arange(delta_insect_prop / 2, 100, delta_insect_prop)
    unique_height_bins = np.arange(delta_height / 2, 1000, delta_height)
    z_dict = {"airspeed_diff": height_ip_df['airspeed_diff'],
              'airspeed_biological': height_ip_df['airspeed_biological'],
              'airspeed_insects': height_ip_df['airspeed_insects']}  # monthly averages

    unique_height_bins, ins_prop_bins, height_ip_grid = prepare_pcolor_grid_from_series(height_ip_df['height_bins'],
                                                                                        height_ip_df[
                                                                                            'insect_prop_bins'],
                                                                                        z_dict,
                                                                                        uniqueX=unique_height_bins,
                                                                                        uniqueY=unique_insect_prop_bins)

    # Plot of height vs insect prop vs airspeed difference
    thresholds = (-Constants.BI_BIAS_DIFF_THRESHOLD, Constants.BI_BIAS_DIFF_THRESHOLD)

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

    plot_averages_pcolor(x=ins_prop_bins, y=unique_height_bins, z=height_ip_grid['airspeed_diff'], cmap='jet',
                         xlab='insect prop bio [%]', ylab='height [m]', title_str=title_str,
                         out_dir=figure_summary_dir, out_name="airspeed_difference.png", min_z=-max_amp,
                         max_z=max_amp, xlim=(0, 100), ylim=(0, 1000), plot_txt=(55, 800, info_str), cbar_label="[m/s]",
                         save_plot=save_plots)

    # Plot of height vs insect prop vs airspeed_bio
    max_airspeed_bio = np.max(np.abs(height_ip_df['airspeed_biological']))
    print("max_airspeed_bio: ", max_airspeed_bio)

    height_ip_df["abs_bio_off_U"], height_ip_df["abs_bio_off_V"] = \
        Polar2CartesianComponentsDf(spd=0.5, dirn=height_ip_df["abs_bio_wind_offset"])

    title_str = "Averaged biological echo biases"
    plot_averages_pcolor_with_vector_field(x=ins_prop_bins, y=unique_height_bins,
                                           z=height_ip_grid['airspeed_biological'], cmap='jet',
                                           xlab='insect prop bio [%]', ylab='height [m]', title_str=title_str,
                                           out_dir=figure_summary_dir,
                                           out_name="airspeed_biological_height_insectprop.png", min_z=0,
                                           max_z=max_airspeed_bio, vec_df=height_ip_df, x_col="insect_prop_bins",
                                           y_col="height_bins", u_col="abs_bio_off_U", v_col="abs_bio_off_V",
                                           xlim=(0, 100), ylim=(0, 1000), cbar_label="[m/s]",
                                           save_plot=save_plots)

    # Plot of height vs insect prop vs airspeed_insects
    max_airspeed_ins = np.max(np.abs(height_ip_df['airspeed_insects']))
    print("max_airspeed_ins: ", max_airspeed_ins)

    plot_averages_pcolor(x=ins_prop_bins, y=unique_height_bins, z=height_ip_grid['airspeed_insects'], cmap='jet',
                         xlab='insect prop bio [%]', ylab='height [m]', title_str=r"$bias_{insects}$",
                         out_dir=figure_summary_dir, out_name="airspeed_ins_height_insectprop.png", min_z=0,
                         max_z=max_airspeed_bio, xlim=(0, 100), ylim=(0, 1000), cbar_label="[m/s]",
                         save_plot=save_plots)

    ####################################################################################################################
    # Height vs wind dev. from mig dir vs airspeed
    # TODO(pjatau) Refactor constants or delete code block
    delta_alpha = 30
    bird_mig_dirn = np.nanmedian(wind_error.loc[idx_all_birds, "birds_direction"])

    height_wind_dev_df = wind_error[
        ["insect_prop_bio", "airspeed_birds", "airspeed_insects", "airspeed_biological", "height_m", "wind_direction"]]
    idx = height_wind_dev_df["insect_prop_bio"] < 30
    height_wind_dev_df = height_wind_dev_df[idx]

    height_wind_dev_df['height_bins'] = height_wind_dev_df['height_m'] // delta_height * delta_height + delta_height / 2
    height_wind_dev_df["dev_from_mig_direction"] = CalcSmallAngleDirDiffDf(height_wind_dev_df["wind_direction"],
                                                                           bird_mig_dirn)
    sign_dev = np.sign(height_wind_dev_df["dev_from_mig_direction"])
    height_wind_dev_df['dev_mig_dirn_bins'] = np.abs(height_wind_dev_df["dev_from_mig_direction"]) // \
                                              delta_alpha * delta_alpha + delta_alpha / 2
    height_wind_dev_df['dev_mig_dirn_bins'] *= sign_dev

    count_df = height_wind_dev_df.loc[:, ["height_bins", "dev_mig_dirn_bins", "airspeed_birds"]]
    count_df = count_df.groupby(["height_bins", "dev_mig_dirn_bins"], as_index=False).count()
    count_df.rename(columns={"airspeed_birds": "airspeed_birds_count"}, inplace=True)
    height_wind_dev_df = height_wind_dev_df.groupby(["height_bins", "dev_mig_dirn_bins"], as_index=False).mean()
    height_wind_dev_df = pd.merge(height_wind_dev_df, count_df, on=["height_bins", "dev_mig_dirn_bins"], how="inner")

    unique_dev_mig_bins = np.arange(delta_alpha / 2 - 180, 180, delta_alpha)
    z_dict = {"airspeed_birds": height_wind_dev_df['airspeed_birds'],
              "airspeed_birds_count": height_wind_dev_df["airspeed_birds_count"]}
    unique_height_bins, unique_dev_mig_bins, height_dev_grid = prepare_pcolor_grid_from_series(
        height_wind_dev_df['height_bins'],
        height_wind_dev_df[
            'dev_mig_dirn_bins'],
        z_dict,
        uniqueX=unique_height_bins,
        uniqueY=unique_dev_mig_bins)

    title_str = r"Birds flight speed compensation"
    plot_averages_pcolor(x=unique_dev_mig_bins, y=unique_height_bins, z=height_dev_grid['airspeed_birds'], cmap='jet',
                         xlab=r'$\alpha_{w} - \alpha_{mig} [^\circ$]', ylab='height [m]', title_str=title_str,
                         out_dir=figure_summary_dir, out_name="flight_comp_airspeed.png", min_z=None,
                         max_z=None, xlim=None, ylim=None, cbar_label="[m/s]", save_plot=save_plots)

    title_str = r"Birds flight compensation counts"
    plot_averages_pcolor(x=unique_dev_mig_bins, y=unique_height_bins, z=height_dev_grid['airspeed_birds_count'],
                         cmap='RdYlBu',
                         xlab=r'$\alpha_{w} - \alpha_{mig} [^\circ$]', ylab='height [m]', title_str=title_str,
                         out_dir=figure_summary_dir, out_name="flight_comp_counts.png", min_z=None,
                         max_z=None, xlim=None, ylim=None, cbar_label="[no units]", save_plot=save_plots)
    ####################################################################################################################

    # Plot: insect prop vs time of day vs height
    echo_profile_df = wind_error.loc[:,
                      ["month", "day", "time_hour", "insect_prop_bio", "height_m", "prop_weather_scan",
                       "num_insects_height", "num_birds_height", "biological_speed", "biological_direction",
                       "airspeed_biological", "biological_wind_offset"]]
    echo_profile_df["height_bins"] = echo_profile_df['height_m'] // delta_height * delta_height + delta_height / 2
    echo_profile_df["time_hour_bins"] = echo_profile_df[
                                            'time_hour'] // delta_time_hour * delta_time_hour + delta_time_hour / 2

    # Plot: Insect profile per day
    ip_day = echo_profile_df[["day", "height_bins", "time_hour_bins", "insect_prop_bio"]]
    ip_day = ip_day.groupby(["day", "height_bins", "time_hour_bins"], as_index=False).mean()

    echoProfileDays = [1, 2, 3, 8, 9, 13, 15, 20, 24]
    unique_time_hr = np.arange(delta_time_hour / 2, 24, delta_time_hour)

    nRows = 3
    nCols = 3
    fig, ax = plt.subplots(nRows, nCols, figsize=(6.4 * 2.8, 4.8 * 2))

    for i_day in range(len(echoProfileDays)):
        day = echoProfileDays[i_day]
        day_idx = ip_day["day"] == day
        if np.sum(day_idx) == 0:
            continue
        z_dict = {"insect_prop_bio": ip_day['insect_prop_bio'][day_idx]}
        time_hr_day, heights_day, insect_prop_day = prepare_pcolor_grid_from_series(ip_day['time_hour_bins'][day_idx],
                                                                                    ip_day['height_bins'][day_idx],
                                                                                    z_dict,
                                                                                    uniqueX=unique_time_hr,
                                                                                    uniqueY=unique_height_bins)
        im = ax[int((i_day) // nCols), int((i_day) % nCols)].pcolor(time_hr_day, heights_day,
                                                                    np.transpose(insect_prop_day["insect_prop_bio"]),
                                                                    cmap='jet', vmin=0,
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
    cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])  # (left, bottom, width, height)
    fig.colorbar(im, cax=cbar_ax)

    if save_plots:
        plt.savefig(
            os.path.join(figure_summary_dir, "insect_prop_height_timeday.png"),
            dpi=200)

    # Height vs time of day vs data analysis.
    height_time_df = echo_profile_df.loc[:,
                     ["height_bins", "time_hour_bins", "insect_prop_bio", "num_insects_height", "num_birds_height",
                      "airspeed_biological", "biological_wind_offset"]]
    height_time_df["biological_wind_offset"] = np.abs(height_time_df["biological_wind_offset"])
    height_time_df["bird_prop_bio"] = 100 - height_time_df["insect_prop_bio"]
    height_time_grouped = height_time_df.groupby(["height_bins", "time_hour_bins"], as_index=False).mean()

    max_num = max(np.max(height_time_grouped["num_insects_height"]), np.max(height_time_grouped["num_birds_height"]))
    height_time_grouped["num_insects_height"] /= max_num
    height_time_grouped["num_birds_height"] /= max_num

    z_dict = {'insect_prop_bio': height_time_grouped['insect_prop_bio'],
              'num_birds_height': height_time_grouped['num_birds_height'],
              'num_insects_height': height_time_grouped['num_insects_height'],
              'airspeed_biological': height_time_grouped['airspeed_biological'],
              'bird_prop_bio': height_time_grouped['bird_prop_bio']}
    time_hr_bins, unique_height_bins, height_time_grid = prepare_pcolor_grid_from_series(
        height_time_grouped['time_hour_bins'],
        height_time_grouped['height_bins'],
        z_dict, uniqueX=unique_time_hr,
        uniqueY=unique_height_bins)

    # Plot: Averaged insect profile
    title_str = "Averaged % insects relative to biological echoes"
    plot_averages_pcolor(x=time_hr_bins, y=unique_height_bins, z=np.transpose(height_time_grid['insect_prop_bio']),
                         cmap='jet',
                         xlab='Time [UTC]', ylab='Height [m]', title_str=title_str,
                         out_dir=figure_summary_dir, out_name="averaged_insect_prop_height_timeday.png", min_z=0,
                         max_z=100, xlim=(0, 24), ylim=(0, 1000), cbar_label="[%]", save_plot=save_plots)

    title_str = "Averaged proportion of predicted birds"
    plot_averages_pcolor(x=time_hr_bins, y=unique_height_bins, z=np.transpose(height_time_grid['bird_prop_bio']),
                         cmap='RdYlBu',
                         xlab='Time [UTC]', ylab='Height [m]', title_str=title_str,
                         out_dir=figure_summary_dir, out_name="averaged_bird_prop_height_timeday.png", min_z=0,
                         max_z=100, xlim=(0, 24), ylim=(0, 1000), cbar_label="[%]", save_plot=save_plots)

    # Plot: Averaged bird population
    title_str = "Averaged birds population"
    plot_averages_pcolor(x=time_hr_bins, y=unique_height_bins, z=np.transpose(height_time_grid['num_birds_height']),
                         cmap='RdYlBu',
                         xlab='Time [UTC]', ylab='Height [m]', title_str=title_str,
                         out_dir=figure_summary_dir, out_name="averaged_birds_population_height_timeday.png", min_z=0,
                         max_z=1, xlim=(0, 24), ylim=(0, 1000), cbar_label="[no units]", save_plot=save_plots)

    # Plot: Averaged bird population
    title_str = "Averaged insects population."
    plot_averages_pcolor(x=time_hr_bins, y=unique_height_bins, z=np.transpose(height_time_grid['num_insects_height']),
                         cmap='RdYlBu',
                         xlab='Time [UTC]', ylab='Height [m]', title_str=title_str,
                         out_dir=figure_summary_dir, out_name="averaged_insects_population_height_timeday.png", min_z=0,
                         max_z=1, xlim=(0, 24), ylim=(0, 1000), cbar_label="[no units]", save_plot=save_plots)

    ####################################################################################################################
    # Plot: Averaged flight vel x time x height. Direction is the absolute offset from the wind.
    height_time_grouped["abs_bio_off_U"], height_time_grouped["abs_bio_off_V"] = \
        Polar2CartesianComponentsDf(spd=0.5, dirn=height_time_grouped["biological_wind_offset"])

    title_str = "Biases from VAD on biological echoes"
    plot_averages_pcolor_with_vector_field(x=time_hr_bins, y=unique_height_bins,
                                           z=np.transpose(height_time_grid['airspeed_biological']), cmap='jet',
                                           xlab='Time [UTC]',
                                           ylab='Height [m]', title_str=title_str, out_dir=figure_summary_dir,
                                           out_name="averaged_flightvel_height_timeday.png", min_z=None, max_z=None,
                                           vec_df=height_time_grouped, x_col="time_hour_bins", y_col="height_bins",
                                           u_col="abs_bio_off_U", v_col="abs_bio_off_V", xlim=(0, 24), ylim=(0, 1000),
                                           plot_txt=None, cbar_label="[m/s]",
                                           save_plot=save_plots)
    ####################################################################################################################

    # Plot: weekly profiles for whole month
    if generate_weekly_month_profiles:
        population_df = wind_error.loc[:,
                        ["month", "day", "time_hour", "insect_prop_bio", "height_m", "prop_weather_scan",
                         "num_insects_height", "num_birds_height", "airspeed_biological", "biological_wind_offset",
                         "biological_speed", "biological_direction", "wind_speed", "wind_direction"]]

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

        unique_time_week, unique_height_bins, weekly_data, xlabels = prepare_weekly_data_for_pcolor_plot(
            key_cols=['num_birds_height', 'num_insects_height', 'insect_prop_bio', 'airspeed_biological',
                      'biological_speed', 'wind_speed'],
            x_col_name='time_hour_week',
            y_col_name='height_bins',
            in_data=population_grouped_df,
            month=month,
            noon_s_midnight=noon_s_midnight,
            uniqueX=unique_time_week,
            uniqueY=unique_height_bins)

        # Plot for number of birds gates
        title_str = "Birds population"
        plot_weekly_averages(weekly_data=weekly_data, day_starts=day_starts, noon_s_midnight=noon_s_midnight,
                             xtick_labs=xlabels,
                             key_col='num_birds_height', x=unique_time_week, y=unique_height_bins,
                             min_z=0, max_z=1, xlab="Time [UTC]", ylab="Height [m]", cmap='RdYlBu',
                             title_str=title_str, out_dir=figure_summary_dir,
                             out_name="bird_population_height_timeweek.png",
                             xlim=None,
                             ylim=(0, 1000), cbar_label="[no units]", save_plot=save_plots)

        # Plot for number of insect gates
        title_str = "Insects population"
        plot_weekly_averages(weekly_data=weekly_data, day_starts=day_starts, noon_s_midnight=noon_s_midnight,
                             xtick_labs=xlabels,
                             key_col='num_insects_height', x=unique_time_week, y=unique_height_bins,
                             min_z=0, max_z=1, xlab="Time [UTC]", ylab="Height [m]", cmap='RdYlBu',
                             title_str=title_str, out_dir=figure_summary_dir,
                             out_name="insect_population_height_timeweek.png",
                             xlim=None,
                             ylim=(0, 1000), cbar_label="[no units]", save_plot=save_plots)

        # Plot for insect_prop_bio
        title_str = "% insects relative to biological echoes"
        plot_weekly_averages(weekly_data=weekly_data, day_starts=day_starts, noon_s_midnight=noon_s_midnight,
                             xtick_labs=xlabels,
                             key_col='insect_prop_bio', x=unique_time_week, y=unique_height_bins,
                             min_z=0, max_z=100, xlab="Time [UTC]", ylab="Height [m]", cmap='jet',
                             title_str=title_str, out_dir=figure_summary_dir,
                             out_name="insect_prop_bio_height_timeweek.png",
                             xlim=None,
                             ylim=(0, 1000), cbar_label="[%]", save_plot=save_plots)

        # Plot for airspeed_bio height profile for whole month.
        title_str = "Biases from biological echoes"
        plot_weekly_averages(weekly_data=weekly_data, day_starts=day_starts, noon_s_midnight=noon_s_midnight,
                             xtick_labs=xlabels,
                             key_col='airspeed_biological', x=unique_time_week, y=unique_height_bins,
                             min_z=None, max_z=None, xlab="Time [UTC]", ylab="Height [m]", cmap='jet',
                             title_str=title_str, out_dir=figure_summary_dir,
                             out_name="airspeed_biological_height_timeweek.png",
                             xlim=None,
                             ylim=(0, 1000), cbar_label="[m/s]", save_plot=save_plots)

        ################################################################################################################
        # Plot for wind vec height profile for whole month.
        population_grouped_df["wind_U"], population_grouped_df["wind_V"] = Polar2CartesianComponentsDf(spd=0.5, dirn=
        population_grouped_df["wind_direction"])
        max_speed = max(np.nanmax(population_grouped_df["biological_speed"]),
                        np.nanmax(population_grouped_df["wind_speed"]))

        plot_weekly_averages_with_vector_field(weekly_data=weekly_data, day_starts=day_starts,
                                               noon_s_midnight=noon_s_midnight, xtick_labs=xlabels,
                                               key_col='wind_speed', x=unique_time_week, y=unique_height_bins,
                                               vec_df=population_grouped_df,
                                               x_col="time_hour_week", y_col="height_bins", u_col="wind_U",
                                               v_col="wind_V", cmap='jet', xlab="Time [UTC]", ylab="Height [m]",
                                               title_str=r"$Wind \, velocity$", min_z=0, max_z=max_speed,
                                               out_dir=figure_summary_dir,
                                               out_name="wind_velocity_height_timeweek.png", xlim=None, ylim=(0, 1000),
                                               save_plot=save_plots)

        ################################################################################################################
        # Plot for mig vec height profile for whole month.
        population_grouped_df["bio_U"], population_grouped_df["bio_V"] = Polar2CartesianComponentsDf(spd=0.5, dirn=
        population_grouped_df["biological_direction"])

        plot_weekly_averages_with_vector_field(weekly_data=weekly_data, day_starts=day_starts,
                                               noon_s_midnight=noon_s_midnight, xtick_labs=xlabels,
                                               key_col='biological_speed', x=unique_time_week, y=unique_height_bins,
                                               vec_df=population_grouped_df,
                                               x_col="time_hour_week", y_col="height_bins", u_col="bio_U",
                                               v_col="bio_V", cmap='jet', xlab="Time [UTC]", ylab="Height [m]",
                                               title_str="Radial velocity, biological.", min_z=0, max_z=max_speed,
                                               out_dir=figure_summary_dir,
                                               out_name="biological_vel_height_timeweek.png", xlim=None, ylim=(0, 1000),
                                               save_plot=save_plots)

        ################################################################################################################
        # Plot for flight vec height profile for whole month.
        bio_off_from_north = (population_grouped_df["biological_wind_offset"] + 360) % 360
        print(np.max(bio_off_from_north))
        print(np.min(bio_off_from_north))
        population_grouped_df["bio_off_U"], population_grouped_df["bio_off_V"] = Polar2CartesianComponentsDf(
            spd=0.5, dirn=bio_off_from_north)

        plot_weekly_averages_with_vector_field(weekly_data=weekly_data, day_starts=day_starts,
                                               noon_s_midnight=noon_s_midnight, xtick_labs=xlabels,
                                               key_col='airspeed_biological', x=unique_time_week, y=unique_height_bins,
                                               vec_df=population_grouped_df,
                                               x_col="time_hour_week", y_col="height_bins", u_col="bio_off_U",
                                               v_col="bio_off_V", cmap='jet', xlab="Time [UTC]", ylab="Height [m]",
                                               title_str="Flight velocity, biological.", min_z=0, max_z=None,
                                               out_dir=figure_summary_dir,
                                               out_name="flight_vel_bio_height_timeweek.png", xlim=None, ylim=(0, 1000),
                                               save_plot=save_plots)

    if show_plots:
        plt.show()
    return


def Main():
    # Inputs
    airspeed_log_dir = r'./batch_analysis_logs'
    airspeed_files = ['debug_parameter_tuning/KOHX_20180501_20180531_weights_0_threshold_50.pkl']
    # airspeed_files = ['KOHX_20180516_20180531_launched_202329_11/KOHX_20180516_20180531.pkl',
    #                   'KOHX_20180501_20180515_launched_202328_19/KOHX_20180501_20180515.pkl']
    # airspeed_files = ['KOHX_20180501_20180515_launched_2023118_16\KOHX_20180501_20180515.pkl',
    #                   'KOHX_20180516_20180531_launched_2023119_10\KOHX_20180516_20180531.pkl']

    experiment_id = "post_processing_default"
    correct_hca_weather = True
    use_ins_height_profile = True
    MAX_WEATHER_PROP = 10  # 10
    MAX_WEATHER_PROP_SCAN = 5  # 5
    remove_cases_list = []
    gt_wind_source = WindSource.rap_130
    target_echoes = [VADMask.birds, VADMask.insects, VADMask.biological]

    figure_dir = "./figures"
    plot_title_suffix = "May 1 - 31, 2018"
    out_name_suffix = "May_1_31_2018"
    save_plots = False  # False
    generate_weekly_month_profiles = True  # False

    wind_source_desc = GetWindSourceDescription(gt_wind_source)
    wind_source_desc = wind_source_desc.replace(' ', '_')

    # Load wind error
    wind_error_constrained, constraints = LoadWindError(airspeed_log_dir=airspeed_log_dir,
                                                        airspeed_files=airspeed_files,
                                                        target_echoes=target_echoes,
                                                        use_ins_height_profile=use_ins_height_profile,
                                                        MAX_WEATHER_PROP=MAX_WEATHER_PROP,
                                                        MAX_WEATHER_PROP_SCAN=MAX_WEATHER_PROP_SCAN,
                                                        remove_cases_list=remove_cases_list)

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
    # constraints = [("height_m", 700, 1000), ("insect_prop_bio", 33.33, 66.66), ("airspeed_insects", 12.5, 19),
    #                ("airspeed_birds", 12, 22), ("prop_weather", 0, MAX_WEATHER_PROP),
    #                ("prop_weather_scan", 0, MAX_WEATHER_PROP_SCAN)]
    constraints = [("height_m", 732, 734)]
    wind_error_filt = FilterFlightspeeds(wind_error_constrained, constraints)
    print(np.unique(wind_error_filt.file_name))

    return


# Main()
if __name__ == "__main__":
    Main()
