import os
import pickle
import numpy as np
from VADMaskEnum import VADMask
import matplotlib.pyplot as plt
import pandas as pd

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 11}
plt.rc('font', **font)


def WindError(x1, y1, x2, y2, error_fn, reduce_fn):
    mapper = {}
    for j in range(len(x2)):
        mapper[x2[j]] = j

    x_scores = x1
    scores = []
    for i in range(len(x1)):
        if x1[i] in mapper.keys():
            ind = mapper[x1[i]]
            error = error_fn(y1[i], y2[ind])
            scores.append(error)
        else:
            scores.append(np.nan)

    reduced_error = reduce_fn(scores)
    return reduced_error, scores, x_scores


def GetAirSpeedForScan(wind_file, wind_files_parent, error_fn, reduce_fn, debug_plots=False, figure_dir=None,
                       save_plots=True):
    def GetAirSpeeds(delta_U, delta_V):
        airspeeds = []
        for i in range(len(delta_U)):
            airspeeds.append(np.sqrt(delta_U[i] ** 2 + delta_V[i] ** 2))
        return airspeeds

    wind_file_no_ext = os.path.splitext(wind_file)[0]

    figure_dir = os.path.join(figure_dir, wind_file_no_ext[:12])
    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)

    with open(os.path.join(wind_files_parent, wind_file), 'rb') as w_in:
        wind_result = pickle.load(w_in)
    w_in.close()

    vad = wind_result['VAD']
    sounding_df = wind_result['Sounding']
    echo_dist = wind_result['echo_dist']

    if sounding_df is None:
        return None

    # Sounding.
    x_sound = np.array(sounding_df['HGHT'])
    y_sound_U = np.array(sounding_df['windU'])
    y_sound_V = np.array(sounding_df['windV'])

    # Birds.
    x_birds = np.array(vad[VADMask.birds]['height'])
    y_birds_U = np.array(vad[VADMask.birds]['wind_U'])
    y_birds_V = np.array(vad[VADMask.birds]['wind_V'])

    # Insects
    x_insects = np.array(vad[VADMask.insects]['height'])
    y_insects_U = np.array(vad[VADMask.insects]['wind_U'])
    y_insects_V = np.array(vad[VADMask.insects]['wind_V'])

    # Sounding v bird
    err_sound_birds_U, scores_sound_birds_U, x_sound_birds_U = WindError(x_sound, y_sound_U, x_birds, y_birds_U,
                                                                         error_fn, reduce_fn)
    err_sound_birds_V, scores_sound_birds_V, x_sound_birds_V = WindError(x_sound, y_sound_V, x_birds, y_birds_V,
                                                                         error_fn, reduce_fn)
    airspeed_birds = GetAirSpeeds(scores_sound_birds_U, scores_sound_birds_V)

    # Sounding v insects
    err_sound_insects_U, scores_sound_insects_U, x_sound_insects_U = WindError(x_sound, y_sound_U, x_insects,
                                                                               y_insects_U, error_fn, reduce_fn)
    err_sound_insects_V, scores_sound_insects_V, x_sound_insects_V = WindError(x_sound, y_sound_V, x_insects,
                                                                               y_insects_V, error_fn, reduce_fn)
    airspeed_insects = GetAirSpeeds(scores_sound_insects_U, scores_sound_insects_V)

    if debug_plots:
        plt.figure()
        plt.scatter(airspeed_birds, x_sound_birds_U, color="blue", alpha=0.4)
        plt.plot(airspeed_birds, x_sound_birds_U, color="blue", label="birds")
        plt.scatter(airspeed_insects, x_sound_insects_U, color="red", alpha=0.4)
        plt.plot(airspeed_insects, x_sound_insects_U, color="red", label="insects")
        plt.xlim(0, 12)
        plt.ylim(0, 900)
        plt.grid(True)
        plt.xlabel("flight speed [m/s]")
        plt.ylabel("height [m]")
        plt.title(wind_file_no_ext)
        plt.legend()
        if save_plots:
            plt.savefig(os.path.join(figure_dir, "".join([wind_file_no_ext, '.png'])))
        plt.show()
        plt.close()

    return pd.DataFrame({'file_name': wind_file_no_ext, 'airspeed_birds': airspeed_birds,
                         'airspeed_insects': airspeed_insects, 'height_m': x_sound_insects_U,
                         'prop_birds': echo_dist['bird'],
                         'prop_insects': echo_dist['insects'],
                         'prop_weather': echo_dist['weather']})


def GetAirSpeedBatch(wind_files_parent, error_fn, reduce_fn, figure_dir, debug_plots, save_plots=False):
    wind_files = os.listdir(wind_files_parent)
    wind_error_df = pd.DataFrame(
        columns=['file_name', 'airspeed_birds', 'airspeed_insects', 'height_m', 'prop_birds', 'prop_insects',
                 'prop_weather'])

    # Get wind here.
    for wind_file in wind_files:
        entry = GetAirSpeedForScan(wind_file, wind_files_parent, error_fn, reduce_fn, debug_plots=debug_plots,
                                   save_plots=save_plots,
                                   figure_dir=figure_dir)
        if entry is not None:
            wind_error_df = wind_error_df.append(entry, ignore_index=True)
    return wind_error_df


def GetAirSpeedManyBatches(wind_dir, batch_folders, experiment_id, echo_count_dir, log_dir, figure_dir,
                           plot_title_suffix, out_name_suffix, save_plots=False, force_airspeed_analysis=False):
    echo_count_batch_folders = []
    for folder in batch_folders:
        echo_count_batch_folders.append(folder[:22])

    error_fn = lambda yTrue, yPred: np.abs(yPred - yTrue)
    reduce_fn = lambda scores: np.nanmean(scores)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # Get airspeed for scans in batch.
    log_output_name = "_".join(batch_folders) + ".pkl"
    log_output_path = os.path.join(log_dir, log_output_name)
    if force_airspeed_analysis or not os.path.isfile(log_output_path):
        if len(batch_folders) == 1:
            figure_dir_batch = os.path.join(figure_dir, batch_folders[0])
            wind_error = GetAirSpeedBatch(os.path.join(wind_dir, batch_folders[0]), error_fn, reduce_fn,
                                          figure_dir=figure_dir_batch, debug_plots=False, save_plots=False)
        else:
            df_list = []
            for batch_folder in batch_folders:
                figure_dir_batch = os.path.join(figure_dir, batch_folder)
                df_list.append(GetAirSpeedBatch(os.path.join(wind_dir, batch_folder), error_fn, reduce_fn,
                                                figure_dir=figure_dir_batch, debug_plots=False, save_plots=False))
            wind_error = pd.concat(df_list, ignore_index=True)
            del df_list

        # Sort airspeeds by insect proportion.
        bird_ratio_bio = wind_error['prop_birds']
        insects_ratio_bio = wind_error['prop_insects']
        total = bird_ratio_bio + insects_ratio_bio
        insects_ratio_bio = np.divide(insects_ratio_bio, total)
        wind_error['insect_prop_bio'] = insects_ratio_bio * 100
        wind_error = wind_error.sort_values(by=['insect_prop_bio'])

        with open(log_output_path, "wb") as p_out:
            pickle.dump(wind_error, p_out)
        p_out.close()
    else:
        pin = open(log_output_path, 'rb')
        wind_error = pickle.load(pin)
        pin.close()

    # Get overall weather proportion
    wind_error["prop_weather_scan"] = ""

    echo_counts = {}
    for batch_folder in echo_count_batch_folders:
        echo_count_batch_path = os.path.join(echo_count_dir, batch_folder)
        for file in os.listdir(echo_count_batch_path):
            key = file.split('_')[0]
            with open(os.path.join(echo_count_batch_path, file), 'rb') as pin_echo:
                echo_counts[key] = pickle.load(pin_echo)
                pin_echo.close()

    unique_scans_wind = np.unique(wind_error['file_name'])

    for curr_scan_wind in unique_scans_wind:
        scans_list, echo_dist = echo_counts[curr_scan_wind.split("_")[0]]
        radar_ext = os.path.splitext(scans_list[0])[1]

        curr_scan_radar = "".join([curr_scan_wind[:-5], radar_ext])
        idx_scan_radar = [i for i in range(len(scans_list)) if scans_list[i].endswith(curr_scan_radar)][0]

        n_birds = echo_dist[VADMask.birds][idx_scan_radar]
        n_insects = echo_dist[VADMask.insects][idx_scan_radar]
        n_weather = echo_dist[VADMask.weather][idx_scan_radar]

        wind_error.loc[wind_error['file_name'] == curr_scan_wind, "prop_weather_scan"] = n_weather * 100 / (
                n_birds + n_insects + n_weather)

    return wind_error


def ImposeConstraints(df, constraints, idx=True):
    """
    :param df:
    :param constraints: list of constraints to be imposed on df. Structured as [(column name, lowerbound, upperbound),....]
    :return: index of df that satisfies constraints.
    """
    for constraint in constraints:
        if constraint[0] == 'file_name':
            idx_files = False
            for file_name in constraint[1]:
                idx_files = np.logical_or(idx_files, df["file_name"] == file_name)

            idx_files = np.logical_not(idx_files)
            idx = np.logical_and(idx, idx_files)
        else:
            idx = np.logical_and(idx,
                                 np.logical_and(df[constraint[0]] > constraint[1], df[constraint[0]] < constraint[2]))
    return idx


def FilterFlightspeeds(wind_error, constraints):
    target_idx = ImposeConstraints(wind_error, constraints)
    return wind_error[target_idx]


def VisualizeFlightspeeds(wind_error, constraints, color_info, c_group, save_plots, figure_summary_dir,
                          plot_title_suffix, out_name_suffix):
    idx_constraints = ImposeConstraints(wind_error, constraints)

    idx_valid = np.isfinite(wind_error['airspeed_birds'])
    idx_valid = np.logical_and(idx_valid, np.isfinite(wind_error['airspeed_insects']))
    idx_valid = np.logical_and(idx_valid, idx_constraints)
    print("Number of airspeed samples: ", np.sum(idx_valid))

    tmp = np.nanmax(wind_error['airspeed_insects'])
    max_airspeed = max(tmp, np.nanmax(wind_error['airspeed_birds']))
    max_airspeed = max_airspeed * 1.05

    # Plot 1. Airspeed vs height.
    max_height = 1.05 * max(wind_error['height_m'])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    im = ax[0].scatter(x=wind_error['airspeed_insects'][idx_valid],
                       y=wind_error['height_m'][idx_valid], s=80, c=color_info[c_group][0][idx_valid], alpha=0.3,
                       cmap=color_info[c_group][1], vmin=0,
                       vmax=100)
    ax[0].set_ylim(0, max_height)
    ax[0].set_xlim(0, max_airspeed)
    ax[0].set_title("VAD on insect detections.")

    im = ax[1].scatter(x=wind_error['airspeed_birds'][idx_valid],
                       y=wind_error['height_m'][idx_valid], s=80, c=color_info[c_group][0][idx_valid], alpha=0.3,
                       cmap=color_info[c_group][1], vmin=0,
                       vmax=100)
    ax[1].set_ylim(0, max_height)
    ax[1].set_xlim(0, max_airspeed)
    ax[1].set_title("VAD on bird detections.")

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Flight speed [m/s]")
    plt.ylabel("Height [m]")
    fig.subplots_adjust(right=0.8)
    fig.subplots_adjust(top=0.84)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=color_info[c_group][2])
    cbar.ax.set_yticklabels(color_info[c_group][3])

    if save_plots:
        plt.savefig(
            os.path.join(figure_summary_dir, "".join(["height_profile_", out_name_suffix, ".png"])),
            dpi=200)

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

    # Plot 4. Histogram of data v insect proportion
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

    # Plot 4. Histogram of bird, insect flightspeeds.
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

    plt.show()


# TODO
# None

def Main():
    # Inputs
    wind_dir = './vad_sounding_comparison_logs'
    batch_folders = ["KOHX_20180501_20180515_2hr_window",
                     "KOHX_20180516_20180531_2hr_window"]  # ["KENX_20180501_20180531"] #["KOHX_20180501_20180515_2hr_window", "KOHX_20180516_20180531_2hr_window"]
    experiment_id = "KOHX_20180501_20180531_2hr_window"  # "KENX_20180501_20180531_2hr_window"

    echo_count_dir = './analysis_output_logs'
    log_dir = "./post_processing_logs"

    figure_dir = "./figures"
    plot_title_suffix = "May 1 - 31, 2018"
    out_name_suffix = "May_1_31_2018"
    save_plots = False
    force_airspeed_analysis = False

    MAX_WEATHER_PROP = 10
    MAX_WEATHER_PROP_SCAN = 20  # 20

    # Get flight speeds of birds and insects
    wind_error = GetAirSpeedManyBatches(wind_dir, batch_folders, experiment_id, echo_count_dir, log_dir, figure_dir,
                                        plot_title_suffix, out_name_suffix, save_plots=save_plots,
                                        force_airspeed_analysis=force_airspeed_analysis)

    # Visualize flight speeds
    color_weather = wind_error['prop_weather']
    color_weather = (color_weather / MAX_WEATHER_PROP) * 100
    ticks_weather = [0, 50, 100]
    ticklabels_weather = ["".join([str(round(tick * MAX_WEATHER_PROP / 100, 1)), '% WEA']) for tick in ticks_weather]

    color_info = {"insect_prop": (wind_error['insect_prop_bio'], "jet", [0, 25, 50, 75, 100],
                                  ['BIR', 'BIR MAJ', r'BIR $\approx$ INS', 'INS MAJ', 'INS']),
                  "weather": (color_weather, "jet", ticks_weather, ticklabels_weather)}
    # values are (colour, colour map, ticks, labels)

    # Visualize flight speeds
    remove_cases_list = ['KOHX20180509_101350_V06_wind', 'KOHX20180509_102337_V06_wind',
                         'KOHX20180509_103324_V06_wind', 'KOHX20180509_104312_V06_wind',
                         'KOHX20180509_105259_V06_wind', 'KOHX20180509_110246_V06_wind',
                         'KOHX20180509_111233_V06_wind', 'KOHX20180509_112221_V06_wind',
                         'KOHX20180509_113206_V06_wind', 'KOHX20180509_114153_V06_wind',
                         'KOHX20180509_115140_V06_wind', 'KOHX20180509_120127_V06_wind',
                         'KOHX20180509_121114_V06_wind', 'KOHX20180509_122101_V06_wind',
                         'KOHX20180509_123046_V06_wind', 'KOHX20180509_124034_V06_wind',
                         'KOHX20180509_125021_V06_wind', 'KOHX20180509_130008_V06_wind',
                         'KOHX20180526_102910_V06_wind', 'KOHX20180526_103857_V06_wind',
                         'KOHX20180526_104842_V06_wind', 'KOHX20180526_105331_V06_wind',
                         'KOHX20180526_105816_V06_wind', 'KOHX20180526_110302_V06_wind',
                         'KOHX20180526_110747_V06_wind']

    # Visualize flight speeds.
    c_group = "insect_prop"  # "insect_prop"
    out_name_suffix = "_".join(["color", c_group, out_name_suffix])
    constraints = [("prop_weather", 0, MAX_WEATHER_PROP), ("prop_weather_scan", 0, MAX_WEATHER_PROP_SCAN),
                   ('file_name', remove_cases_list)]
    figure_summary_dir = os.path.join(figure_dir, experiment_id, 'summary')
    if not os.path.isdir(figure_summary_dir):
        os.makedirs(figure_summary_dir)

    VisualizeFlightspeeds(wind_error, constraints, color_info, c_group, save_plots, figure_summary_dir,
                          plot_title_suffix, out_name_suffix)

    # Search for cases defined by constraints.
    constraints = [("height_m", 0, 350), ("insect_prop_bio", 66.66, 100), ("airspeed_insects", 8, 11),
                   ("airspeed_birds", 7.5, 11), ("prop_weather", 0, MAX_WEATHER_PROP),
                   ("prop_weather_scan", 0, MAX_WEATHER_PROP_SCAN), ('file_name', remove_cases_list)]
    wind_error_filt = FilterFlightspeeds(wind_error, constraints)
    # print(np.unique(wind_error_filt.file_name))


Main()
