import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from VADMaskEnum import VADMask, GetVADMaskDescription
from TrueWindEnum import *


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


def VisualizeProfilesScan(vel_profiles, wind_file_no_ext, figure_dir, save_plots=False):
    out_dir = os.path.join(figure_dir, 'vad_profiles')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Plot options
    line_width = 2
    max_height = 1100

    # VAD probability profile.
    fig, ax = plt.subplots()
    idx_profile = np.isfinite(vel_profiles["mean_prob_biological"])
    ax.plot(vel_profiles["mean_prob_biological"][idx_profile], vel_profiles["height_m"][idx_profile], linewidth=line_width)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max_height)
    ax.grid(True)
    ax.set_xlabel("Wind tracing score (no unit)")
    ax.set_ylabel("Height (m)")
    ax.set_title("VAD wind tracing profile")
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(out_dir, ''.join([wind_file_no_ext, '_probs_profile'])), dpi=200)

    # VAD Census profile.
    num_birds_height = vel_profiles["num_birds_height"] / 1000
    num_insects_height = vel_profiles["num_insects_height"] / 1000

    fig, ax = plt.subplots()
    idx_birds = np.isfinite(num_birds_height)
    idx_insects = np.isfinite(num_insects_height)
    max_pop = max(np.nanmax(num_birds_height), np.nanmax(num_insects_height))
    max_pop = max(max_pop, 16500 / 1000)

    ax.plot(num_birds_height[idx_birds], vel_profiles["height_m"][idx_birds], label='birds', c='b',
            linewidth=line_width)
    ax.plot(num_insects_height[idx_insects], vel_profiles["height_m"][idx_insects], label='insects',
            c='r', linewidth=line_width)
    ax.set_xlim(0, max_pop)
    ax.set_ylim(0, max_height)
    ax.grid(True)
    ax.set_xlabel(r"Number of gates ($\times\,10^3$ )")
    ax.set_ylabel("Height (m)")
    ax.set_title("Bird and insect census")
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(out_dir, ''.join([wind_file_no_ext, '_census_profile'])), dpi=200)
    # plt.show()

    return

def GetVelocitiesScan(wind_file, vad, sounding_df, echo_dist, figure_dir, debug_plots=False):
    # Sounding
    vel_profiles = sounding_df.loc[:, ["HGHT", "DRCT", "SMPS"]]
    vel_profiles.rename(columns={"HGHT": "height_m", "DRCT": "wind_direction", "SMPS": "wind_speed"}, inplace=True)
    wind_file_no_ext = os.path.splitext(wind_file)[0]
    vel_profiles['file_name'] = wind_file_no_ext
    vel_profiles = vel_profiles.drop_duplicates(subset='height_m', keep='last')

    # VAD
    vad_vel_cols_base = ["height", "wind_speed", "wind_direction", "num_samples"]
    new_cols_base = ["height_m", "{}_speed", "{}_direction", "num_{}_height"]

    for echo in vad:
        if echo == VADMask.weather:
            continue

        vad_vel_cols = vad_vel_cols_base.copy()
        new_cols = new_cols_base.copy()
        new_cols = [new_cols[i] if i < 1 else new_cols[i].format(GetVADMaskDescription(echo)) for i in
                    range(len(new_cols))]

        if echo == VADMask.biological:
            vad_vel_cols.insert(1, 'mean_ref')
            vad_vel_cols.insert(2, 'mean_prob')
            new_cols.insert(1, '_'.join(['mean_ref', GetVADMaskDescription(echo)]))
            new_cols.insert(2, '_'.join(['mean_prob', GetVADMaskDescription(echo)]))

        echo_df = vad[echo].loc[:, vad_vel_cols]
        echo_df.rename(columns=dict(zip(vad_vel_cols, new_cols)), inplace=True)
        echo_df = echo_df.drop_duplicates(subset='height_m', keep='last')

        vel_profiles = pd.merge(vel_profiles, echo_df, on="height_m", how="outer")

    vel_profiles['prop_birds'] = echo_dist['bird']
    vel_profiles['prop_insects'] = echo_dist['insects']
    vel_profiles['prop_weather'] = echo_dist['weather']

    if debug_plots:
        VisualizeProfilesScan(vel_profiles=vel_profiles, wind_file_no_ext=wind_file_no_ext, figure_dir=figure_dir,
                              save_plots=False)

    return vel_profiles

def extract_features_from_ppi_name(s):
    radar_name = s[:4]
    yyyy = s[4:8]
    month = float(s[8:10])
    day = float(s[10:12])
    hh = float(s[13:15])
    mm = float(s[15:17])
    ss = float(s[17:19])
    time_hour = float(hh) + float(mm) / 60 + float(ss) / 3600
    return radar_name, yyyy, month, day, time_hour


def UpdateWindError(wind_error_df, target_file, vad_profiles, sounding_df, echo_dist_VAD, error_fn, reduce_fn,
                    ground_truth_source, figure_dir, batch_folder, n_birds, n_insects, n_weather):
    if sounding_df is None or vad_profiles is None or target_file == '':
        return wind_error_df

    for echo_type in vad_profiles:
        if vad_profiles[echo_type].empty:
            return wind_error_df

    airspeed_scan = GetVelocitiesScan(wind_file=target_file, vad=vad_profiles, sounding_df=sounding_df,
                                      echo_dist=echo_dist_VAD, figure_dir=figure_dir, debug_plots=True)

    # airspeed_scan = GetAirSpeedScan(wind_file=target_file, vad=vad_profiles, sounding_df=sounding_df,
    #                                 echo_dist=echo_dist_VAD, error_fn=error_fn, reduce_fn=reduce_fn,
    #                                 ground_truth_wind_source=ground_truth_source,
    #                                 figure_dir=os.path.join(figure_dir, batch_folder))

    # Calculate proportion of weather echoes for the whole scan.
    airspeed_scan["prop_weather_scan"] = n_weather * 100 / (
            n_birds + n_insects + n_weather)

    # Calculate proportion of insects relative to biological echoes.
    bird_ratio_bio = airspeed_scan['prop_birds']
    insects_ratio_bio = airspeed_scan['prop_insects']
    total = bird_ratio_bio + insects_ratio_bio
    insects_ratio_bio = np.divide(insects_ratio_bio, total)
    airspeed_scan['insect_prop_bio'] = insects_ratio_bio * 100
    # airspeed_scan = airspeed_scan.sort_values(by=['insect_prop_bio'])

    radar_name, yyyy, month, day, time_hour = extract_features_from_ppi_name(target_file)
    airspeed_scan["radar"] = radar_name
    airspeed_scan["year"] = yyyy
    airspeed_scan["month"] = month
    airspeed_scan["day"] = day
    airspeed_scan["time_hour"] = time_hour

    # Update airspeed table.
    if airspeed_scan is not None:
        wind_error_df = wind_error_df.append(airspeed_scan, ignore_index=True)

    return wind_error_df


def GetAirSpeedScanFromLog(wind_file, wind_files_parent, error_fn, reduce_fn, debug_plots=False, figure_dir=None,
                           save_plots=True, ground_truth_wind_source=WindSource.sounding):
    def GetAirSpeeds(delta_U, delta_V):
        airspeeds = []
        for i in range(len(delta_U)):
            airspeeds.append(np.sqrt(delta_U[i] ** 2 + delta_V[i] ** 2))
        return airspeeds

    wind_source_desc = GetWindSourceDescription(ground_truth_wind_source)
    wind_source_desc = wind_source_desc.replace(' ', '_')

    wind_file_no_ext = os.path.splitext(wind_file)[0]

    figure_dir = os.path.join(figure_dir, wind_file_no_ext[:12], 'airspeed', wind_source_desc)
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

    if debug_plots or save_plots:
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
        # if debug_plots:
        #     plt.show()
        plt.close()

    return pd.DataFrame({'file_name': wind_file_no_ext, 'airspeed_birds': airspeed_birds,
                         'airspeed_insects': airspeed_insects, 'height_m': x_sound_insects_U,
                         'prop_birds': echo_dist['bird'],
                         'prop_insects': echo_dist['insects'],
                         'prop_weather': echo_dist['weather']})


def GetAirSpeedBatch(wind_files_parent, error_fn, reduce_fn, figure_dir, debug_plots, save_plots=False,
                     ground_truth_wind_source=WindSource.sounding):
    wind_files = os.listdir(wind_files_parent)
    wind_error_df = pd.DataFrame(
        columns=['file_name', 'airspeed_birds', 'airspeed_insects', 'height_m', 'prop_birds', 'prop_insects',
                 'prop_weather'])

    # Get wind here.
    for wind_file in wind_files:
        entry = GetAirSpeedScanFromLog(wind_file, wind_files_parent, error_fn, reduce_fn, debug_plots=debug_plots,
                                       save_plots=save_plots, figure_dir=figure_dir,
                                       ground_truth_wind_source=ground_truth_wind_source)
        if entry is not None:
            wind_error_df = wind_error_df.append(entry, ignore_index=True)
    return wind_error_df


def GetAirSpeedManyBatches(wind_dir, batch_folders, experiment_id, echo_count_dir, log_dir, figure_dir,
                           plot_title_suffix, out_name_suffix, save_plots=False, force_airspeed_analysis=False,
                           ground_truth_source=WindSource.sounding, debug_airspeed_plots=False,
                           save_airspeed_plots=False, correct_hca_weather=False):
    wind_source_desc = GetWindSourceDescription(ground_truth_source)
    wind_source_desc = wind_source_desc.replace(' ', '_')

    echo_count_batch_folders = []
    for folder in batch_folders:
        # echo_count_batch_folders.append(folder[:22])
        echo_count_batch_folders.append(folder)

    error_fn = lambda yTrue, yPred: np.abs(yPred - yTrue)
    reduce_fn = lambda scores: np.nanmean(scores)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # Get airspeed for scans in batch.
    log_output_name = "_".join(batch_folders) + "_" + wind_source_desc + ".pkl"

    if correct_hca_weather:
        log_files_parent = os.path.join(log_dir, 'hca_weather_corrected')
    else:
        log_files_parent = os.path.join(log_dir, 'hca_default')

    if not os.path.isdir(log_files_parent):
        os.makedirs(log_files_parent)
    log_output_path = os.path.join(log_files_parent, log_output_name)

    if force_airspeed_analysis or not os.path.isfile(log_output_path):
        if len(batch_folders) == 1:
            figure_dir_batch = os.path.join(figure_dir, batch_folders[0])
            wind_folder = os.path.join(wind_dir, batch_folders[0], 'hca_weather_corrected',
                                       wind_source_desc) if correct_hca_weather else os.path.join(wind_dir,
                                                                                                  batch_folders[0],
                                                                                                  'hca_default',
                                                                                                  wind_source_desc)
            wind_error = GetAirSpeedBatch(wind_folder, error_fn,
                                          reduce_fn,
                                          figure_dir=figure_dir_batch, debug_plots=debug_airspeed_plots,
                                          save_plots=save_airspeed_plots,
                                          ground_truth_wind_source=ground_truth_source)
        else:
            df_list = []
            for batch_folder in batch_folders:
                figure_dir_batch = os.path.join(figure_dir, batch_folder)
                wind_folder = os.path.join(wind_dir, batch_folder, 'hca_weather_corrected',
                                           wind_source_desc) if correct_hca_weather else os.path.join(wind_dir,
                                                                                                      batch_folder,
                                                                                                      'hca_default',
                                                                                                      wind_source_desc)
                print(wind_folder)
                df_list.append(
                    GetAirSpeedBatch(wind_folder, error_fn, reduce_fn,
                                     figure_dir=figure_dir_batch, debug_plots=debug_airspeed_plots,
                                     save_plots=save_airspeed_plots,
                                     ground_truth_wind_source=ground_truth_source))
            wind_error = pd.concat(df_list, ignore_index=True)
            del df_list

        # Sort airspeeds by insect proportion.
        bird_ratio_bio = wind_error['prop_birds']
        insects_ratio_bio = wind_error['prop_insects']
        total = bird_ratio_bio + insects_ratio_bio
        insects_ratio_bio = np.divide(insects_ratio_bio, total)
        wind_error['insect_prop_bio'] = insects_ratio_bio * 100
        wind_error = wind_error.sort_values(by=['insect_prop_bio'])

        # Get overall weather proportion
        wind_error["prop_weather_scan"] = ""

        echo_counts = {}
        for batch_folder in echo_count_batch_folders:
            echo_count_batch_path = os.path.join(echo_count_dir, batch_folder,
                                                 'hca_weather_corrected') if correct_hca_weather else os.path.join(
                echo_count_dir,
                batch_folder, 'hca_default')
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

        with open(log_output_path, "wb") as p_out:
            pickle.dump(wind_error, p_out)
        p_out.close()
    else:
        pin = open(log_output_path, 'rb')
        wind_error = pickle.load(pin)
        pin.close()

    return wind_error
