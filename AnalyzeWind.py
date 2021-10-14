import pyart
import os
import math
import pickle
import enum
import sys
import numpy as np
import matplotlib.pyplot as plt
from SoundingDataUtils import *
from VADUtils import fitVAD, VADWindProfile
from RadarHCAUtils import *

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 11}
plt.rc('font', **font)


# TODO
# Test: Reproduce profiles from Steph et al 2016
# need check for if sounding is available.
# Might need to tune for best availability threshold.

class VADMask(enum.Enum):
    default = 0
    biological = 1
    insects = 2
    birds = 3
    weather = 4


def VisualizeWinds(vad_jobs, vad_profiles_job, sounding_wind_df, max_height, description_jobs, title_str, prop_str,
                   output_folder):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    red_color_wheel = {VADMask.weather: "gold", VADMask.insects: "tomato", VADMask.birds: "lime"}
    blue_color_wheel = {VADMask.weather: "deepskyblue", VADMask.insects: "blueviolet", VADMask.birds: "cornflowerblue"}

    fig, ax = plt.subplots(1, 2)
    # Radar wind components.
    for job_idx in range(len(vad_jobs)):
        wind_profile_vad = vad_profiles_job[job_idx]
        vad_height_idx = wind_profile_vad['height'] < max_height

        ax[0].plot(wind_profile_vad["wind_direction"][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                   color=blue_color_wheel[vad_jobs[job_idx]], marker=description_jobs[vad_jobs[job_idx]][1], alpha=0.5,
                   label=description_jobs[vad_jobs[job_idx]][0] + " dir")
        ax[1].plot(wind_profile_vad["wind_speed"][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                   color=red_color_wheel[vad_jobs[job_idx]], marker=description_jobs[vad_jobs[job_idx]][1], alpha=0.5,
                   label=description_jobs[vad_jobs[job_idx]][0] + " spd")

    # Sounding wind components.
    sounding_height_idx = sounding_wind_df['HGHT'] < max_height
    ax[0].plot(sounding_wind_df["DRCT"][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
               label="sound dir", color="blue", linestyle='dashed')
    ax[1].plot(sounding_wind_df["SMPS"][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
               label="Sound spd", color="red", linestyle='dashed')

    ax[0].set_xlim(0, 360)
    ax[0].set_ylim(0, 1.4 * max_height)
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_xlim(0, 20)
    ax[1].set_ylim(0, 1.4 * max_height)
    ax[1].grid(True)
    ax[1].legend()
    fig.suptitle(title_str)
    plt.savefig(os.path.join(output_folder, "wind_comparison_spherical.png"))

    # Compare wind U and V components.
    plt.figure()

    # TODO generalize.
    wind_profile_vad = vad_profiles_job[1]
    vad_height_idx = wind_profile_vad['height'] < max_height
    plt.plot(wind_profile_vad['mean_ref'][vad_height_idx], wind_profile_vad['height'][vad_height_idx], color='black',
             label="ref")

    for job_idx in range(len(vad_jobs)):
        wind_profile_vad = vad_profiles_job[job_idx]
        vad_height_idx = wind_profile_vad['height'] < max_height

        # Radar wind components.
        plt.plot(wind_profile_vad['wind_U'][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                 color=blue_color_wheel[vad_jobs[job_idx]], marker=description_jobs[vad_jobs[job_idx]][1], alpha=0.5,
                 label=description_jobs[vad_jobs[job_idx]][0] + " vad_U")
        plt.plot(wind_profile_vad['wind_V'][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                 color=red_color_wheel[vad_jobs[job_idx]], marker=description_jobs[vad_jobs[job_idx]][1], alpha=0.5,
                 label=description_jobs[vad_jobs[job_idx]][0] + " vad_V")

    # Sounding wind components.
    sounding_height_idx = sounding_wind_df['HGHT'] < max_height
    plt.plot(sounding_wind_df['windU'][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
             label="wind_U", color="blue", linestyle='dashed')
    plt.plot(sounding_wind_df['windV'][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
             label="wind_V", color="red", linestyle='dashed')

    plt.ylim(0, 1.4 * max_height)
    plt.xlim(-16, 24)
    plt.grid(True)
    plt.xlabel("Wind components [mps]")
    plt.ylabel("Height [m]")
    plt.title(title_str + '\n' + prop_str)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "wind_comparison_components.png"))
    plt.show()

    return


def Main():
    # Radar.
    radar_data_file = "KOHX20150503_050828_V06.gz"  # "KOHX20180506_231319_V06"
    radar_data_folder = "./radar_data"
    hca_data_folder = "./hca_data"
    radar_t_sounding = {'KHTX': 'BNA', 'KTLX': 'LMN', 'KOHX': 'BNA'}
    station_infos = {'LMN': ('74646', 'Lamont, Oklahoma'), 'BNA': ('72327', 'Nashville, Tennessee')}

    # HCA.
    radar_base = radar_data_file[:12]
    radar_data_folder = os.path.join(radar_data_folder, radar_base)
    hca_data_folder = os.path.join(hca_data_folder, radar_base)

    # Sounding.
    radar_name, year, month, day, hh, mm, ss = read_info_from_radar_name(radar_data_file)
    station_id = station_infos[radar_t_sounding[radar_name]][0]
    station_desc = station_infos[radar_t_sounding[radar_name]][1]
    sounding_url_base = "http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR={}&MONTH={}&FROM={}&TO={}&STNM={}"

    # Classification model.
    norm_stats_file = "./models/ridge_bi/mean_std_for_normalization_1.pkl"
    bi_clf_file = "./models/ridge_bi/RidgeRegModels_SGD_1.pkl"

    # VAD options
    vad_jobs = [VADMask.birds, VADMask.insects, VADMask.weather]

    # Read HCA data.
    hca_vol = GetHcaVol(hca_data_folder, radar_data_file)

    # Read radar data.
    radar = pyart.io.read(os.path.join(radar_data_folder, radar_data_file))
    location_radar = {"latitude": radar.latitude['data'][0],
                      "longitude": radar.longitude['data'][0],
                      "height": radar.altitude['data'][0]}

    radar_products_slice = {0: ["differential_reflectivity", "differential_phase", "cross_correlation_ratio"],
                            1: ["reflectivity", "velocity", "spectrum_width"]}
    data_table = MergeRadarAndHCAUpdate(radar, hca_vol, 300)
    # TODO get original ZDR mask.
    data_table["mask_differential_reflectivity"] = data_table["differential_reflectivity"] > -8.0
    data_table["hca_bio"] = data_table["hca"] == 10.0
    data_table["hca_weather"] = np.logical_and(data_table["hca"] >= 30.0, data_table["hca"] <= 100.0)
    data_table["height"] = data_table["range"] * np.sin(data_table["elevation"] * np.pi / 180)

    height_binsize = 0.04  # 0.05
    data_table["height_bin_meters"] = (np.floor(
        data_table["height"] / height_binsize) + 1) * height_binsize - height_binsize / 2
    data_table["height_bin_meters"] *= 1000

    # Apply bird-insect classifier. -1 is non-bio, 1 is bird and 0 is insects.
    # TODO load directly from stored file.
    norm_stats = {
        'mean': pd.Series(data={'ZDR': 2.880010, 'pdp': 112.129741, 'RHV': 0.623049}, index=['ZDR', 'pdp', 'RHV']),
        'standard deviation': pd.Series(data={'ZDR': 2.936261, 'pdp': 52.774116, 'RHV': 0.201977},
                                        index=['ZDR', 'pdp', 'RHV'])}
    pin = open(bi_clf_file, 'rb')
    bi_clf = pickle.load(pin)
    pin.close()

    valid_bio_mask = np.logical_and(data_table["mask_differential_reflectivity"], data_table["hca_bio"])
    X = data_table.loc[
        valid_bio_mask, ['differential_reflectivity', 'differential_phase', 'cross_correlation_ratio']]
    X.rename(columns={"differential_reflectivity": "ZDR"}, inplace=True)
    X.rename(columns={"differential_phase": "pdp"}, inplace=True)
    X.rename(columns={"cross_correlation_ratio": "RHV"}, inplace=True)
    X = X - norm_stats['mean']
    X = X.div(norm_stats['standard deviation'], axis=1)
    X = np.array(X)
    data_table['BIClass'] = -1
    data_table.loc[valid_bio_mask, 'BIClass'] = bi_clf['model'].predict(X)

    # Visualize data table.
    color_map = {'differential_phase': ('Theodore16', [0, 180], 10),
                 'reflectivity': ('NWSRef', [-25, 75], 10),
                 'differential_reflectivity': ('RefDiff', [-7.9, 7.9], 10),
                 'cross_correlation_ratio': ('RRate11', [0.2, 1.05], 10),
                 'velocity': ('NWSVel', [-37, 37], 10),
                 'hca': ('viridis', [0, 150], 16),
                 'hca_bio': ('viridis', [0, 1], 2),
                 'hca_weather': ('Spectral', [0, 1], 2),
                 'BIClass': ('viridis', [-1, 1], 3)}

    figure_folder = os.path.join('./figures', radar_base)
    # VisualizeDataTable(data_table, color_map, figure_folder)

    # VAD wind profile
    signal_func = lambda x, t: x[0] * np.sin(2 * np.pi * (1 / 360) * t + x[1])
    max_height_VAD = 1000
    vad_heights = np.arange(50, max_height_VAD, 25)

    vad_profiles_job = []
    for vad_mask in vad_jobs:
        # Select type of echoes for VAD.
        if vad_mask == VADMask.biological:
            vad_mask_arr = np.logical_and(data_table["hca_bio"], data_table["mask_velocity"])
        elif vad_mask == VADMask.insects:
            vad_mask_arr = np.logical_and(data_table["mask_differential_reflectivity"], data_table["hca_bio"])
            vad_mask_arr = np.logical_and(vad_mask_arr, data_table["BIClass"] == 0)
            vad_mask_arr = np.logical_and(vad_mask_arr, data_table["mask_velocity"])
        elif vad_mask == VADMask.birds:
            vad_mask_arr = np.logical_and(data_table["mask_differential_reflectivity"], data_table["hca_bio"])
            vad_mask_arr = np.logical_and(vad_mask_arr, data_table["BIClass"] == 1)
            vad_mask_arr = np.logical_and(vad_mask_arr, data_table["mask_velocity"])
        elif vad_mask == VADMask.weather:
            vad_mask_arr = np.logical_and(data_table["hca_weather"], data_table["mask_velocity"])
        else:
            vad_mask_arr = data_table["mask_velocity"]

        wind_profile_vad = VADWindProfile(signal_func, vad_heights, vad_mask_arr, data_table, showDebugPlot=False)
        vad_profiles_job.append(wind_profile_vad)

    # Sounding wind profile.
    year_sounding, month_sounding, ddhh_sounding = GetSoundingDateTimeFromRadarFile(radar_data_file)
    sounding_wind_df, sounding_location, sounding_url = GetSoundingWind(sounding_url_base, radar_data_file,
                                                                        location_radar,
                                                                        station_id, False)
    distance_radar_sounding = GetHaverSineDistance(location_radar["latitude"], location_radar["longitude"],
                                                   sounding_location["latitude"],
                                                   sounding_location["longitude"])
    distance_radar_sounding = round(distance_radar_sounding / 1000, 2)

    # plots
    height_msk = data_table["height_bin_meters"] < max_height_VAD
    total_echoes = np.sum(np.logical_or(data_table["hca_bio"][height_msk], data_table["hca_weather"][height_msk]))

    prop_birds = np.sum(data_table['BIClass'][height_msk] == 1) / total_echoes
    prop_birds = round(prop_birds * 100)
    prop_insects = np.sum(data_table['BIClass'][height_msk] == 0) / total_echoes
    prop_insects = round(prop_insects * 100)
    prop_weather = np.sum(data_table['hca_weather'][height_msk] == 1) / total_echoes
    prop_weather = round(prop_weather * 100)

    prop_str = "{}% birds, {}% insects, {}% weather".format(prop_birds, prop_insects, prop_weather)

    title_str = "{}, {}/{}/{}, {}:{}:{} UTC.\n{} km from {} UTC {} sounding.".format(
        radar_name, year, month,
        day, hh, mm, ss,
        distance_radar_sounding,
        ddhh_sounding[2:],
        station_desc)

    description_jobs = {VADMask.biological: ("bio", "."), VADMask.insects: ("ins", "2"),
                        VADMask.weather: ("wea", "d"), VADMask.birds: ("bir", "^")}
    VisualizeWinds(vad_jobs, vad_profiles_job, sounding_wind_df, 1000, description_jobs, title_str, prop_str,
                   figure_folder)


Main()
