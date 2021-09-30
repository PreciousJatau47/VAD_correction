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

# There is a trade off between sample size for VAD and having only insects. Consider combining samples over 0.75 km intervals


class VADMask(enum.Enum):
    default = 0
    biological = 1
    insects = 2


def Main():
    # radar
    radar_data_file = "KTLX20150916_203558_V06.gz" #"KHTX20150811_111510_V06.gz" # "KOHX20150503_050828_V06.gz"  # "KHTX20150811_111510_V06.gz"
    radar_data_folder = "./radar_data"
    hca_data_folder = "./hca_data"
    radar_t_sounding = {'KHTX': 'BNA', 'KTLX': 'LMN', 'KOHX': 'BNA'}
    station_infos = {'LMN': ('74646', 'Lamont, Oklahoma'), 'BNA': ('72327', 'Nashville, Tennessee')}

    # VAD options
    vad_mask = VADMask.insects

    # plottting output TODO Undecided
    desc = {VADMask.insects: 'insects', VADMask.biological: 'biological', VADMask.default: ''}
    out_name = os.path.splitext(radar_data_file)[0] + '_' + desc[vad_mask]

    # sounding
    radar_name, year, month, day, hh, mm, ss = read_info_from_radar_name(radar_data_file)
    station_id = station_infos[radar_t_sounding[radar_name]][0]
    station_desc = station_infos[radar_t_sounding[radar_name]][1]
    sounding_url_base = "http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR={}&MONTH={}&FROM={}&TO={}&STNM={}"

    # classification model
    norm_stats_file = "./models/ridge_bi/mean_std_for_normalization_1.pkl"
    bi_clf_file = "./models/ridge_bi/RidgeRegModels_SGD_1.pkl"

    radar_base = radar_data_file[:12]
    radar_data_folder = os.path.join(radar_data_folder, radar_base)
    hca_data_folder = os.path.join(hca_data_folder, radar_base)
    hca_file = find_hca_file_name_from_radar(radar_data_file)
    print(hca_file)

    # Read HCA data.
    hca = pyart.io.read_nexrad_level3(os.path.join(hca_data_folder, hca_file))

    # Read radar data.
    radar = pyart.io.read(os.path.join(radar_data_folder, radar_data_file))
    location_radar = {"latitude": radar.latitude['data'][0],
                      "longitude": radar.longitude['data'][0],
                      "height": radar.altitude['data'][0]}

    radar_products_slice = {0: ["differential_reflectivity", "differential_phase", "cross_correlation_ratio"],
                            1: ["reflectivity", "velocity", "spectrum_width"]}
    # radar_dp_table = ReadRadarCutAsTable(radar, radar_products_slice, 0)
    # radar_sp_table = ReadRadarCutAsTable(radar, radar_products_slice, 1)
    data_table = MergeRadarAndHCA(radar, radar_products_slice, hca, 300)
    data_table["hca_bio"] = data_table["hca"] == 10.0

    # Apply bird-insect classifier. -1 is non-bio, 1 is bird and 0 is insects.
    # TODO load directly from stored file.
    norm_stats = {
        'mean': pd.Series(data={'ZDR': 2.880010, 'pdp': 112.129741, 'RHV': 0.623049}, index=['ZDR', 'pdp', 'RHV']),
        'standard deviation': pd.Series(data={'ZDR': 2.936261, 'pdp': 52.774116, 'RHV': 0.201977},
                                        index=['ZDR', 'pdp', 'RHV'])}
    pin = open(bi_clf_file, 'rb')
    bi_clf = pickle.load(pin)
    pin.close()

    X = data_table.loc[
        data_table['hca_bio'], ['differential_reflectivity', 'differential_phase', 'cross_correlation_ratio']]
    X.rename(columns={"differential_reflectivity": "ZDR"}, inplace=True)
    X.rename(columns={"differential_phase": "pdp"}, inplace=True)
    X.rename(columns={"cross_correlation_ratio": "RHV"}, inplace=True)
    X = X - norm_stats['mean']
    X = X.div(norm_stats['standard deviation'], axis=1)
    X = np.array(X)
    data_table['BIClass'] = -1
    data_table.loc[data_table['hca_bio'], 'BIClass'] = bi_clf['model'].predict(X)
    print("{}% birds, {}% insects, and {}% non biological".format(np.mean(data_table['BIClass'] == 1),
                                                                  np.mean(data_table['BIClass'] == 0),
                                                                  np.mean(data_table['BIClass'] == -1)))

    # VAD wind profile
    signal_func = lambda x, t: x[0] * np.sin(2 * np.pi * (1 / 360) * t + x[1])
    max_range_VAD = 300  # km
    vad_ranges = np.arange(10, max_range_VAD, 2)

    if vad_mask == VADMask.biological:
        data_table["vad_mask"] = np.logical_and(data_table["hca_bio"], data_table["mask_sp"])
    elif vad_mask == VADMask.insects:
        data_table["vad_mask"] = np.logical_and(data_table["hca_bio"], data_table["mask_sp"])
        data_table["vad_mask"] = np.logical_and(data_table["vad_mask"], data_table["BIClass"])
    else:
        data_table["vad_mask"] = data_table["mask_sp"]

    wind_profile_vad = VADWindProfile(signal_func, vad_ranges, data_table, False)

    year_sounding, month_sounding, ddhh_sounding = GetSoundingDateTimeFromRadarFile(radar_data_file)
    sounding_wind_df, sounding_location, sounding_url = GetSoundingWind(sounding_url_base, radar_data_file,
                                                                        location_radar,
                                                                        station_id, False)
    distance_radar_sounding = GetHaverSineDistance(location_radar["latitude"], location_radar["longitude"],
                                                   sounding_location["latitude"],
                                                   sounding_location["longitude"])
    distance_radar_sounding = round(distance_radar_sounding / 1000, 2)

    # plots
    title_str = "{}, {}/{}/{}, {}:{}:{} UTC.\n{} km from {} UTC {} sounding.".format(radar_name, year, month,
                                                                                     day, hh, mm, ss,
                                                                                     distance_radar_sounding,
                                                                                     ddhh_sounding[2:],
                                                                                     station_desc)
    max_height = 1000  # m
    vad_height_idx = wind_profile_vad['height'] < max_height
    sounding_height_idx = sounding_wind_df['HGHT'] < max_height

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(wind_profile_vad["wind_direction"][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
               label="VAD dir")
    ax[0].plot(sounding_wind_df["DRCT"][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
               label="Sounding dir")
    ax[0].set_xlim(0, 360)
    ax[0].set_ylim(0, 1.4 * max_height)
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(wind_profile_vad["wind_speed"][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
               label="VAD speed")
    ax[1].plot(sounding_wind_df["SMPS"][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
               label="Sounding speed")
    ax[1].set_xlim(0, 20)
    ax[1].set_ylim(0, 1.4 * max_height)
    ax[1].grid(True)
    ax[1].legend()
    fig.suptitle(title_str)
    plt.savefig(out_name + '_wind_vector.png', dpi = 200)
    # plt.show()

    plt.figure()
    plt.plot(wind_profile_vad['wind_U'][vad_height_idx], wind_profile_vad['height'][vad_height_idx], label="vad_U",
             color="blue")
    plt.plot(wind_profile_vad['wind_V'][vad_height_idx], wind_profile_vad['height'][vad_height_idx], label="vad_V",
             color="red")
    plt.plot(sounding_wind_df['windU'][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
             label="wind_U", color="blue", linestyle='dashed')
    plt.plot(sounding_wind_df['windV'][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
             label="wind_V", color="red", linestyle='dashed')
    plt.title(title_str)
    plt.ylim(0, 1.4 * max_height)
    plt.xlim(-25, 25)
    plt.grid(True)
    plt.xlabel("Wind components [mps]")
    plt.ylabel("Height [m]")
    plt.legend()
    plt.savefig(out_name + '_wind_comp.png', dpi = 200)
    # plt.savefig("vad.png")

    plt.show()


Main()
