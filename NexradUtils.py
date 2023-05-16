import os
import pickle
import fnmatch
from RadarHCAUtils import *
from ClfUtils import classify_echoes
from HaverSineDistance import GetHaverSineDistance


def GetTimeHourUTC(some_str: str) -> float:
    idx_timestart = some_str.find('_') + 1
    hh = some_str[idx_timestart:idx_timestart + 2]
    mm = some_str[idx_timestart + 2:idx_timestart + 4]
    ss = some_str[idx_timestart + 4:idx_timestart + 6]
    time_hour = float(hh) + float(mm) / 60 + float(ss) / 3600
    return time_hour


def GetFileListL3(batch_folder_path):
    """
    :param batch_folder_path:
    :return:
    """
    batch_filelist_dict_path = os.path.join(batch_folder_path, 'fileListDic.pkl')
    if os.path.exists(batch_filelist_dict_path):
        print(batch_filelist_dict_path, " exists. Loading file.")
        p_in = open(batch_filelist_dict_path, "rb")
        filelist_dic = pickle.load(p_in)
        p_in.close()
    else:
        print(batch_filelist_dict_path, " does not exist. Reading fileList.txt.")
        file_obj = open(os.path.join(batch_folder_path, 'fileList.txt'), 'r')
        filelists = file_obj.read().splitlines()
        file_obj.close()

        filelist_dic = {}
        for line in filelists:
            idx_first_underscore = line.find('_')
            filelist_dic[line[idx_first_underscore + 1:]] = line

        p_out = open(batch_filelist_dict_path, "wb")
        pickle.dump(filelist_dic, p_out)
        p_out.close()

    return filelist_dic


def GetFileListRadar(batch_folder_path, start_day, stop_day, date_pattern):
    file_obj = open(os.path.join(batch_folder_path, 'fileList.txt'), 'r')
    filelists = file_obj.read().splitlines()
    file_obj.close()

    filelist_dic = {}
    # Get the radar filename.
    key_fn = lambda x: os.path.splitext(os.path.split(x)[1])[0]

    for current_day in range(start_day, stop_day + 1):
        current_day_str = '0' + str(current_day) if current_day < 10 else str(current_day)
        pattern = date_pattern.format(current_day_str)
        filtered_files = fnmatch.filter(filelists, pattern)
        filtered_files.sort(key=key_fn)
        # print(pattern[1:13])
        filelist_dic[pattern[1:13]] = filtered_files
    return filelist_dic


def PrepareDataTable(batch_folder_path_radar, radar_subpath, batch_folder_path_l3, l3_files_dic, max_range,
                     height_binsize=0.04, clf_file=None, norm_stats_file=None, correct_hca_weather=False,
                     max_height_correction=1000, biw_norm_stats_file=None, biw_clf_file=None, allowed_el_hca=None):
    # Read radar volume.
    try:
        print("Opening ", os.path.join(batch_folder_path_radar, radar_subpath))
        radar_obj = pyart.io.read_nexrad_archive(os.path.join(batch_folder_path_radar, radar_subpath))
    except Exception as e:
        print("An error occured in reading radar file: ", str(e), "\nSkipping to next iteration.")
        return None, None, None

    # Read HCA volume.
    radar_filename = os.path.splitext(os.path.split(radar_subpath)[1])[0]
    if allowed_el_hca is None:
        sys.exit("allowed_el_hca is None. Specify the elevations to process.")

    try:
        hca_vol = GetHcaVolFromFileList(batch_folder_path_l3, radar_filename, l3_files_dic, allowed_el_hca)
    except:
        print('Read failed for expected l3 file. Might not exist or might be corrupted. Skipping to next iteration.')
        return None, None, None

    data_table = MergeRadarAndHCAUpdate(radar_obj, hca_vol, max_range)
    if data_table.empty:
        print(
            'Empty data table. True elevation angle might be far from expected elevation. Skipping to next iteration.')
        return None, None, None

    data_table["height"] = data_table["range"] * np.sin(data_table["elevation"] * np.pi / 180)

    if correct_hca_weather:
        X_biw = data_table.loc[:, ['differential_reflectivity', 'differential_phase', 'cross_correlation_ratio']]
        X_biw.rename(columns={"differential_reflectivity": "ZDR"}, inplace=True)
        X_biw.rename(columns={"differential_phase": "pdp"}, inplace=True)
        X_biw.rename(columns={"cross_correlation_ratio": "RHV"}, inplace=True)
        data_table['BIWClass'] = -1
        biw_class, _ = classify_echoes(X_biw, biw_clf_file, norm_stats_path=biw_norm_stats_file)
        data_table.loc[:, 'BIWClass'] = biw_class
        # data_table.loc[:, 'BIWClass'] = classify_echoes(X_biw, biw_clf_file, norm_stats_path=biw_norm_stats_file)

        # Correct HCA's misclassification of birds as weather within VAD region.
        # if weather hca, and non-weather biw, and within collection region, set to biological
        correction_msk = data_table["height"] < max_height_correction
        weather_hca = np.logical_and(data_table["hca"] >= 30.0, data_table["hca"] <= 100.0)
        non_weather_biw = data_table['BIWClass'] != 3
        correction_msk = np.logical_and(correction_msk, weather_hca)
        correction_msk = np.logical_and(correction_msk, non_weather_biw)
        data_table.loc[correction_msk, "hca"] = 10.0

    data_table["mask_differential_reflectivity"] = data_table["differential_reflectivity"] > -8.0
    data_table["hca_bio"] = data_table["hca"] == 10.0
    data_table["hca_weather"] = np.logical_and(data_table["hca"] >= 30.0, data_table["hca"] <= 100.0)
    data_table["height_bin_meters"] = (data_table[
                                           "height"] // height_binsize * height_binsize + height_binsize / 2) * 1000

    echo_mask = np.logical_and(data_table["mask_differential_reflectivity"], data_table["hca_bio"])
    X = data_table.loc[
        echo_mask, ['differential_reflectivity', 'differential_phase', 'cross_correlation_ratio']]
    X.rename(columns={"differential_reflectivity": "ZDR"}, inplace=True)
    X.rename(columns={"differential_phase": "pdp"}, inplace=True)
    X.rename(columns={"cross_correlation_ratio": "RHV"}, inplace=True)
    data_table['BIClass'], data_table['BIProb'] = -1, -1
    if not X.empty:
        bi_class, bi_probs = classify_echoes(X, clf_file, norm_stats_file)
        data_table.loc[echo_mask, 'BIClass'] = bi_class
        data_table.loc[echo_mask, 'BIProb'] = bi_probs[:, 1]

    return data_table, radar_obj, hca_vol


def GetNexradTable(radar_base_folder, radar_folder_table, save_table=False, force_collection=False):
    output_path = os.path.join(radar_base_folder, 'nexrad_location_table.pkl')

    if not force_collection and os.path.isfile(output_path):
        print("Loading local copy of NEXRAD table ...")
        with open(output_path, 'rb') as p_in:
            radar_table = pickle.load(p_in)
        p_in.close()
        return radar_table

    radar_folder_table = os.path.join(radar_base_folder, radar_folder_table)
    radar_files = os.listdir(radar_folder_table)
    radar_table = {}

    for radar_file in radar_files:
        radar_obj = pyart.io.read_nexrad_archive(os.path.join(radar_folder_table, radar_file))
        print("Obtaining location for ", radar_file[:4])
        radar_table[radar_file[:4]] = {"latitude": radar_obj.latitude['data'][0],
                                       "longitude": radar_obj.longitude['data'][0],
                                       "height": radar_obj.altitude['data'][0]}

    if save_table:
        with open(output_path, 'wb') as p_out:
            pickle.dump(radar_table, p_out)
        p_out.close()

    return radar_table


def RadarToOthersDistance(radar_id, nexrad_table):
    if radar_id not in nexrad_table:
        print(radar_id, " not found in nexrad table. Returning ...")
        return None
    self_location = nexrad_table[radar_id]
    self_t_others = []

    for other_id in nexrad_table:
        if other_id == radar_id:
            continue
        self_t_others.append((other_id, GetHaverSineDistance(self_location["latitude"], self_location["longitude"],
                                                             nexrad_table[other_id]["latitude"],
                                                             nexrad_table[other_id]["longitude"])))
    # sort by distance
    self_t_others = sorted(self_t_others, key=lambda x: x[1])

    return self_t_others


def RadarXRadarDistance(nexrad_table, output_folder, to_save=False, force_update=False):
    output_path = os.path.join(output_folder, 'RadarXRadarDistances.pkl')

    if not force_update and os.path.isfile(output_path):
        print("Loading local copy of RadarXRadarDistance ...")
        with open(output_path, 'rb') as p_in:
            radars_t_radars_dist = pickle.load(p_in)
        p_in.close()
        return radars_t_radars_dist

    radars_t_radars_dist = {}
    for radar_id in nexrad_table:
        radars_t_radars_dist[radar_id] = RadarToOthersDistance(radar_id, nexrad_table)

    if to_save:
        with open(output_path, 'wb') as p_out:
            pickle.dump(radars_t_radars_dist, p_out)
        p_out.close()

    return radars_t_radars_dist


def GetRadarsWithinRadius(radar_id, radius_km, radar_x_radar_dist):
    radius_m = radius_km * 1e3
    if radar_id not in radar_x_radar_dist:
        print(radar_id, " not found in radar x radar distance table.")
        return []

    self_t_other_sorted = radar_x_radar_dist[radar_id]
    num_other = len(self_t_other_sorted)

    idx = 0
    while idx < num_other and self_t_other_sorted[idx][1] <= radius_m:
        idx += 1

    return self_t_other_sorted[:idx]


# TODO
# None
def Main():
    radar_base_folder = "./radar_data"
    radar_folder_table = "radar_table_files"
    radar_table = GetNexradTable(radar_base_folder=radar_base_folder, radar_folder_table=radar_folder_table,
                                 save_table=True, force_collection=False)
    radar_x_radar = RadarXRadarDistance(radar_table, radar_base_folder, to_save=True, force_update=False)
    print(radar_x_radar['KOHX'][:10])
    tmp = GetRadarsWithinRadius('KOHX', 200, radar_x_radar)
    print(tmp)

# Main()
