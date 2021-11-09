import os
import fnmatch
import pickle
import time
import pyart
from RadarHCAUtils import *
from AnalyzeWind import classify_echoes, AnalyzeWind
from VADMaskEnum import VADMask


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


def GetClfEchoDistribution(batch_folder_path_radar, radar_subpath, batch_folder_path_l3, l3_files_dic, max_range,
                           clf_file):
    # Read radar volume.
    try:
        print("Opening ", os.path.join(batch_folder_path_radar, radar_subpath))
        radar_obj = pyart.io.read_nexrad_archive(os.path.join(batch_folder_path_radar, radar_subpath))
    except:
        print("Read failed. Skipping to next iteration")
        return np.nan, np.nan, np.nan

    # Read HCA volume.
    radar_filename = os.path.splitext(os.path.split(radar_subpath)[1])[0]
    try:
        hca_vol = GetHcaVolFromFileList(batch_folder_path_l3, radar_filename, l3_files_dic)
    except:
        print('Read failed for expected l3 file. Might not exist or might be corrupted. Skipping to next iteration.')
        return np.nan, np.nan, np.nan

    data_table = MergeRadarAndHCAUpdate(radar_obj, hca_vol, max_range)
    data_table["mask_differential_reflectivity"] = data_table["differential_reflectivity"] > -8.0
    data_table["hca_bio"] = data_table["hca"] == 10.0
    data_table["hca_weather"] = np.logical_and(data_table["hca"] >= 30.0, data_table["hca"] <= 100.0)

    echo_mask = np.logical_and(data_table["mask_differential_reflectivity"], data_table["hca_bio"])
    X = data_table.loc[
        echo_mask, ['differential_reflectivity', 'differential_phase', 'cross_correlation_ratio']]
    X.rename(columns={"differential_reflectivity": "ZDR"}, inplace=True)
    X.rename(columns={"differential_phase": "pdp"}, inplace=True)
    X.rename(columns={"cross_correlation_ratio": "RHV"}, inplace=True)
    data_table['BIClass'] = -1
    data_table.loc[echo_mask, 'BIClass'] = classify_echoes(X, clf_file)

    return np.sum(data_table['BIClass'] == 1), np.sum(data_table['BIClass'] == 0), np.sum(
        data_table['hca_weather']), radar_obj, hca_vol, data_table

def GetTimeHourUTC(some_str: str) -> float:
    idx_timestart = some_str.find('_') + 1
    hh = some_str[idx_timestart:idx_timestart + 2]
    mm = some_str[idx_timestart + 2:idx_timestart + 4]
    ss = some_str[idx_timestart + 4:idx_timestart + 6]
    time_hour = float(hh) + float(mm) / 60 + float(ss) / 3600
    return time_hour

def Main():
    level3_folder = "./level3_data"
    radar_folder = "./radar_data"
    force_output_logging = False
    output_log_dir = "./analysis_output_logs"
    figure_dir = './figures'

    batch_folder = "KOHX_20180501_20180515"
    date_pattern = "*KOHX201805{}*_V06.*"
    start_day = 1
    stop_day = 15
    max_range = 400  # in km

    # Model.
    norm_stats_file = "./models/ridge_bi/mean_std_for_normalization_1.pkl"
    clf_file = "./models/ridge_bi/RidgeRegModels_SGD_1.pkl"

    # Wind analysis.
    radar_t_sounding = {'KHTX': 'BNA', 'KTLX': 'LMN', 'KOHX': 'BNA'}
    station_infos = {'LMN': ('74646', 'Lamont, Oklahoma'), 'BNA': ('72327', 'Nashville, Tennessee')}
    sounding_log_dir = "./sounding_logs"
    vad_jobs = [VADMask.birds, VADMask.insects, VADMask.weather]
    delta_time_hr = 1 / 6
    time_window = {'noon': (12 - delta_time_hr, 12 + delta_time_hr),
                   'midnight': (24 - delta_time_hr, (24 + delta_time_hr) % 24)}

    batch_folder_path_l3 = os.path.join(level3_folder, batch_folder)
    batch_folder_path_radar = os.path.join(radar_folder, batch_folder)

    radar_scans_day = GetFileListRadar(batch_folder_path_radar, start_day=start_day, stop_day=stop_day,
                                       date_pattern=date_pattern)
    l3_files_dic = GetFileListL3(batch_folder_path_l3)

    days = list(radar_scans_day.keys())
    days.sort()

    for curr_day in days:
        # Should take ~17 minutes per day.
        print("Processing ", curr_day, " ........")
        radar_scans = radar_scans_day[curr_day]  # ~200 scans

        bird_count_scan = []
        insect_count_scan = []
        weather_count_scan = []

        for radar_subpath in radar_scans:  # Takes ~5 seconds for 1 scan
            # Echo distribution.
            n_birds, n_insects, n_weather, radar_obj, hca_vol, data_table = GetClfEchoDistribution(
                batch_folder_path_radar,
                radar_subpath,
                batch_folder_path_l3, l3_files_dic,
                max_range,
                clf_file)

            bird_count_scan.append(n_birds)
            insect_count_scan.append(n_insects)
            weather_count_scan.append(n_weather)

            # Analyze wind
            time_hour = GetTimeHourUTC(radar_subpath)
            is_near_sounding = (time_hour > time_window['noon'][0] and time_hour < time_window['noon'][1]) or (
                    time_hour > time_window['midnight'][0] or time_hour < time_window['midnight'][1])

            if is_near_sounding:
                target_folder = os.path.split(radar_subpath)
                target_file = target_folder[1]
                target_folder = os.path.join(radar_folder, batch_folder, target_folder[0])
                wind_figure_dir = os.path.join(figure_dir, batch_folder, curr_day, 'radar_sounding_wind')

                print('Analyzing wind for ', target_file, ' ....')

                vad_profiles, sounding_df = AnalyzeWind(target_file, target_folder, batch_folder_path_l3,
                                                        radar_t_sounding,
                                                        station_infos, sounding_log_dir,
                                                        norm_stats_file, clf_file, vad_jobs, figure_dir=wind_figure_dir,
                                                        match_radar_and_sounding_grid=True,
                                                        save_wind_figure=True, radar=radar_obj, hca_vol=hca_vol,
                                                        data_table=data_table, l3_filelist=None)

        echo_count_scan = {VADMask.birds: bird_count_scan, VADMask.insects: insect_count_scan,
                           VADMask.weather: weather_count_scan}
        result = (radar_scans, echo_count_scan)

        output_log_path = os.path.join(output_log_dir, ''.join([curr_day, "_echo_count.pkl"]))
        if force_output_logging or (not os.path.exists(output_log_path)):
            with open(output_log_path, 'wb') as p_out:
                pickle.dump(result, p_out)
            p_out.close()


Main()
