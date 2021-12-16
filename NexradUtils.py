import os
import pickle
import fnmatch
from RadarHCAUtils import *
from AnalyzeWind import classify_echoes

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
                     height_binsize=0.04, clf_file=None):
    # Read radar volume.
    try:
        print("Opening ", os.path.join(batch_folder_path_radar, radar_subpath))
        radar_obj = pyart.io.read_nexrad_archive(os.path.join(batch_folder_path_radar, radar_subpath))
    except:
        print("Read failed. Skipping to next iteration")
        return None, None, None

    # Read HCA volume.
    radar_filename = os.path.splitext(os.path.split(radar_subpath)[1])[0]
    try:
        hca_vol = GetHcaVolFromFileList(batch_folder_path_l3, radar_filename, l3_files_dic)
    except:
        print('Read failed for expected l3 file. Might not exist or might be corrupted. Skipping to next iteration.')
        return None, None, None

    data_table = MergeRadarAndHCAUpdate(radar_obj, hca_vol, max_range)
    data_table["mask_differential_reflectivity"] = data_table["differential_reflectivity"] > -8.0
    data_table["hca_bio"] = data_table["hca"] == 10.0
    data_table["hca_weather"] = np.logical_and(data_table["hca"] >= 30.0, data_table["hca"] <= 100.0)
    data_table["height"] = data_table["range"] * np.sin(data_table["elevation"] * np.pi / 180)

    data_table["height_bin_meters"] = (np.floor(
        data_table["height"] / height_binsize) + 1) * height_binsize - height_binsize / 2
    data_table["height_bin_meters"] *= 1000

    echo_mask = np.logical_and(data_table["mask_differential_reflectivity"], data_table["hca_bio"])
    X = data_table.loc[
        echo_mask, ['differential_reflectivity', 'differential_phase', 'cross_correlation_ratio']]
    X.rename(columns={"differential_reflectivity": "ZDR"}, inplace=True)
    X.rename(columns={"differential_phase": "pdp"}, inplace=True)
    X.rename(columns={"cross_correlation_ratio": "RHV"}, inplace=True)
    data_table['BIClass'] = -1
    data_table.loc[echo_mask, 'BIClass'] = classify_echoes(X, clf_file)

    return data_table, radar_obj, hca_vol

