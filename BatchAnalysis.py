import os
import fnmatch
import pickle
from RadarHCAUtils import *

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

    for current_day in range(start_day, stop_day+1):
        current_day_str = '0' + str(current_day) if current_day < 10 else str(current_day)
        pattern = date_pattern.format(current_day_str)
        filtered_files = fnmatch.filter(filelists, pattern)
        filtered_files.sort(key=key_fn)
        filelist_dic[pattern[1:-1]] = filtered_files

    return filelist_dic

def Main():
    level3_folder = "./level3_data"
    radar_folder = "./radar_data"
    batch_folder = "KOHX_20180501_20180515"
    batch_folder_path_l3 = os.path.join(level3_folder , batch_folder)
    batch_folder_path_radar = os.path.join(radar_folder, batch_folder)

    date_pattern = "*KOHX201805{}*"
    radar_scans_day = GetFileListRadar(batch_folder_path_radar, start_day=1, stop_day=2, date_pattern=date_pattern)
    l3_files_dic = GetFileListL3(batch_folder_path_l3)

    days = list(radar_scans_day.keys())
    days.sort()

    curr_day = days[0]  # TODO to be iterated over.

    radar_subpath = radar_scans_day[curr_day][0]    # TODO to be iterated over.
    print(radar_subpath)

    radar_filename = os.path.splitext(os.path.split(radar_subpath)[1])[0]
    print(radar_filename)

    hca_vol = GetHcaVolFromFileList(batch_folder_path_l3, radar_filename, l3_files_dic)


Main()