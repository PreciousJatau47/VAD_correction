import os
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

def Main():
    level3_folder = "./level3_data"
    batch_folder = "20180501_20180515"
    batch_folder_path = os.path.join(level3_folder,batch_folder)
    filelist_dic = GetFileListL3(batch_folder_path)

    radar_data_file = "KOHX20180502_020640_V06"    # TODO obtain from filelist
    hca_vol = GetHcaVolFromFileList(batch_folder_path, radar_data_file, filelist_dic)

Main()