import os
from VADMaskEnum import VADMask
from AnalyzeWind import *
from NexradUtils import *


def Main():
    # Radar/Sounding.
    radar_data_file = "KOHX20180503_183251_V06.ar2v"  # "KOHX20180506_231319_V06"
    radar_data_folder = "./radar_data"
    batch_folder = "KOHX_20180501_20180515"
    hca_data_folder = "./level3_data"
    radar_t_sounding = {'KHTX': 'BNA', 'KTLX': 'LMN', 'KOHX': 'BNA'}
    station_infos = {'LMN': ('74646', 'Lamont, Oklahoma'), 'BNA': ('72327', 'Nashville, Tennessee')}
    sounding_log_dir = "./sounding_logs"
    is_batch = True

    # Model.
    norm_stats_file = "./models/ridge_bi/mean_std_for_normalization_1.pkl"
    clf_file = "./models/ridge_bi/RidgeRegModels_SGD_1.pkl"

    # VAD options
    vad_jobs = [VADMask.birds, VADMask.insects, VADMask.weather]
    vad_debug_params = {'show_plot': True, 'vad_heights': np.array([200])}

    figure_dir = "./figures/temp_EM"    # TODO specify proper path.

    radar_obj = None
    hca_vol = None
    data_table = None

    if is_batch:
        start_day = int(radar_data_file[10:12])
        stop_day = start_day

        date_pattern = "*{}_V06*".format("".join([radar_data_file[:10], "{}*"]))
        hca_data_folder = os.path.join(hca_data_folder, batch_folder)
        batch_folder_path_radar = os.path.join(radar_data_folder, batch_folder)
        radar_scans_day = GetFileListRadar(batch_folder_path_radar, start_day=start_day, stop_day=stop_day,
                                           date_pattern=date_pattern)
        l3_files_dic = GetFileListL3(hca_data_folder)

        # find subpath for radar file
        radar_subpath = None
        for key in radar_scans_day.keys():
            for element in radar_scans_day[key]:
                if element.endswith(radar_data_file):
                    radar_subpath = element
                    break

        data_table, radar_obj, hca_vol = PrepareDataTable(batch_folder_path_radar,
                                                          radar_subpath,
                                                          hca_data_folder, l3_files_dic,
                                                          max_range=400,
                                                          clf_file=clf_file)

        target_folder = os.path.split(radar_subpath)
        radar_data_file = target_folder[1]
        radar_data_folder = os.path.join(radar_data_folder, batch_folder, target_folder[0])
    else:
        radar_data_folder = os.path.join(radar_data_folder, batch_folder)


    vad_profiles, sounding_df, echo_dist = AnalyzeWind(radar_data_file, radar_data_folder, hca_data_folder,
                                                       radar_t_sounding, station_infos, sounding_log_dir,
                                                       norm_stats_file, clf_file, vad_jobs, figure_dir=figure_dir,
                                                       match_radar_and_sounding_grid=True,
                                                       save_wind_figure=False, vad_debug_params=vad_debug_params,
                                                       radar=radar_obj, hca_vol=hca_vol, data_table=data_table,
                                                       l3_filelist=None)


Main()
