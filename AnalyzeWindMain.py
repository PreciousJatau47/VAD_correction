import os
from VADMaskEnum import VADMask
from AnalyzeWind import *

def Main():
    # Radar/Sounding.
    radar_data_file = "KOHX20150503_050828_V06.gz"  # "KOHX20180506_231319_V06"
    radar_data_folder = "./radar_data/KOHX20150503"
    hca_data_folder = "./hca_data"
    radar_t_sounding = {'KHTX': 'BNA', 'KTLX': 'LMN', 'KOHX': 'BNA'}
    station_infos = {'LMN': ('74646', 'Lamont, Oklahoma'), 'BNA': ('72327', 'Nashville, Tennessee')}
    sounding_log_dir = "./sounding_logs"

    # Model.
    norm_stats_file = "./models/ridge_bi/mean_std_for_normalization_1.pkl"
    clf_file = "./models/ridge_bi/RidgeRegModels_SGD_1.pkl"

    # VAD options
    vad_jobs = [VADMask.birds, VADMask.insects, VADMask.weather]
    # vad_jobs = [VADMask.insects]

    figure_dir = os.path.join('./figures', radar_data_file[:12])

    vad_profiles, sounding_df, echo_dist = AnalyzeWind(radar_data_file, radar_data_folder, hca_data_folder,
                                                       radar_t_sounding, station_infos, sounding_log_dir,
                                                       norm_stats_file, clf_file, vad_jobs, figure_dir=figure_dir,
                                                       match_radar_and_sounding_grid=True,
                                                       save_wind_figure=False)


Main()
