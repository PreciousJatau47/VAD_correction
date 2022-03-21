import os
import warnings
from VADMaskEnum import VADMask
from AnalyzeWind import *


def Main():
    # Radar/Sounding.
    radar_data_file = 'KOHX20180501_000411_V06.ar2v'  # "KENX20180424_120113_V06.ar2v"  # "KENX20180410_124739_V06.ar2v"  # "KENX20180502_053506_V06.ar2v" #
    radar_data_folder = "./radar_data"
    batch_folder = "KOHX_20180501_20180515"  # "KENX_20180401_20180430"  # "KENX_20180501_20180531" #"KOHX_20180601_20180630"
    hca_data_folder = "./level3_data"
    radar_t_sounding = {'KHTX': 'BNA', 'KTLX': 'LMN', 'KOHX': 'BNA', 'KENX': 'ALB'}
    station_infos = {'LMN': ('74646', 'Lamont, Oklahoma'), 'BNA': ('72327', 'Nashville, Tennessee'),
                     'ALB': ('72518', 'Albany, New York')}
    sounding_log_dir = "./sounding_logs"
    is_batch = True

    # RAP
    rap_folder = r"./atmospheric_model_data/rap_130_20180501_20180515"

    # Model.
    norm_stats_file = "./models/ridge_bi/mean_std_for_normalization_1.pkl"
    clf_file = "./models/ridge_bi/RidgeRegModels_SGD_1.pkl"

    # VAD options
    vad_jobs = [VADMask.birds, VADMask.insects, VADMask.weather]
    # vad_debug_params = {'show_plot': True, 'vad_heights': np.array([200])}
    vad_debug_params = False

    figure_dir = "./figures/temp_EM"  # TODO specify proper path.

    radar_data_file, radar_data_folder, data_table, radar_obj, hca_vol = PrepareAnalyzeWindInputs(radar_data_file,
                                                                                                  batch_folder,
                                                                                                  radar_data_folder,
                                                                                                  hca_data_folder,
                                                                                                  clf_file, is_batch)

    vad_profiles, sounding_df, echo_dist = AnalyzeWind(radar_data_file, radar_data_folder, hca_data_folder,
                                                       radar_t_sounding, station_infos, sounding_log_dir,
                                                       norm_stats_file, clf_file, vad_jobs, figure_dir=figure_dir,
                                                       match_radar_and_sounding_grid=True,
                                                       save_wind_figure=False, vad_debug_params=vad_debug_params,
                                                       radar=radar_obj, hca_vol=hca_vol, data_table=data_table,
                                                       l3_filelist=None, ground_truth_source=WindSource.rap_130,
                                                       rap_folder = rap_folder)
    return


Main()
