import os
import warnings
from VADMaskEnum import VADMask
from AnalyzeWind import *
from RadarXSoundingUtils import RadarXSoundingDistance

def Main():
    # Radar/wind source.
    radar_data_file = 'KOHX20180501_070531_V06.ar2v'#'KOHX20180503_104250_V06.ar2v' #'KHTX20180501_110357_V06.ar2v' #'KHPX20180515_200448_V06.ar2v' #'KOHX20180503_180336_V06.ar2v' #'KOHX20180501_042926_V06.ar2v'  # "KENX20180424_120113_V06.ar2v"  # "KENX20180410_124739_V06.ar2v"  # "KENX20180502_053506_V06.ar2v" #
    radar_data_folder = "./radar_data"
    batch_folder =  "KOHX_20180501_20180515" #"KOHX_20180503_test_data" #"KHPX_20180501_20180531" #"KOHX_20180501_20180515"  # "KOHX_20180501_20180515"  # "KENX_20180401_20180430"  # "KENX_20180501_20180531" #"KOHX_20180601_20180630"
    hca_data_folder = "./level3_data"
    allowed_el_hca = {0.5: "N0H", 1.5: "N1H", 2.5: "N2H", 3.5: "N3H"}

    radar_t_sounding = RadarXSoundingDistance(nexrad_table=None, sounding_table=None, output_folder="./radar_data")
    print(radar_t_sounding['KOHX'])
    sounding_log_dir = "./sounding_logs"
    is_batch = True
    gt_wind_source = WindSource.rap_130

    # RAP
    rap_folder = r"./atmospheric_model_data/rap_130_20180501_20180531"

    # Model.
    norm_stats_file = "./models/ridge_bi/mean_std_for_normalization_2.pkl"
    clf_file = "./models/ridge_bi/RidgeRegModels_SGD_1.pkl"
    correct_hca_weather = True
    save_wind_figure = False
    biw_norm_stats_file = "./models/ridge_biw/mean_std_for_normalization_with_weather.pkl"
    biw_clf_file = "./models/ridge_biw/RidgeRegModels_SGD_weather.pkl"

    # VAD options
    vad_jobs = [VADMask.birds, VADMask.insects, VADMask.biological] #, VADMask.weather] #, VADMask.biological]
    # vad_debug_params = {'show_plot': True, 'vad_heights': np.array([200])}
    vad_debug_params = False

    figure_dir = "./figures/AnalyzeWindMainOutput"  # TODO specify proper path.

    radar_data_file, radar_data_folder, data_table, radar_obj, hca_vol = PrepareAnalyzeWindInputs(radar_data_file,
                                                                                                  batch_folder,
                                                                                                  radar_data_folder,
                                                                                                  hca_data_folder,
                                                                                                  clf_file, is_batch,
                                                                                                  norm_stats_file=norm_stats_file,
                                                                                                  correct_hca_weather=correct_hca_weather,
                                                                                                  biw_norm_stats_file=biw_norm_stats_file,
                                                                                                  biw_clf_file=biw_clf_file,
                                                                                                  allowed_el_hca=allowed_el_hca)

    vad_profiles, sounding_df, echo_dist = AnalyzeWind(radar_data_file, radar_data_folder, hca_data_folder,
                                                       radar_t_sounding, sounding_log_dir,
                                                       norm_stats_file, clf_file, vad_jobs, figure_dir=figure_dir,
                                                       match_radar_and_sounding_grid=True,
                                                       save_wind_figure=save_wind_figure,
                                                       vad_debug_params=vad_debug_params,
                                                       radar=radar_obj, hca_vol=hca_vol, data_table=data_table,
                                                       l3_filelist=None, ground_truth_source=gt_wind_source,
                                                       rap_folder=rap_folder, correct_hca_weather=correct_hca_weather,
                                                       biw_norm_stats_file=biw_norm_stats_file,
                                                       biw_clf_file=biw_clf_file, allowed_el_hca=allowed_el_hca)

    return


Main()
