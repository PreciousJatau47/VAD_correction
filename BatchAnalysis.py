import os
import fnmatch
import pickle
import time
import datetime
import pyart
from RadarHCAUtils import *
from TrueWindEnum import *
from AnalyzeWind import classify_echoes, AnalyzeWind
from VADMaskEnum import VADMask
from NexradUtils import *
from RadarXSoundingUtils import RadarXSoundingDistance
from AirspeedAnalysisUtils import GetAirSpeedScan, UpdateWindError

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 15}
plt.rc('font', **font)


def AccumulateOrAverageSelector(prev_scan_time, radar_scan_time, delta_time):
    # Decide between accumulate and average.
    if prev_scan_time < 0:
        to_accumm = True
    else:
        prev_time_partition = prev_scan_time // delta_time
        radar_time_partition = radar_scan_time // delta_time
        to_accumm = prev_time_partition == radar_time_partition
    return to_accumm


"TODO(pjatau) Add documentation."


def AccumulateAndAverage(to_accumm, radar_file, files_accumm, vad_profiles, vad_profiles_accumm, averaged_profiles,
                         height_bin_size):
    if to_accumm:
        files_accumm.append(radar_file)
        for echo_type in vad_profiles:
            vad_profiles_accumm[echo_type] = vad_profiles_accumm[echo_type].append(vad_profiles[echo_type])
        return vad_profiles_accumm, None, files_accumm

    averaged_profiles = vad_profiles_accumm.copy()

    for echo_type in vad_profiles_accumm:
        # Average.
        if averaged_profiles[echo_type] is None or averaged_profiles[echo_type].empty:
            continue
        averaged_profiles[echo_type]["height_bin"] = averaged_profiles[echo_type]["height"] // height_bin_size
        averaged_profiles[echo_type] = averaged_profiles[echo_type].groupby(["height_bin"]).median()

    # Restart accumulation.
    if vad_profiles is None:
        vad_profiles_accumm = {VADMask.birds: pd.DataFrame(), VADMask.insects: pd.DataFrame(),
                               VADMask.weather: pd.DataFrame(), VADMask.biological: pd.DataFrame()}
    else:
        vad_profiles_accumm = vad_profiles.copy()

    if not files_accumm:
        return vad_profiles_accumm, averaged_profiles, [""]

    return vad_profiles_accumm, averaged_profiles, [files_accumm[(len(files_accumm) - 1) // 2]]


def TestAccumulateAndAverage():
    radar_files = ['0001/KOHX20180503_181320_V06.ar2v', '0001/KOHX20180503_182320_V06.ar2v',
                   '0001/KOHX20180503_182920_V06.ar2v', '0001/KOHX20180503_183120_V06.ar2v']
    table_values = [1, 2, 3, 4]

    delta_time = 0.5
    vad_profiles_accumm = {VADMask.birds: pd.DataFrame(), VADMask.insects: pd.DataFrame(),
                           VADMask.weather: pd.DataFrame(), VADMask.biological: pd.DataFrame()}
    files_accumm = []
    averaged_profiles = None
    height_bin_size = 40
    prev_scan_time = -1

    for idx in range(len(radar_files)):
        # Create test data frame.
        curr_value = table_values[idx]
        curr_df = pd.DataFrame(
            {'height': [curr_value for i in range(3)], 'wind_U': [curr_value for i in range(3)],
             'wind_V': [curr_value for i in range(3)]})
        vad_profiles = {VADMask.birds: curr_df.copy(), VADMask.insects: curr_df.copy(), VADMask.weather: curr_df.copy(),
                        VADMask.biological: curr_df.copy()}

        # Accumulate and average.
        radar_file = radar_files[idx]
        radar_scan_time = GetTimeHourUTC(radar_file)

        # Decide between accumulate and average.
        if prev_scan_time < 0:
            to_accumm = True
        else:
            prev_time_partition = prev_scan_time // delta_time
            radar_time_partition = radar_scan_time // delta_time
            to_accumm = prev_time_partition == radar_time_partition

        vad_profiles_accumm, averaged_profiles, mid_file_log = AccumulateAndAverage(to_accumm=to_accumm,
                                                                                    radar_file=radar_file,
                                                                                    files_accumm=files_accumm,
                                                                                    vad_profiles=vad_profiles,
                                                                                    vad_profiles_accumm=vad_profiles_accumm,
                                                                                    averaged_profiles=averaged_profiles,
                                                                                    height_bin_size=height_bin_size)
        if not to_accumm:
            # Test for middle radar file
            assert mid_file_log[
                       0] == radar_files[1], "Median radar file does not match expected file."

            # Test for averaged df.
            expected_averaged_df = pd.DataFrame({'height': [2], 'wind_U': [2],
                                                 'wind_V': [2]})
            expected_averaged_profiles = {VADMask.birds: expected_averaged_df.copy(),
                                          VADMask.insects: expected_averaged_df.copy(),
                                          VADMask.weather: expected_averaged_df.copy(),
                                          VADMask.biological: expected_averaged_df.copy()}
            for echo_type in averaged_profiles:
                assert averaged_profiles[echo_type].equals(
                    expected_averaged_profiles[
                        echo_type]), "Averaged profiles do not match expected profiles."

            # Test for new accumulation.
            expected_new_accumm_df = pd.DataFrame(
                {'height': [table_values[-1] for i in range(3)],
                 'wind_U': [table_values[-1] for i in range(3)],
                 'wind_V': [table_values[-1] for i in range(3)]})
            expected_new_accum_profiles = {VADMask.birds: expected_new_accumm_df.copy(),
                                           VADMask.insects: expected_new_accumm_df.copy(),
                                           VADMask.weather: expected_new_accumm_df.copy(),
                                           VADMask.biological: expected_new_accumm_df.copy()}
            for echo_type in expected_new_accum_profiles:
                assert vad_profiles_accumm[echo_type].equals(
                    expected_new_accum_profiles[echo_type]), "New accumulated profiles do not match expected profiles."

        prev_scan_time = radar_scan_time


# def TempMain():
#     TestAccumulateAndAverage()
#
# TempMain()

def E2EWindAnalysis(batch_folder, radar_folder, level3_folder, start_day, stop_day, date_pattern, max_range,
                    max_height_VAD, time_window, clf_file, radar_t_sounding, sounding_log_dir,
                    norm_stats_file, vad_jobs, figure_dir, vad_sounding_dir, echo_count_log_dir, save_ppi_plots,
                    force_output_logging,
                    ground_truth_source=WindSource.sounding,
                    rap_folder=None, correct_hca_weather=False, biw_norm_stats_file=None,
                    biw_clf_file=None, log_dir='./', experiment_name='', allowed_el_hca=None, use_vad_weights=False,
                    clf_purity_threshold=0.5,
                    min_required_nsamples=720, height_binsize=0.04):

    # Echo count options
    echo_count_log_dir = os.path.join(echo_count_log_dir, batch_folder)
    if correct_hca_weather:
        echo_count_log_dir = os.path.join(echo_count_log_dir, 'hca_weather_corrected')
    else:
        echo_count_log_dir = os.path.join(echo_count_log_dir, 'hca_default')

    if not os.path.isdir(echo_count_log_dir):
        os.makedirs(echo_count_log_dir)

    # Colormap for visualizing radar products.
    color_map = GetDataTableColorMap()
    color_map.pop('hca')

    # AnalyzeWind options.
    wind_source_desc = GetWindSourceDescription(ground_truth_source)
    wind_source_desc = wind_source_desc.replace(' ', '_')
    if correct_hca_weather:
        vad_sounding_dir = os.path.join(vad_sounding_dir, batch_folder, 'hca_weather_corrected', wind_source_desc)
    else:
        vad_sounding_dir = os.path.join(vad_sounding_dir, batch_folder, 'hca_default', wind_source_desc)

    if not os.path.isdir(vad_sounding_dir):
        os.makedirs(vad_sounding_dir)

    continue_from_last_checkpoint = bool(experiment_name)
    if continue_from_last_checkpoint:
        experiment_dir = os.path.join(log_dir, experiment_name)
        continue_from_last_checkpoint = continue_from_last_checkpoint and os.path.isdir(experiment_dir)

    if continue_from_last_checkpoint:
        log_suffix = '_weights_{}_threshold_{}'.format(int(use_vad_weights), int(clf_purity_threshold * 100))
        log_path = os.path.join(experiment_dir, ''.join([batch_folder, log_suffix, '.pkl']))

        if os.path.isfile(log_path): # Continue previous subexperiment
            p_in = open(log_path, "rb")
            _, idx_days_last_log = pickle.load(p_in)
            idx_days_last_log += 1
            print(_.shape)
            p_in.close()
        else: # Start new subexperiment
            idx_days_last_log = 0
    else:
        current_time = datetime.datetime.now()
        datetime_id = "{}{}{}_{}".format(current_time.year, current_time.month, current_time.day, current_time.hour)
        experiment_name = batch_folder + "_launched_" + datetime_id
        experiment_dir = os.path.join(log_dir, experiment_name)
        idx_days_last_log = 0

    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    # Airspeed analysis options
    error_fn = lambda yTrue, yPred: np.abs(yPred - yTrue)
    reduce_fn = lambda scores: np.nanmean(scores)

    # TODO(pjatau) move out, maybe to function parameters.
    # Airspeed averaging options
    height_bin_size_m = int(height_binsize * 1000)
    delta_time = 1  # 0.5

    sounding_df = None
    echo_dist_VAD = None

    # Get radar and l3 files.
    batch_folder_path_l3 = os.path.join(level3_folder, batch_folder)
    batch_folder_path_radar = os.path.join(radar_folder, batch_folder)
    radar_scans_day = GetFileListRadar(batch_folder_path_radar, start_day=start_day, stop_day=stop_day,
                                       date_pattern=date_pattern)
    l3_files_dic = GetFileListL3(batch_folder_path_l3)

    days = list(radar_scans_day.keys())
    days.sort()

    first_day = days[0]
    for idx_days in range(idx_days_last_log, len(days)):
        # for idx_days in range(12, 12+1): # TODO(pjatau) erase me
        curr_day = days[idx_days]
        print("Processing ", curr_day, " ........")
        radar_scans = radar_scans_day[curr_day]  # ~200 scans
        # wind_error_df = pd.DataFrame(
        #     columns=['file_name', 'airspeed_birds', 'airspeed_insects', 'height_m', 'num_insects_height',
        #              'num_birds_height',
        #              'prop_birds', 'prop_insects',
        #              'prop_weather', 'prop_weather_scan', 'insect_prop_bio', "radar", "year", "month", "day",
        #              "time_hour"])
        wind_error_df = pd.DataFrame()
        wind_error_averaged_df = wind_error_df.copy()

        # Initialize bird, insect and weather count
        bird_count_scan = []
        insect_count_scan = []
        weather_count_scan = []

        prev_scan_time = -1
        vad_profiles_accumm = None
        averaged_profiles = None
        files_accumm = []

        for radar_subpath in radar_scans:  # Takes ~5 seconds for 1 scan
            print("Processing ", radar_subpath, " ........")
            time_hour = GetTimeHourUTC(radar_subpath)
            is_near_sounding = (time_hour > time_window['noon'][0] and time_hour < time_window['noon'][1]) or (
                    time_hour > time_window['midnight'][0] or time_hour < time_window['midnight'][1])

            # Load data. Takes ~7s.
            data_table, radar_obj, hca_vol = PrepareDataTable(batch_folder_path_radar, radar_subpath,
                                                              batch_folder_path_l3, l3_files_dic, max_range=max_range,
                                                              clf_file=clf_file, norm_stats_file=norm_stats_file,
                                                              correct_hca_weather=correct_hca_weather,
                                                              max_height_VAD=1000,
                                                              biw_norm_stats_file=biw_norm_stats_file,
                                                              biw_clf_file=biw_clf_file, allowed_el_hca=allowed_el_hca,
                                                              height_binsize=height_binsize)

            # Echo distribution.
            if data_table is None:
                n_birds, n_insects, n_weather = np.nan, np.nan, np.nan
            else:
                n_birds = np.sum(data_table['BIClass'] == 1)
                n_insects = np.sum(data_table['BIClass'] == 0)
                n_weather = np.sum(data_table['hca_weather'])

            bird_count_scan.append(n_birds)
            insect_count_scan.append(n_insects)
            weather_count_scan.append(n_weather)

            # Visualize scan.
            if save_ppi_plots and radar_obj is not None:
                data_table.loc[data_table['BIClass'] == -1, 'BIClass'] = np.nan

                scan_figure_dir = os.path.join(figure_dir, batch_folder, curr_day)
                scan_name = os.path.split(radar_subpath)[1]
                scan_name = os.path.splitext(scan_name)[0]

                total_echoes = n_birds + n_insects + n_weather
                prop_birds = round(n_birds / total_echoes * 100)
                prop_insects = round(n_insects / total_echoes * 100)
                prop_weather = round(n_weather / total_echoes * 100)
                title_suffix = "{}% birds, {}% insects, {}% weather.".format(prop_birds, prop_insects, prop_weather)

                # 16s per plot item.

                c_map = {}
                c_map['reflectivity'] = color_map['reflectivity']
                c_map['hca_weather'] = color_map['hca_weather']
                c_map['BIClass'] = color_map['BIClass']
                color_map = c_map
                # VisualizeDataTable(data_table=data_table, color_map=color_map, output_folder=scan_figure_dir,
                #                    scan_name=scan_name, title_suffix=title_suffix, combine_plots=True,
                #                    correct_hca_weather=correct_hca_weather)
                # VisualizeDataTable(data_table=data_table, color_map=color_map, output_folder=scan_figure_dir,
                #                    scan_name=scan_name, title_suffix=title_suffix, combine_plots=False,
                #                    correct_hca_weather=False)

            if is_near_sounding or ground_truth_source == WindSource.rap_130:

                if radar_obj is not None:
                    # Prepare inputs for wind analysis.
                    target_folder = os.path.split(radar_subpath)
                    target_file = target_folder[1]
                    target_file_no_ext = os.path.splitext(target_file)[0]
                    target_folder = os.path.join(radar_folder, batch_folder, target_folder[0])

                    if correct_hca_weather:
                        wind_figure_dir = os.path.join(figure_dir, batch_folder, curr_day, 'radar_sounding_wind',
                                                       'hca_weather_corrected', wind_source_desc)
                    else:
                        wind_figure_dir = os.path.join(figure_dir, batch_folder, curr_day, 'radar_sounding_wind',
                                                       wind_source_desc)

                    vad_sounding_path = os.path.join(vad_sounding_dir, ''.join([target_file_no_ext, '_wind', '.pkl']))

                    # Analyze wind. Takes ~16s.
                    print('Analyzing wind for ', target_file, ' ....')
                    vad_profiles, sounding_df, echo_dist_VAD = AnalyzeWind(target_file, target_folder,
                                                                           batch_folder_path_l3,
                                                                           radar_t_sounding,
                                                                           sounding_log_dir, norm_stats_file, clf_file,
                                                                           vad_jobs, figure_dir=wind_figure_dir,
                                                                           max_range=max_range,
                                                                           max_height_VAD=max_height_VAD,
                                                                           match_radar_and_sounding_grid=True,
                                                                           save_wind_figure=True, radar=radar_obj,
                                                                           hca_vol=hca_vol,
                                                                           data_table=data_table, l3_filelist=None,
                                                                           ground_truth_source=ground_truth_source,
                                                                           rap_folder=rap_folder,
                                                                           correct_hca_weather=correct_hca_weather,
                                                                           biw_norm_stats_file=biw_norm_stats_file,
                                                                           biw_clf_file=biw_clf_file,
                                                                           use_vad_weights=use_vad_weights,
                                                                           clf_purity_threshold=clf_purity_threshold,
                                                                           min_required_nsamples=min_required_nsamples,
                                                                           height_binsize=height_binsize)

                    # Initialize accumulator for VAD profiles.
                    if vad_profiles_accumm is None:
                        vad_profiles_accumm = {echo_type: pd.DataFrame() for echo_type in vad_jobs}
                        averaged_profiles = vad_profiles_accumm.copy()

                    radar_scan_time = GetTimeHourUTC(radar_subpath)
                    to_accumm = AccumulateOrAverageSelector(prev_scan_time, radar_scan_time, delta_time)
                    vad_profiles_accumm, averaged_profiles, mid_file_log = AccumulateAndAverage(to_accumm=to_accumm,
                                                                                                radar_file=target_file,
                                                                                                files_accumm=files_accumm,
                                                                                                vad_profiles=vad_profiles,
                                                                                                vad_profiles_accumm=vad_profiles_accumm,
                                                                                                averaged_profiles=averaged_profiles,
                                                                                                height_bin_size=height_bin_size_m)

                    if not to_accumm:  # averaged scan available.
                        wind_error_averaged_df = UpdateWindError(wind_error_averaged_df, mid_file_log[0],
                                                                 averaged_profiles,
                                                                 sounding_df,
                                                                 echo_dist_VAD, error_fn,
                                                                 reduce_fn,
                                                                 ground_truth_source, figure_dir, batch_folder, n_birds,
                                                                 n_insects, n_weather)
                        files_accumm = []

                    prev_scan_time = radar_scan_time

                    # Get airspeeds for current scan and update wind error.
                    wind_error_df = UpdateWindError(wind_error_df, target_file, vad_profiles, sounding_df,
                                                    echo_dist_VAD, error_fn,
                                                    reduce_fn,
                                                    ground_truth_source, figure_dir, batch_folder, n_birds, n_insects,
                                                    n_weather)

                    # Save output from AnalyzeWind.
                    with open(vad_sounding_path, 'wb') as p_out:
                        pickle.dump({'VAD': vad_profiles, 'Sounding': sounding_df, 'echo_dist': echo_dist_VAD}, p_out)
                    p_out.close()
                else:  # radar_obj is None
                    target_folder = os.path.split(radar_subpath)
                    target_file = target_folder[1]

                    # Initialize accumulator for VAD profiles.
                    if vad_profiles_accumm is None:
                        vad_profiles_accumm = {echo_type: pd.DataFrame() for echo_type in vad_jobs}
                        averaged_profiles = vad_profiles_accumm.copy()

                    radar_scan_time = GetTimeHourUTC(radar_subpath)
                    to_accumm = AccumulateOrAverageSelector(prev_scan_time, radar_scan_time, delta_time)

                    # Create empty vad profiles.
                    vad_profiles = {echo_type: pd.DataFrame() for echo_type in vad_jobs}
                    sounding_df = None
                    vad_profiles_accumm, averaged_profiles, mid_file_log = AccumulateAndAverage(to_accumm=to_accumm,
                                                                                                radar_file=target_file,
                                                                                                files_accumm=files_accumm,
                                                                                                vad_profiles=vad_profiles,
                                                                                                vad_profiles_accumm=vad_profiles_accumm,
                                                                                                averaged_profiles=averaged_profiles,
                                                                                                height_bin_size=height_bin_size_m)

                    if not to_accumm:
                        # Need to update averaged df.
                        wind_error_averaged_df = UpdateWindError(wind_error_averaged_df, mid_file_log[0],
                                                                 averaged_profiles,
                                                                 sounding_df,
                                                                 echo_dist_VAD, error_fn,
                                                                 reduce_fn,
                                                                 ground_truth_source, figure_dir, batch_folder, n_birds,
                                                                 n_insects, n_weather)
                        files_accumm = []

                    # Update prev_scan_time.
                    prev_scan_time = radar_scan_time

        # Accumulate and average last batch.
        vad_profiles_accumm, averaged_profiles, mid_file_log = AccumulateAndAverage(to_accumm=False,
                                                                                    radar_file=target_file,
                                                                                    files_accumm=files_accumm,
                                                                                    vad_profiles=None,
                                                                                    vad_profiles_accumm=vad_profiles_accumm,
                                                                                    averaged_profiles=averaged_profiles,
                                                                                    height_bin_size=height_bin_size_m)
        wind_error_averaged_df = UpdateWindError(wind_error_averaged_df, mid_file_log[0], averaged_profiles,
                                                 sounding_df,
                                                 echo_dist_VAD, error_fn,
                                                 reduce_fn,
                                                 ground_truth_source, figure_dir, batch_folder, n_birds,
                                                 n_insects, n_weather)

        echo_count_scan = {VADMask.birds: bird_count_scan, VADMask.insects: insect_count_scan,
                           VADMask.weather: weather_count_scan}
        result = (radar_scans, echo_count_scan)

        # Save echo distribution for the whole scan.
        output_log_path = os.path.join(echo_count_log_dir, ''.join([curr_day, "_echo_count.pkl"]))
        if force_output_logging or (not os.path.exists(output_log_path)):
            with open(output_log_path, 'wb') as p_out:
                pickle.dump(result, p_out)
            p_out.close()

        # Save wind error everyday.
        # TODO(pjatau) define a proper experiment name
        log_suffix = '_weights_{}_threshold_{}'.format(int(use_vad_weights), int(clf_purity_threshold * 100))
        log_path = os.path.join(experiment_dir, ''.join([batch_folder, log_suffix, '.pkl']))
        log_path_averaged = os.path.join(experiment_dir,
                                         ''.join([batch_folder, log_suffix, '_averaged_', str(delta_time), '.pkl']))

        # Save(update) wind error results on a daily basis
        if curr_day == first_day:  # First data set to be logged
            ## Unaveraged data ##
            p_out = open(log_path, 'wb')
            pickle.dump((wind_error_df, idx_days), p_out)
            p_out.close()

            ## Averaged data ##
            p_out = open(log_path_averaged, 'wb')
            pickle.dump((wind_error_averaged_df, idx_days), p_out)
            p_out.close()
        else:
            ## Unaveraged data ##
            # load previously logged data
            p_in = open(log_path, "rb")
            wind_error_logged, idx_days_logged = pickle.load(p_in)
            p_in.close()
            print("Last logged idx: ", idx_days_logged)

            # Update previous log with current dataset.
            print(wind_error_df.shape)
            wind_error_df = wind_error_logged.append(wind_error_df)
            print(wind_error_df.shape)

            p_out = open(log_path, 'wb')
            pickle.dump((wind_error_df, idx_days), p_out)
            p_out.close()

            ## Averaged data ##
            # load previously logged data
            p_in = open(log_path_averaged, "rb")
            wind_error_avg_logged, idx_days_avg_logged = pickle.load(p_in)
            p_in.close()
            print("Last logged idx avg: ", idx_days_avg_logged)

            # Update previous log with current dataset.
            print(wind_error_averaged_df.shape)
            wind_error_averaged_df = wind_error_avg_logged.append(wind_error_averaged_df)
            print(wind_error_averaged_df.shape)

            p_out = open(log_path_averaged, 'wb')
            pickle.dump((wind_error_averaged_df, idx_days), p_out)
            p_out.close()

            # Clean up memory.
            del wind_error_df
            del wind_error_logged
            del wind_error_averaged_df
            del wind_error_avg_logged

    return


def AnalyzeWindBatch(batch_folder, radar_folder, level3_folder, start_day, stop_day, date_pattern, max_range,
                     max_height_VAD, time_window, clf_file, radar_t_sounding, station_infos, sounding_log_dir,
                     norm_stats_file, vad_jobs, figure_dir, vad_sounding_dir, ground_truth_source=WindSource.sounding,
                     rap_folder=None, correct_hca_weather=False, biw_norm_stats_file=None,
                     biw_clf_file=None):
    wind_source_desc = GetWindSourceDescription(ground_truth_source)
    wind_source_desc = wind_source_desc.replace(' ', '_')
    if correct_hca_weather:
        vad_sounding_dir = os.path.join(vad_sounding_dir, batch_folder, 'hca_weather_corrected', wind_source_desc)
    else:
        vad_sounding_dir = os.path.join(vad_sounding_dir, batch_folder, 'hca_default', wind_source_desc)

    if not os.path.isdir(vad_sounding_dir):
        os.makedirs(vad_sounding_dir)

    batch_folder_path_l3 = os.path.join(level3_folder, batch_folder)
    batch_folder_path_radar = os.path.join(radar_folder, batch_folder)
    radar_scans_day = GetFileListRadar(batch_folder_path_radar, start_day=start_day, stop_day=stop_day,
                                       date_pattern=date_pattern)
    l3_files_dic = GetFileListL3(batch_folder_path_l3)

    days = list(radar_scans_day.keys())
    days.sort()

    for curr_day in days:
        print("Analyzing wind for ", curr_day, " ........")
        radar_scans = radar_scans_day[curr_day]  # ~200 scans

        for radar_subpath in radar_scans:
            time_hour = GetTimeHourUTC(radar_subpath)
            is_near_sounding = (time_hour > time_window['noon'][0] and time_hour < time_window['noon'][1]) or (
                    time_hour > time_window['midnight'][0] or time_hour < time_window['midnight'][1])

            # Analyze wind
            if is_near_sounding or ground_truth_source == WindSource.rap_130:
                data_table, radar_obj, hca_vol = PrepareDataTable(batch_folder_path_radar,
                                                                  radar_subpath,
                                                                  batch_folder_path_l3, l3_files_dic,
                                                                  max_range=max_range,
                                                                  clf_file=clf_file, norm_stats_file=norm_stats_file,
                                                                  correct_hca_weather=correct_hca_weather,
                                                                  max_height_VAD=1000,
                                                                  biw_norm_stats_file=biw_norm_stats_file,
                                                                  biw_clf_file=biw_clf_file)

                if radar_obj is not None:
                    target_folder = os.path.split(radar_subpath)
                    target_file = target_folder[1]
                    target_file_no_ext = os.path.splitext(target_file)[0]
                    target_folder = os.path.join(radar_folder, batch_folder, target_folder[0])

                    if correct_hca_weather:
                        wind_figure_dir = os.path.join(figure_dir, batch_folder, curr_day, 'radar_sounding_wind',
                                                       'hca_weather_corrected', wind_source_desc)
                    else:
                        wind_figure_dir = os.path.join(figure_dir, batch_folder, curr_day, 'radar_sounding_wind',
                                                       wind_source_desc)

                    vad_sounding_path = os.path.join(vad_sounding_dir, ''.join([target_file_no_ext, '_wind', '.pkl']))

                    # Takes ~16s.
                    print('Analyzing wind for ', target_file, ' ....')
                    vad_profiles, sounding_df, echo_dist_VAD = AnalyzeWind(target_file, target_folder,
                                                                           batch_folder_path_l3,
                                                                           radar_t_sounding, station_infos,
                                                                           sounding_log_dir, norm_stats_file, clf_file,
                                                                           vad_jobs, figure_dir=wind_figure_dir,
                                                                           max_range=max_range,
                                                                           max_height_VAD=max_height_VAD,
                                                                           match_radar_and_sounding_grid=True,
                                                                           save_wind_figure=True, radar=radar_obj,
                                                                           hca_vol=hca_vol,
                                                                           data_table=data_table, l3_filelist=None,
                                                                           ground_truth_source=ground_truth_source,
                                                                           rap_folder=rap_folder,
                                                                           correct_hca_weather=correct_hca_weather,
                                                                           biw_norm_stats_file=biw_norm_stats_file,
                                                                           biw_clf_file=biw_clf_file)

                    with open(vad_sounding_path, 'wb') as p_out:
                        pickle.dump({'VAD': vad_profiles, 'Sounding': sounding_df, 'echo_dist': echo_dist_VAD}, p_out)
                    p_out.close()
        plt.close('all')
    return


def GetEchoDistributionBatch(batch_folder, radar_folder, level3_folder, start_day, stop_day, date_pattern, max_range,
                             clf_file, output_log_dir, figure_dir, save_ppi_plots, force_output_logging,
                             norm_stats_file=None, correct_hca_weather=False, biw_norm_stats_file=None,
                             biw_clf_file=None):
    output_log_dir = os.path.join(output_log_dir, batch_folder)
    if correct_hca_weather:
        output_log_dir = os.path.join(output_log_dir, 'hca_weather_corrected')
    else:
        output_log_dir = os.path.join(output_log_dir, 'hca_default')

    if not os.path.isdir(output_log_dir):
        os.makedirs(output_log_dir)

    batch_folder_path_l3 = os.path.join(level3_folder, batch_folder)
    batch_folder_path_radar = os.path.join(radar_folder, batch_folder)

    # Colormap for visualizing radar products.
    color_map = GetDataTableColorMap()
    color_map.pop('hca')

    # Get radar, HCA filelists.
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
            time_hour = GetTimeHourUTC(radar_subpath)

            # Load data. Takes ~7s.
            data_table, radar_obj, hca_vol = PrepareDataTable(batch_folder_path_radar, radar_subpath,
                                                              batch_folder_path_l3, l3_files_dic, max_range=max_range,
                                                              clf_file=clf_file, norm_stats_file=norm_stats_file,
                                                              correct_hca_weather=correct_hca_weather,
                                                              max_height_VAD=1000,
                                                              biw_norm_stats_file=biw_norm_stats_file,
                                                              biw_clf_file=biw_clf_file)

            # Echo distribution.
            if data_table is None:
                n_birds, n_insects, n_weather = np.nan, np.nan, np.nan
            else:
                n_birds = np.sum(data_table['BIClass'] == 1)
                n_insects = np.sum(data_table['BIClass'] == 0)
                n_weather = np.sum(data_table['hca_weather'])

            bird_count_scan.append(n_birds)
            insect_count_scan.append(n_insects)
            weather_count_scan.append(n_weather)

            # Visualize scan.
            if save_ppi_plots and radar_obj is not None:
                data_table.loc[data_table['BIClass'] == -1, 'BIClass'] = np.nan

                scan_figure_dir = os.path.join(figure_dir, batch_folder, curr_day)
                scan_name = os.path.split(radar_subpath)[1]
                scan_name = os.path.splitext(scan_name)[0]

                total_echoes = n_birds + n_insects + n_weather
                prop_birds = round(n_birds / total_echoes * 100)
                prop_insects = round(n_insects / total_echoes * 100)
                prop_weather = round(n_weather / total_echoes * 100)
                title_suffix = "{}% birds, {}% insects, {}% weather.".format(prop_birds, prop_insects, prop_weather)

                # 16s per plot item.
                VisualizeDataTable(data_table=data_table, color_map=color_map, output_folder=scan_figure_dir,
                                   scan_name=scan_name, title_suffix=title_suffix, combine_plots=True,
                                   correct_hca_weather=correct_hca_weather)

        echo_count_scan = {VADMask.birds: bird_count_scan, VADMask.insects: insect_count_scan,
                           VADMask.weather: weather_count_scan}
        result = (radar_scans, echo_count_scan)

        output_log_path = os.path.join(output_log_dir, ''.join([curr_day, "_echo_count.pkl"]))
        if force_output_logging or (not os.path.exists(output_log_path)):
            with open(output_log_path, 'wb') as p_out:
                pickle.dump(result, p_out)
            p_out.close()
    return


"""
TODO
Enforce format for batch_folder. 
"""
def Main():
    level3_folder = "./level3_data"
    radar_folder = "./radar_data"
    allowed_el_hca = {0.5: "N0H", 1.5: "N1H", 2.5: "N2H", 3.5: "N3H"}
    force_output_logging = True
    output_log_dir = "./analysis_output_logs"
    figure_dir = './figures'
    save_ppi_plots = False

    batch_folder = "KOHX_20180503_test_data" #"KOHX_20180516_20180531" #"KOHX_20180501_20180515" #"KOHX_20180516_20180531" # 'KLVX_20180501_20180531' #'KHTX_20180501_20180531'
    # date_pattern = "*KENX201804{}*_V06.*"
    start_day = 3 #16
    stop_day = 3 #31
    max_range = 400  # in km.
    max_height_VAD = 1000  # in m.

    date_pattern = "*{}{}".format(batch_folder[:4], batch_folder[5:11])
    date_pattern = "".join([date_pattern, '{}*_V06.*'])

    # Model.
    norm_stats_file = "./models/ridge_bi/mean_std_for_normalization_2.pkl"
    clf_file = "./models/ridge_bi/RidgeRegModels_SGD_1.pkl"
    correct_hca_weather = True
    biw_norm_stats_file = "./models/ridge_biw/mean_std_for_normalization_with_weather.pkl"
    biw_clf_file = "./models/ridge_biw/RidgeRegModels_SGD_weather.pkl"

    # Sounding/ RAP 130.
    ground_truth_wind = WindSource.rap_130

    # radar_t_sounding = {'KHTX': 'BNA', 'KTLX': 'LMN', 'KOHX': 'BNA', 'KENX': 'ALB'}
    radar_t_sounding = RadarXSoundingDistance(nexrad_table=None, sounding_table=None, output_folder="./radar_data")
    sounding_log_dir = "./sounding_logs"

    rap_folder = r"./atmospheric_model_data/rap_130_20180501_20180531"

    # VAD
    vad_jobs = [VADMask.birds, VADMask.insects, VADMask.weather, VADMask.biological]
    min_req_nsamples_vad = 720
    delta_time_hr = 2 * 60 / 60
    time_window = {'noon': (12 - delta_time_hr, 12 + delta_time_hr),
                   'midnight': (24 - delta_time_hr, (24 + delta_time_hr) % 24)}
    vad_sounding_output_dir = "./vad_sounding_comparison_logs"

    e2e_analysis_log_dir = './batch_analysis_logs'

    # GetEchoDistributionBatch(batch_folder, radar_folder, level3_folder, start_day, stop_day, date_pattern, max_range,
    #                          clf_file, output_log_dir, figure_dir, save_ppi_plots, force_output_logging,
    #                          norm_stats_file=norm_stats_file,
    #                          correct_hca_weather=correct_hca_weather, biw_clf_file=biw_clf_file,
    #                          biw_norm_stats_file=biw_norm_stats_file)

    # AnalyzeWindBatch(batch_folder, radar_folder, level3_folder, start_day, stop_day, date_pattern, max_range,
    #                  max_height_VAD, time_window, clf_file, radar_t_sounding, station_infos, sounding_log_dir,
    #                  norm_stats_file, vad_jobs, figure_dir, vad_sounding_output_dir,
    #                  ground_truth_source=ground_truth_wind, rap_folder=rap_folder,
    #                  correct_hca_weather=correct_hca_weather, biw_clf_file=biw_clf_file,
    #                  biw_norm_stats_file=biw_norm_stats_file)

    # Experiment parameters space
    experiment_name = ""
    use_vad_weights_grid = [False] #[False, True]
    clf_purity_threshold_grid = [0.5] #[0.5, 0.4, 0.3, 0.2, 0.1]

    for use_vad_weights in use_vad_weights_grid:
        for clf_purity_threshold in clf_purity_threshold_grid:
            print("use_vad_weights: ", use_vad_weights)
            print("clf_purity_threshold: ", clf_purity_threshold)

            E2EWindAnalysis(batch_folder=batch_folder, radar_folder=radar_folder, level3_folder=level3_folder,
                            start_day=start_day, stop_day=stop_day, date_pattern=date_pattern, max_range=max_range,
                            max_height_VAD=max_height_VAD, time_window=time_window, clf_file=clf_file,
                            radar_t_sounding=radar_t_sounding, sounding_log_dir=sounding_log_dir,
                            norm_stats_file=norm_stats_file, vad_jobs=vad_jobs, figure_dir=figure_dir,
                            vad_sounding_dir=vad_sounding_output_dir, echo_count_log_dir=output_log_dir,
                            save_ppi_plots=save_ppi_plots, force_output_logging=force_output_logging,
                            ground_truth_source=ground_truth_wind, rap_folder=rap_folder,
                            correct_hca_weather=correct_hca_weather,
                            biw_norm_stats_file=biw_norm_stats_file, biw_clf_file=biw_clf_file,
                            log_dir=e2e_analysis_log_dir,
                            experiment_name=experiment_name, allowed_el_hca=allowed_el_hca,
                            use_vad_weights=use_vad_weights,
                            clf_purity_threshold=clf_purity_threshold, min_required_nsamples=min_req_nsamples_vad,
                            height_binsize=0.04)

    return


Main()
