from NexradUtils import *
from SoundingDataUtils import *
from RapUtils import *
from RadarXSoundingUtils import RadarXSoundingDistance
import calendar
from AnalyzeWind import InterpolateSoundingWind

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def RapXSoundingComparison():
    level3_folder = "./level3_data"
    radar_folder = "./radar_data"
    batch_folder = "KOHX_20180501_20180531"

    output_dir_base = "./rap_X_sounding_comparison_logs"
    output_dir = os.path.join(output_dir_base, batch_folder)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    start_day = 1
    stop_day = 31

    date_pattern = "*{}{}".format(batch_folder[:4], batch_folder[5:11])
    date_pattern = "".join([date_pattern, '{}*_V06.*'])

    delta_time_hr = 2 * 60 / 60
    time_window = {'noon': (12 - delta_time_hr, 12 + delta_time_hr),
                   'midnight': (24 - delta_time_hr, (24 + delta_time_hr) % 24)}

    batch_folder_path_l3 = os.path.join(level3_folder, batch_folder)
    batch_folder_path_radar = os.path.join(radar_folder, batch_folder)
    radar_scans_day = GetFileListRadar(batch_folder_path_radar, start_day=start_day, stop_day=stop_day,
                                       date_pattern=date_pattern)
    # l3_files_dic = GetFileListL3(batch_folder_path_l3)

    days = list(radar_scans_day.keys())
    days.sort()

    sounding_url_base = "http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR={}&MONTH={}&FROM={}&TO={}&STNM={}"
    sounding_log_dir = "./sounding_logs"
    radar_t_sounding = RadarXSoundingDistance(nexrad_table=None, sounding_table=None, output_folder="./radar_data")

    RAP_FILE_BASE = "rapanl_130_{}{}{}{}.g2.tar"
    rap_folder = r"./atmospheric_model_data"
    log_dir_rap = './atmospheric_model_data/UV_wind_logs'
    log_file_base_rap = '{}_windcomponents_lat_{}_lon_{}.pkl'

    max_height_VAD = 1500
    max_height_bin = 1.1 * max_height_VAD
    height_binsize_m = 0.04 * 1000
    vad_heights = np.arange(0, max_height_bin, height_binsize_m)
    height_grid_interp = pd.Series(vad_heights)

    # Interpolate wind components.
    max_height = 1.1 * max_height_VAD  # meters.
    max_height_diff = 300  # meters.

    for idx_days in range(len(days)):
        curr_day = days[idx_days]
        print("Processing ", curr_day, " ........")
        radar_scans = radar_scans_day[curr_day]  # ~200 scans

        for radar_subpath in radar_scans:
            print("Processing ", radar_subpath, " ........")
            time_hour = GetTimeHourUTC(radar_subpath)
            is_near_sounding = (time_hour > time_window['noon'][0] and time_hour < time_window['noon'][1]) or (
                    time_hour > time_window['midnight'][0] or time_hour < time_window['midnight'][1])

            target_folder = os.path.split(radar_subpath)
            target_file = target_folder[1]
            radar_data_file = target_file
            radar_data_file_no_ext = os.path.splitext(radar_data_file)[0]
            output_path = os.path.join(output_dir, radar_data_file_no_ext)

            try:
                print("Opening ", os.path.join(batch_folder_path_radar, radar_subpath))
                radar = pyart.io.read_nexrad_archive(os.path.join(batch_folder_path_radar, radar_subpath))
            except Exception as e:
                print("An error occured in reading radar file: ", str(e), "\nSkipping to next iteration.")
                continue

            location_radar = {"latitude": radar.latitude['data'][0],
                              "longitude": radar.longitude['data'][0],
                              "height": radar.altitude['data'][0]}
            radar_name, year, month, day, hh, mm, ss = read_info_from_radar_name(radar_data_file)
            station_id = str(radar_t_sounding[radar_name][0][0])

            if is_near_sounding:
                year_sounding, month_sounding, gt_wind_ddhh = GetSoundingDateTimeFromRadarFile(radar_data_file)
                sounding_wind_df, sounding_wind_location, sounding_url = GetSoundingWind(sounding_url_base,
                                                                                         radar_data_file,
                                                                                         location_radar,
                                                                                         station_id, sounding_log_dir,
                                                                                         showDebugPlot=False,
                                                                                         log_sounding_data=True,
                                                                                         force_website_download=False)
                if sounding_wind_df is None:
                    continue

                # Remove radar height referencing
                sounding_wind_df['HGHT'] = sounding_wind_df['HGHT'] + location_radar["height"]

                sounding_wind_df_interp = InterpolateSoundingWind(sounding_df=sounding_wind_df,
                                                                  height_grid_interp=height_grid_interp,
                                                                  max_height_diff=max_height_diff,
                                                                  max_height=max_height)

                rap_file = RAP_FILE_BASE.format(year, month, day, hh)
                _, month_end = calendar.monthrange(year=int(year), month=int(month))
                rap_folder_case = os.path.join(rap_folder,
                                               "rap_130_{}{}{}_{}{}{}".format(year, month, '01', year, month,
                                                                              str(month_end)))

                rap_wind_df, rap_wind_location = GetRapWindProfile(in_lat=sounding_wind_location['latitude'],
                                                                   in_lon=sounding_wind_location['longitude'],
                                                                   rap_dir=rap_folder_case, rap_file=rap_file,
                                                                   log_dir=log_dir_rap, log_base=log_file_base_rap,
                                                                   show_fig=False, force_update=False,
                                                                   save_wind_profile=True)

                spd_rap, dirn_rap = Cartesian2PolarComponentsDf(u=rap_wind_df["windU"], v=rap_wind_df["windV"])
                rap_wind_df['SMPS'] = spd_rap
                rap_wind_df['DRCT'] = dirn_rap

                rap_wind_df_interp = InterpolateSoundingWind(sounding_df=rap_wind_df,
                                                             height_grid_interp=height_grid_interp,
                                                             max_height_diff=max_height_diff,
                                                             max_height=max_height)
                rap_wind_df_interp.rename(columns=dict(zip(['SMPS', 'DRCT'], ['rap_SMPS', 'rap_DRCT'])),
                                          inplace=True)

                distance_sounding_rap = GetHaverSineDistance(sounding_wind_location["latitude"],
                                                             sounding_wind_location["longitude"],
                                                             rap_wind_location["latitude"],
                                                             rap_wind_location["longitude"])
                distance_sounding_rap = round(distance_sounding_rap / 1000, 2)
                print(distance_sounding_rap)

                diff_wind = sounding_wind_df_interp.loc[:, ["HGHT", "SMPS", "DRCT"]]
                other = rap_wind_df_interp.loc[:, ["HGHT", "rap_SMPS", "rap_DRCT"]]
                diff_wind = pd.merge(diff_wind, other, on="HGHT", how="inner")

                fig, ax = plt.subplots(nrows=1, ncols=2)
                ax[0].plot(sounding_wind_df['SMPS'], sounding_wind_df['HGHT'], c='blue')
                ax[0].plot(rap_wind_df['SMPS'], rap_wind_df['HGHT'], c='red')
                # ax[0].set_ylim(0, max_height_bin)
                ax[0].set_title("Speed (mps)")
                ax[1].plot(sounding_wind_df['DRCT'], sounding_wind_df['HGHT'], c='blue')
                ax[1].plot(rap_wind_df['DRCT'], rap_wind_df['HGHT'], c='red')
                # ax[1].set_ylim(0, max_height_bin)
                ax[1].set_title("Direction ($^\circ$)")
                plt.tight_layout()
                plt.show()

                out_data = (diff_wind, sounding_wind_location, rap_wind_location)
                with open(output_path, 'wb') as sd:
                    pickle.dump(out_data, sd)
                sd.close()

        # loop for radar scans
    # loop for days


if __name__ == "__main__":
    RapXSoundingComparison()
