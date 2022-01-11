from SoundingDataUtils import *
from RadarHCAUtils import read_info_from_radar_name
from AnalyzeWind import PrepareAnalyzeWindInputs
import matplotlib.pyplot as plt
import pyart
import parse
from NexradUtils import *

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 11}
plt.rc('font', **font)


def GetRadarScansNearSounding(batch_folder_path_radar, start_day, stop_day, date_pattern):
    radar_scans_day = GetFileListRadar(batch_folder_path_radar, start_day=start_day, stop_day=stop_day,
                                       date_pattern=date_pattern)

    scans_near_sounding = []
    for day_id in radar_scans_day.keys():
        curr_day_scans = radar_scans_day[day_id]
        zero_utc_scan = os.path.split(curr_day_scans[0])[-1]
        scans_near_sounding.append(zero_utc_scan)  # closest to 0 UTC
        twelve_utc_scan = os.path.split(curr_day_scans[len(curr_day_scans) // 2])[-1]
        scans_near_sounding.append(twelve_utc_scan)  # TODO search for closest to 12 UTC sounding

    return scans_near_sounding


def Main():
    radar_t_sounding = {'KHTX': 'BNA', 'KTLX': 'LMN', 'KOHX': 'BNA'}
    station_infos = {'LMN': ('74646', 'Lamont, Oklahoma'), 'BNA': ('72327', 'Nashville, Tennessee')}

    radar_data_folder = "./radar_data"
    hca_data_folder = "./level3_data"
    is_batch = True

    clf_file = "./models/ridge_bi/RidgeRegModels_SGD_1.pkl"

    sounding_url_base = "http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR={}&MONTH={}&FROM={}&TO={}&STNM={}"
    sounding_log_dir = "./sounding_logs"

    figure_dir = "./figures/DebugSoundingData"
    save_fig = True

    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)

    start_day = 16
    stop_day = 31
    batch_folder_base = "KOHX_201805{}_201805{}"

    start_day_str = '0' + str(start_day) if start_day < 10 else str(start_day)
    stop_day_str = '0' + str(stop_day) if stop_day < 10 else str(stop_day)
    batch_folder = batch_folder_base.format(start_day_str, stop_day_str)

    batch_folder_path_radar = os.path.join(radar_data_folder, batch_folder)
    date_pattern = "*KOHX201805{}*_V06.*"
    scans_near_sounding = GetRadarScansNearSounding(batch_folder_path_radar, start_day, stop_day, date_pattern)

    # Read radar data.
    for radar_data_file in scans_near_sounding:
        radar_data_file, _, data_table, radar_obj, hca_vol = PrepareAnalyzeWindInputs(radar_data_file,
                                                                                      batch_folder,
                                                                                      radar_data_folder,
                                                                                      hca_data_folder,
                                                                                      clf_file,
                                                                                      is_batch)

        if radar_obj is None:
            continue

        # radar info
        radar_name, year, month, day, hh, mm, ss = read_info_from_radar_name(radar_data_file)
        radar_data_file_no_ext = os.path.splitext(radar_data_file)[0]

        location_radar = {"latitude": radar_obj.latitude['data'][0],
                          "longitude": radar_obj.longitude['data'][0],
                          "height": radar_obj.altitude['data'][0]}

        station_id = station_infos[radar_t_sounding[radar_name]][0]
        sounding_wind_df, sounding_location, sounding_url = GetSoundingWind(sounding_url_base, radar_data_file,
                                                                            location_radar,
                                                                            station_id, sounding_log_dir,
                                                                            showDebugPlot=False, log_sounding_data=True,
                                                                            force_website_download=True,
                                                                            return_all_fields=True)
        # print(sounding_wind_df.columns)
        if sounding_wind_df is None:
            continue

        sound_year, sound_month, sound_ddhh_start, sound_ddhh_end, sound_station_id = parse.parse(sounding_url_base,
                                                                                                  sounding_url)

        variables_plot = ['DRCT', 'SMPS', 'TEMP', 'PRES', 'DWPT', 'RELH', 'MIXR', 'THTA', 'THTE', 'THTV']
        units_plot = [r'[$^{\circ}$]', '[m/s]', '[C]', '[hPa]', '[C]', '[%]', '[g/kg]', '[K]', '[K]', '[K]']
        title_plot_base = "{}, {}/{}/{}, {}:{}:{} UTC\n" \
                          "{}, {}/{}/{}, {}:00 UTC"
        title_plot = title_plot_base.format(radar_name, month, day, year, hh, mm, ss,
                                            radar_t_sounding[radar_name], sound_month, sound_ddhh_start[:2], sound_year,
                                            sound_ddhh_start[2:])

        nrows = 3
        ncols = 4
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.4 * 2.2, 4.8 * 2.0))
        for plot_idx in range(len(variables_plot)):
            curr_var = variables_plot[plot_idx]
            ax[plot_idx // ncols][plot_idx % ncols].scatter(sounding_wind_df[curr_var], sounding_wind_df['HGHT'],
                                                            alpha=0.5)
            ax[plot_idx // ncols][plot_idx % ncols].plot(sounding_wind_df[curr_var], sounding_wind_df['HGHT'])
            ax[plot_idx // ncols][plot_idx % ncols].set_title(variables_plot[plot_idx])
            ax[plot_idx // ncols][plot_idx % ncols].set_ylim(0, 1000)
            ax[plot_idx // ncols][plot_idx % ncols].set_xlabel(
                " ".join([variables_plot[plot_idx], units_plot[plot_idx]]))
            ax[plot_idx // ncols][plot_idx % ncols].grid(True)
        plt.tight_layout()

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.ylabel('HGHT [m]')
        fig.subplots_adjust(top=0.9)
        plt.suptitle(title_plot)

        if save_fig:
            plt.savefig(os.path.join(figure_dir, radar_data_file_no_ext + '.png'), )

        # plt.show()


Main()
