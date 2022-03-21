import os
import matplotlib.pyplot as plt
from SoundingDataUtils import *
from RapUtils import *

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 11}
plt.rc('font', **font)


# TODO
# logging raw sounding data

def Main():
    station_id = 72327

    rap_folder = './atmospheric_model_data/rap_130_20180501_20180515'
    rap_files = os.listdir(rap_folder)

    log_dir = './atmospheric_model_data/UV_wind_logs'
    log_file_base = '{}_windcomponents_lat_{}_lon_{}.pkl'

    figure_file_base = "rap_{}_sounding_{}_lat_{}_lon_{}.png"
    figure_folder = './figures/RapVsSounding'
    if not os.path.isdir(figure_folder):
        os.makedirs(figure_folder)

    soundings_loc_table = GetSoundingsTable()
    ss_info = GetSoundingStations()
    current_station = ss_info[station_id]
    station_name = current_station[1]
    city_state = ', '.join(current_station[2:])
    station_lat = soundings_loc_table[station_id]['latitude']
    station_lon = soundings_loc_table[station_id]['longitude']

    max_height_plot_m = 1000
    save_figure = True

    title_base = "{}/{}/{}, {}:{} UTC.\nSounding: {}. {}.\nRAP: {}, {}km from sounding."
    latlon_base = "Lat {}{}, Lon {}{}"
    latlon_sounding = latlon_base.format(station_lat, '$^{\circ}$', station_lon, '$^{\circ}$')

    for rap_file in rap_files:

        print(rap_file)

        year = rap_file[11:15]
        month = rap_file[15:17]
        day = rap_file[17:19]
        hour = rap_file[19:21]
        hhmm = ''.join([hour, '00'])

        # Rap wind profile
        rap_wind_profile, rap_latlon = GetRapWindProfile(station_lat, station_lon, rap_folder, rap_file, log_dir,
                                                      log_file_base, show_fig=False, force_update=False,
                                                      save_wind_profile=True)
        rap_sounding_distance_km = 1e-3 * GetHaverSineDistance(station_lat, station_lon, rap_latlon['latitude'],
                                                               rap_latlon['longitude'])

        latlon_rap = latlon_base.format(round(rap_latlon['latitude'], 2), '$^{\circ}$',
                                        round(rap_latlon['longitude'], 2), '$^{\circ}$')

        # Sounding wind profile
        year, month, sounding_ddhh = GetSoundingDateTime(year=year, month=month, day=day, hhmm=hhmm)
        sounding_url = GetSoundingUrl(station_id=station_id, year=year, month=month, sounding_ddhh=sounding_ddhh)
        sounding_data_df, header_units, sounding_metadata = GetSoundingData(sounding_url)

        if sounding_data_df is None:
            continue

        sounding_data_df['SKNT'] = sounding_data_df['SKNT'] * KNOT_T_MPS
        sounding_data_df.rename(columns={"SKNT": "SMPS"}, inplace=True)
        sounding_data_df['DRCT'] = (sounding_data_df['DRCT'] + 180) % 360
        sounding_data_df['windU'] = sounding_data_df['SMPS'] * np.sin(sounding_data_df['DRCT'] * np.pi / 180)
        sounding_data_df['windV'] = sounding_data_df['SMPS'] * np.cos(sounding_data_df['DRCT'] * np.pi / 180)

        max_height_m = 1.2 * min(max(sounding_data_df['HGHT']), max(
            rap_wind_profile['height_gpm'])) if max_height_plot_m is None else max_height_plot_m
        height_idx_rap = rap_wind_profile['height_gpm'] <= max_height_m
        height_idx_sounding = sounding_data_df['HGHT'] <= max_height_m

        title_str = title_base.format(year, month, day, hour, '00', city_state, latlon_sounding, latlon_rap,
                                      round(rap_sounding_distance_km, 2))
        outname = figure_file_base.format(rap_file[11:21], station_name, station_lat, station_lon)

        plt.figure()

        # Rap wind
        plt.scatter(x=rap_wind_profile['U_ms'][height_idx_rap], y=rap_wind_profile['height_gpm'][height_idx_rap],
                    marker='^', color="cornflowerblue", alpha=0.4)
        plt.plot(rap_wind_profile['U_ms'][height_idx_rap], rap_wind_profile['height_gpm'][height_idx_rap],
                 linestyle='dashed', color="cornflowerblue", alpha=0.8,
                 label='rap U')
        plt.scatter(x=rap_wind_profile['V_ms'][height_idx_rap], y=rap_wind_profile['height_gpm'][height_idx_rap],
                    marker='^', color="lime", alpha=0.4)
        plt.plot(rap_wind_profile['V_ms'][height_idx_rap], rap_wind_profile['height_gpm'][height_idx_rap],
                 linestyle='dashed', color="lime", alpha=0.8,
                 label='rap V')

        # Sounding wind.
        plt.scatter(x=sounding_data_df['windU'][height_idx_sounding], y=sounding_data_df['HGHT'][height_idx_sounding],
                    marker='o', color='blue', alpha=0.4)
        plt.plot(sounding_data_df['windU'][height_idx_sounding], sounding_data_df['HGHT'][height_idx_sounding],
                 linestyle='dashed', color='blue', alpha=0.8,
                 label='sound U')
        plt.scatter(x=sounding_data_df['windV'][height_idx_sounding], y=sounding_data_df['HGHT'][height_idx_sounding],
                    marker='o', color='red', alpha=0.4)
        plt.plot(sounding_data_df['windV'][height_idx_sounding], sounding_data_df['HGHT'][height_idx_sounding],
                 linestyle='dashed', color='red', alpha=0.8,
                 label='sound V')

        plt.xlim(-30, 45)
        plt.ylim(0, max_height_m)
        plt.ylim()
        plt.grid(True)
        plt.xlabel("Wind components [mps]")
        plt.ylabel("Geopotential height [m]")
        plt.title(title_str)
        plt.legend()
        plt.tight_layout()
        # plt.show()

        if save_figure:
            plt.savefig(os.path.join(figure_folder, outname))
        plt.close()

    return


Main()
