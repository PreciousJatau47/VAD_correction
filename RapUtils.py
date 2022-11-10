import pygrib
import numpy as np
import pandas as pd
import sys
import os
import pickle
from HaverSineDistance import GetHaverSineDistance
import matplotlib.pyplot as plt


# TODO
# consider getting r_idx, c_idx once, and reusing.

def GetLatLonIdx(lat_key, lon_key, delta, lat_grid, lon_grid):
    assert lat_grid.shape == lon_grid.shape
    assert lat_key >= np.min(lat_grid) and lat_key <= np.max(lat_grid)
    assert lon_key >= np.min(lon_grid) and lon_key <= np.max(lon_grid)

    search_box = [lat_key - delta, lat_key + delta, lon_key - delta, lon_key + delta]
    found = np.logical_and(lat_grid > search_box[0], lat_grid < search_box[1])
    found = np.logical_and(found, lon_grid > search_box[2])
    found = np.logical_and(found, lon_grid < search_box[3])

    rows_found, cols_found = np.where(found)

    best_row, best_col, best_distance = None, None, sys.maxsize

    for i in range(len(rows_found)):
        curr_distance = GetHaverSineDistance(lat_key, lon_key, lat_grid[rows_found[i]][cols_found[i]],
                                             lon_grid[rows_found[i]][cols_found[i]])

        if curr_distance < best_distance:
            best_distance = curr_distance
            best_row = rows_found[i]
            best_col = cols_found[i]
            print('best distance: ', curr_distance, 'm')
    print()

    return best_row, best_col


def GetVariableProfile(row_idx, col_idx, msg, name, pressure_levels):
    if row_idx is None:
        return None
    tmp_msg = msg[1]
    lat, lon = tmp_msg.latlons()
    print("Getting {} profile at lat {} degrees, lon {} degrees.".format(name, lat[row_idx][col_idx],
                                                                         lon[row_idx][col_idx]))
    var_profile = []

    for level in pressure_levels:
        curr_msg = msg.select(name=name, typeOfLevel='isobaricInhPa', level=level)
        print(curr_msg)
        var_profile.append(curr_msg[0].values[row_idx][col_idx])
    print()

    return var_profile


"Obtains and returns the wind profile for a chosen lat,lon location from rap 130. Wind profile contains U [mps], " \
"V[mps] and HGHT[gpm]"


def GetRapWindProfile(in_lat, in_lon, rap_dir, rap_file, log_dir, log_base, show_fig=False, force_update=False,
                      save_wind_profile=False):
    rap_full_path = os.path.join(rap_dir, rap_file)
    log_file = log_base.format(rap_file[:-7], round(in_lat, 4), round(in_lon, 4))
    log_file_path = os.path.join(log_dir, log_file)

    if not force_update and os.path.isfile(log_file_path):
        with open(log_file_path, 'rb') as p_in:
            wind_profile, grid_loc = pickle.load(p_in)
        p_in.close()
        return wind_profile, grid_loc

    gr = pygrib.open(rap_full_path)
    levels = np.arange(100, 1000 + 25, 25)  # np.arange(100, 175, 25)
    gpt_height_level = gr.select(name='Geopotential Height', typeOfLevel='isobaricInhPa', level=levels[0])
    lat, lon = gpt_height_level[0].latlons()

    r_idx, c_idx = GetLatLonIdx(in_lat, in_lon, 0.5, lat, lon)
    gpt_height_profile = GetVariableProfile(r_idx, c_idx, gr, 'Geopotential Height', levels)
    U_wind = GetVariableProfile(r_idx, c_idx, gr, 'U component of wind', levels)
    V_wind = GetVariableProfile(r_idx, c_idx, gr, 'V component of wind', levels)
    wind_profile = pd.DataFrame({"windU": U_wind, "windV": V_wind, "HGHT": gpt_height_profile})
    wind_profile['DRCT'] = np.nan
    wind_profile['SMPS'] = np.nan
    grid_loc = {'latitude': lat[r_idx][c_idx], 'longitude': lon[r_idx][c_idx]}

    height_idx = wind_profile['HGHT'] < sys.maxsize

    if show_fig:
        plt.scatter(wind_profile["windU"][height_idx], wind_profile['HGHT'][height_idx], color='blue', alpha=0.2)
        plt.plot(wind_profile["windU"][height_idx], wind_profile['HGHT'][height_idx], color='blue')
        plt.scatter(wind_profile["windV"][height_idx], wind_profile['HGHT'][height_idx], color='red', alpha=0.2)
        plt.plot(wind_profile["windV"][height_idx], wind_profile['HGHT'][height_idx], color='red')
        plt.xlabel("Wind component [m/s]")
        plt.ylabel("Geopotential height [gpm]")
        plt.title(rap_file)
        plt.show()

    if force_update or save_wind_profile:
        with open(log_file_path, 'wb') as p_out:
            pickle.dump((wind_profile, grid_loc), p_out)
        p_out.close()

    return wind_profile, grid_loc


def LogRapWindProfileBatch(in_lat, in_lon, rap_dir, log_dir, log_base, show_fig=False, force_update=False,
                           save_wind_profile=False):
    rap_files = os.listdir(rap_dir)
    for rap_file in rap_files:
        gt_wind_df, gt_wind_location = GetRapWindProfile(in_lat, in_lon, rap_dir, rap_file, log_dir, log_base,
                                                         show_fig=show_fig, force_update=force_update,
                                                         save_wind_profile=save_wind_profile)
    return


def GetRapWindProfileRelativeToRadar(in_lat, in_lon, radar_location, rap_dir, rap_file, log_dir, log_base,
                                     show_fig=False, force_update=False,
                                     save_wind_profile=False):
    radar_to_rap = GetHaverSineDistance(radar_location["latitude"], radar_location["longitude"], in_lat, in_lon)
    print("Distance between radar and sounding station is {} km.".format(round(radar_to_rap / 1000, 2)))

    gt_wind_df, gt_wind_location = GetRapWindProfile(in_lat, in_lon, rap_dir, rap_file, log_dir, log_base,
                                                     show_fig=False, force_update=False,
                                                     save_wind_profile=True)

    gt_wind_df['HGHT'] = gt_wind_df['HGHT'] - radar_location['height']

    return gt_wind_df, gt_wind_location


def Main():
    rap_data_dir = './atmospheric_model_data/rap_130_20180501_20180531/'
    log_dir = './atmospheric_model_data/UV_wind_logs'
    log_file_base = '{}_windcomponents_lat_{}_lon_{}.pkl'

    # radar_location_kohx = {'latitude': 36.247222900390625, 'longitude': -86.5625, 'height': 205.0}
    radar_location_khpx = {'latitude': 36.73697280883789, 'longitude': -87.28558349609375, 'height': 186.0}
    radar_location_kohx = {'latitude': 34.93055725097656, 'longitude': -86.08361053466797, 'height': 566.0}
    radar_location_klvx = {'latitude':37.975276947021484, 'longitude': -85.94388580322266, 'height': 253.0}
    radar_location = radar_location_klvx

    # BNA sounding station
    station_lat = 36.25
    station_lon = -86.57

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    LogRapWindProfileBatch(in_lat=radar_location['latitude'], in_lon=radar_location['longitude'],
                           rap_dir=rap_data_dir, log_dir=log_dir, log_base=log_file_base, show_fig=False,
                           force_update=False, save_wind_profile=True)

    return


Main()
