import pandas as pd
import os
import pickle
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from HaverSineDistance import GetHaverSineDistance
import Station

KNOT_T_MPS = 0.514444

DAYS_PER_MONTH = {'01': 31, '02': 28, '03': 31, '04': 30, '05': 31, '06': 30, '07': 31, '08': 31, '09': 30, '10': 31,
                  '11': 30, '12': 31}

# TODO Consider leap years.
def GetSoundingDateTime(year='2018', month='05', day='01', hhmm='00'):
    day = int(day)
    hhmm_int = round(int(hhmm[0:2]) + int(hhmm[2:4]) / 60)

    # Find closest sounding
    sounding_hh = (hhmm_int // 12 + int((hhmm_int % 12) > 6)) * 12

    if sounding_hh == 24:
        day = day + 1
        # TODO need to check for leap years.
        if day > DAYS_PER_MONTH[month]:
            day = day % DAYS_PER_MONTH[month]
            month = int(month) + 1
            month = str(month) if month >= 10 else '0' + str(month)
        sounding_hh = 0
    day = ''.join(['0', str(day)]) if day < 10 else str(day)
    sounding_hh = ''.join(['0', str(sounding_hh)]) if sounding_hh < 10 else str(sounding_hh)
    sounding_ddhh = ''.join([day, sounding_hh])

    return year, month, sounding_ddhh

# TODO Consider leap years.
def GetSoundingDateTimeFromRadarFile(radar_file):
    year = radar_file[4:8]
    month = radar_file[8:10]
    day = radar_file[10:12]
    hhmm = radar_file[13:17]

    day = int(day)
    hhmm_int = round(int(hhmm[0:2]) + int(hhmm[2:4]) / 60)

    # Find closest sounding
    sounding_hh = (hhmm_int // 12 + int((hhmm_int % 12) > 6)) * 12

    if sounding_hh == 24:
        day = day + 1
        # TODO need to check for leap years.
        if day > DAYS_PER_MONTH[month]:
            day = day % DAYS_PER_MONTH[month]
            month = int(month) + 1
            month = str(month) if month >= 10 else '0' + str(month)
        sounding_hh = 0
    day = ''.join(['0', str(day)]) if day < 10 else str(day)
    sounding_hh = ''.join(['0', str(sounding_hh)]) if sounding_hh < 10 else str(sounding_hh)
    sounding_ddhh = ''.join([day, sounding_hh])

    return year, month, sounding_ddhh


def GetSoundingUrlFromRadarFile(sounding_url_base, radar_file, station_id):
    year, month, sounding_ddhh = GetSoundingDateTimeFromRadarFile(radar_file)
    sounding_url = sounding_url_base.format(year, month, sounding_ddhh, sounding_ddhh, station_id)
    return sounding_url, ''.join([year, month, sounding_ddhh])


def GetSoundingData(url):
    # Import table from url.
    response = urllib.request.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, features="lxml")
    pre = soup.find_all('pre')

    if not pre:
        return None, None, None

    # Load sounding data.
    sounding_data = str(pre[0])
    sounding_data_lines = sounding_data.splitlines()

    # Read header information.
    header = sounding_data_lines[2].split()
    numFields = len(header)
    header_units = sounding_data_lines[3].split()

    # Read measurements and save to data frame.
    sounding_data_list = []
    count = 5
    while count < len(sounding_data_lines):
        fields = sounding_data_lines[count].split()
        if len(fields) == numFields:
            sounding_data_list.append(fields)
        count += 1
    sounding_data_df = pd.DataFrame(sounding_data_list, columns=header, dtype=np.float32)

    # Read metadata into dictionary.
    meta_data = str(pre[1]).splitlines()
    meta_data_dic = {}
    for meta_data_line in meta_data:
        meta_data_line = meta_data_line.split(':')
        if meta_data_line[0] != '</pre>' and meta_data_line[0] != '<pre>':
            meta_data_dic[meta_data_line[0].lstrip()] = meta_data_line[1].lstrip()

    return sounding_data_df, header_units, meta_data_dic


def GetSoundingWind(sounding_url_base, radar_data_file, radar_location, station_id, sounding_log_dir, showDebugPlot,
                    log_sounding_data,
                    force_website_download, return_all_fields=False):
    """
    :param sounding_url_base:
    :param radar_data_file:
    :param radar_location:
    :param station_id:
    :param showDebugPlot:
    :return: data frame containing height, temperature, wind speed, wind direction, U and V components.
    """
    # Read sounding data.
    sounding_url, timestamp = GetSoundingUrlFromRadarFile(sounding_url_base, radar_data_file, station_id)
    print(sounding_url)

    # Load sounding data if local copy exists.
    log_filename = "".join([station_id, '_', timestamp, '.pkl'])
    sounding_log_path = os.path.join(sounding_log_dir, log_filename)
    if not force_website_download and os.path.exists(sounding_log_path):
        print("Loading local copy of sounding data.... ")
        pin = open(sounding_log_path, 'rb')
        sounding_data_logged = pickle.load(pin)
        pin.close()

        if return_all_fields:
            return sounding_data_logged
        return sounding_data_logged[0][['HGHT', 'TEMP', 'DRCT', 'SMPS', 'windU', 'windV']], sounding_data_logged[1], \
               sounding_data_logged[2]

    sounding_data_df, header_units, meta_data = GetSoundingData(sounding_url)

    if sounding_data_df is None:
        return None, None, sounding_url

    # Location of sounding station in degrees and meters.
    location_sounding = {"latitude": float(meta_data['Station latitude']),
                         "longitude": float(meta_data['Station longitude']),
                         "height": float(meta_data['Station elevation'])}

    # Distance between radar and sounding station.
    radar_to_sounding = GetHaverSineDistance(radar_location["latitude"], radar_location["longitude"],
                                             location_sounding["latitude"],
                                             location_sounding["longitude"])
    print("Distance between radar and sounding station is {} km.".format(round(radar_to_sounding / 1000, 2)))

    #  Convert sounding data to radar's frame
    # TODO check if radar and sounding height are measured thesame way.
    sounding_data_df['HGHT'] = sounding_data_df['HGHT'] - radar_location["height"]

    # convert wind speed from knots to mps.
    sounding_data_df['SKNT'] = sounding_data_df['SKNT'] * KNOT_T_MPS
    sounding_data_df.rename(columns={"SKNT": "SMPS"}, inplace=True)

    # Set wind direction to where the wind is blowing.
    sounding_data_df['DRCT'] = (sounding_data_df['DRCT'] + 180) % 360

    # Get U and V components of the wind.
    sounding_data_df['windU'] = sounding_data_df['SMPS'] * np.sin(sounding_data_df['DRCT'] * np.pi / 180)
    sounding_data_df['windV'] = sounding_data_df['SMPS'] * np.cos(sounding_data_df['DRCT'] * np.pi / 180)
    print(sounding_data_df.columns)

    # Save sounding data.
    if log_sounding_data:
        out_data = (
            sounding_data_df, location_sounding, sounding_url)
        with open(sounding_log_path, 'wb') as sd:
            pickle.dump(out_data, sd)
        sd.close()

    if showDebugPlot:
        plt.figure()
        plt.plot(sounding_data_df['windU'], sounding_data_df['HGHT'], color='blue', label="windU")
        plt.plot(sounding_data_df['windV'], sounding_data_df['HGHT'], color='red', label="windV")
        plt.xlabel("wind component [m/s]")
        plt.ylabel("height [m]")
        plt.title("Sounding.")
        plt.legend()
        # plt.show()

    if return_all_fields:
        return sounding_data_df, location_sounding, sounding_url
    return sounding_data_df[['HGHT', 'TEMP', 'DRCT', 'SMPS', 'windU', 'windV']], location_sounding, sounding_url


def GetSoundingStations(sounding_stations_path=None):
    if not sounding_stations_path:
        sounding_stations_path = 'sounding_logs/wyoming_sounding_stations.pkl'

    with open(sounding_stations_path, 'rb') as p_in:
        sounding_stations = pickle.load(p_in)
    p_in.close()

    return sounding_stations


# TODO
# Add capability of updating sounding table with missed soundings.
def GetSoundingsTable(sounding_stations=None, year='2018', month='07', ddhh="0212",
                      sounding_url_base="http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR={}&MONTH={}&FROM={}&TO={}&STNM={}",
                      to_save=False, force_download=False):
    output_path = './sounding_logs/wyoming_sounding_locations.pkl'
    if not force_download and os.path.isfile(output_path):
        with open(output_path, 'rb') as p_in:
            soundings_location_table = pickle.load(p_in)
        p_in.close()
        return soundings_location_table

    soundings_location_table = {}
    unavailable_soundings = []
    for station_id in sounding_stations:
        print("Collecting coordinates for ", sounding_stations[station_id][1], ", ", sounding_stations[station_id][2],
              ", ", sounding_stations[station_id][3])
        sounding_url = sounding_url_base.format(year, month, ddhh, ddhh, station_id)
        sounding_data_df, header_units, meta_data = GetSoundingData(sounding_url)
        if sounding_data_df is None:
            print("Sounding unavailable")
            unavailable_soundings.append(station_id)
            continue

        soundings_location_table[station_id] = {'latitude': float(meta_data['Station latitude']),
                                                'longitude': float(meta_data['Station longitude']),
                                                'elevation': float(meta_data['Station elevation'])}
    print(unavailable_soundings)

    if to_save:
        with open(output_path, 'wb') as p_out:
            pickle.dump(soundings_location_table, p_out)
        p_out.close()

    return soundings_location_table

# TODO
# Delete Main()
# def Main():
#     sounding_stations = GetSoundingStations()
#
#     year = "2018"
#     month = "07"
#     ddhh = "0212"
#     sounding_url_base = "http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR={}&MONTH={}&FROM={}&TO={}&STNM={}"
#     soundings_table = GetSoundingsTable(sounding_stations, year, month, ddhh, sounding_url_base, to_save=True,
#                                         force_download=True)
#
#     return
#
#
# Main()
