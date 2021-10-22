import pandas as pd
import os
import pickle
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

from HaverSineDistance import GetHaverSineDistance

KNOT_T_MPS = 0.514444


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
        # TODO need to check if day is valid for the current month.
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


def GetSoundingWind(sounding_url_base, radar_data_file, radar_location, station_id, sounding_log_dir, showDebugPlot, log_sounding_data,
                    force_website_download):
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
    sounding_log_path = os.path.join(sounding_log_dir, timestamp + ".pkl")
    if not force_website_download and os.path.exists(sounding_log_path):
        print("Loading local copy of sounding data.... ")
        pin = open(sounding_log_path, 'rb')
        sounding_data_logged = pickle.load(pin)
        pin.close()
        return sounding_data_logged

    sounding_data_df, header_units, meta_data = GetSoundingData(sounding_url)

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
        sounding_data_df[['HGHT', 'TEMP', 'DRCT', 'SMPS', 'windU', 'windV']], location_sounding, sounding_url)
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

    return sounding_data_df[['HGHT', 'TEMP', 'DRCT', 'SMPS', 'windU', 'windV']], location_sounding, sounding_url
