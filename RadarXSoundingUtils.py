from NexradUtils import *
from SoundingDataUtils import *


def RadarToSoundingsDistance(radar_id, nexrad_table, sounding_location_table):
    if radar_id not in nexrad_table:
        print(radar_id, " not found in nexrad table. Returning ...")
        return None
    radar_location = nexrad_table[radar_id]
    radar_t_soundings = []

    for sounding_id in sounding_location_table:
        # print(sounding_id)
        radar_t_soundings.append(
            (sounding_id, GetHaverSineDistance(radar_location["latitude"], radar_location["longitude"],
                                               sounding_location_table[sounding_id]["latitude"],
                                               sounding_location_table[sounding_id]["longitude"])))

    # sort by distance
    radar_t_soundings = sorted(radar_t_soundings, key=lambda x: x[1])
    return radar_t_soundings


def RadarXSoundingDistance(nexrad_table, sounding_table, output_folder, to_save=False, force_update=False):
    output_path = os.path.join(output_folder, 'RadarsXSoundingsDistances.pkl')

    if not force_update and os.path.isfile(output_path):
        with open(output_path, 'rb') as p_in:
            radars_x_soundings = pickle.load(p_in)
        p_in.close()
        return radars_x_soundings

    radars_x_soundings = {}
    for radar_id in nexrad_table:
        radars_x_soundings[radar_id] = RadarToSoundingsDistance(radar_id, nexrad_table, sounding_table)

    if to_save:
        with open(output_path, 'wb') as p_in:
            pickle.dump(radars_x_soundings, p_in)
        p_in.close()

    return radars_x_soundings


def GetSoundingsWithinRadius(radar_id, radius_km, radars_x_soundings):
    radius_m = radius_km * 1e3
    if radar_id not in radars_x_soundings:
        print(radar_id, " not found in radar x soundings distance table.")
        return []

    soundings = radars_x_soundings[radar_id]
    i = 0
    while i < len(soundings) and soundings[i][1] < radius_m:
        i += 1
    return soundings[:i]


def GetRadarsClosestSoundingsWithinRadius(radar_id, radius_km, radar_x_radar, radar_x_sounding):
    print("Getting radars within {}km of {}".format(radius_km, radar_id))
    radars_in_radius = GetRadarsWithinRadius(radar_id, radius_km, radar_x_radar)
    radars_closest_sounding = {}
    print("Radar\tDistance (km)\tClosest Sounding\tDistance (km)")
    print("{}\t{}\t\t\t{}\t\t\t\t{}".format(radar_id, '0.00', radar_x_sounding[radar_id][0][0],
                                            round(radar_x_sounding[radar_id][0][1] / 1000, 2)))

    for radar in radars_in_radius:
        radars_closest_sounding[radar[0]] = radar_x_sounding[radar[0]][0]
        print("{}\t{}\t\t\t{}\t\t\t\t{}".format(radar[0], round(radar[1] / 1000, 2), radar_x_sounding[radar[0]][0][0],
                                                round(radar_x_sounding[radar[0]][0][1] / 1000, 2)))
    print()

    return radars_closest_sounding


def Main():
    radar_base_folder = "./radar_data"
    radar_folder_table = "radar_table_files"
    radar_table = GetNexradTable(radar_base_folder=radar_base_folder, radar_folder_table=radar_folder_table,
                                 save_table=True, force_collection=False)
    radar_x_radar = RadarXRadarDistance(radar_table, radar_base_folder)

    sounding_location_table = GetSoundingsTable()
    radar_id = 'KHPX'
    output_folder = "./radar_data"

    # radar_t_sounding = RadarToSoundingsDistance(radar_id, radar_table, sounding_location_table)
    radar_x_sounding = RadarXSoundingDistance(radar_table, sounding_location_table, output_folder, to_save=True,
                                              force_update=True)
    closest_soundings = GetSoundingsWithinRadius('KTLX', 100, radar_x_sounding)

    print()
    radars_closest_sounding = GetRadarsClosestSoundingsWithinRadius('KOHX', 300, radar_x_radar,
                                                                    radar_x_sounding)

    return


Main()
