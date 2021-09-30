import numpy as np
import math

def GetHaverSineDistance(lat1, lon1, lat2, lon2):
    """
    :param lat1: latitude of point 1 in degrees north.
    :param lon1: longitude of point 1 in degrees east.
    :param lat2:
    :param lon2:
    :return: distance between point 1 and point 2 in meters.
    """
    EARTH_RADIUS = 6371e3

    # convert to radians
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    delta_lat = lat1 - lat2
    delta_lon = lon1 - lon2
    a = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + \
        math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) * math.sin(delta_lon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = EARTH_RADIUS * c

    return d
