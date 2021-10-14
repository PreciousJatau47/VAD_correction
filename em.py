from HaverSineDistance import GetHaverSineDistance

def Main():
    lat1, lon1 = 35.33, -97.2775
    lat2, lon2 = 59, 112.58 # LMN
    d = GetHaverSineDistance(lat1,lon1, lat2, lon2)
    print(d)

Main()
