import sys
from WindUtils import *

WIND_SPD = 8
WIND_DIRN = 20
FLIGHT_SPD_INSECTS = 1.924
FLIGHT_SPD_BIRDS = 6.101
WIND_OFFSET_INSECTS = -8
WIND_OFFSET_BIRDS = 14

def SearchAlphaFlightAndMagMig(alpha_mig_true, spd_flight, spd_wind, alpha_wind):
    u_w, v_w = Polar2CartesianComponentsDf(spd=spd_wind, dirn=alpha_wind)
    min_delta = sys.maxsize
    best_mag_mig = sys.maxsize
    best_alpha_flight = 180
    for alpha_flight in np.arange(0, 360, .5):
        u_f, v_f = Polar2CartesianComponentsDf(spd=spd_flight, dirn=alpha_flight)
        u_mig = u_f + u_w
        v_mig = v_f + v_w
        mag_mig, alpha_mig_pred = Cartesian2PolarComponentsDf(u_mig, v_mig)
        delta_alpha_mig = abs(CalcSmallAngleDirDiff(alpha_mig_true, alpha_mig_pred))
        if delta_alpha_mig < min_delta:
            min_delta = delta_alpha_mig
            best_mag_mig = mag_mig
            best_alpha_flight = alpha_flight
    return best_alpha_flight, best_mag_mig

def Main():
    # Initialize known winds.
    wind_offset_insects = -4
    mig_dirn_insects = WIND_DIRN + wind_offset_insects
    mig_dirn_birds = WIND_DIRN + WIND_OFFSET_BIRDS

    print(mig_dirn_insects)
    print(mig_dirn_birds)

    # Search for missing wind components.
    alpha_flight_insects, mig_spd_insects = SearchAlphaFlightAndMagMig(alpha_mig_true=mig_dirn_insects,
                                                                       spd_flight=FLIGHT_SPD_INSECTS,
                                                                       spd_wind=WIND_SPD, alpha_wind=WIND_DIRN)
    alpha_flight_birds, mig_spd_birds = SearchAlphaFlightAndMagMig(alpha_mig_true=mig_dirn_birds,
                                                                       spd_flight=FLIGHT_SPD_BIRDS,
                                                                       spd_wind=WIND_SPD, alpha_wind=WIND_DIRN)
    print(alpha_flight_insects)
    print(mig_spd_insects)
    print(alpha_flight_birds)
    print(mig_spd_birds)



    # Generate bird and insect vads.


    return

if __name__ == '__main__':
    Main()