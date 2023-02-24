import sys
import matplotlib.pyplot as plt
from WindUtils import *
from VADUtils import *

# Wind constants
WIND_SPD = 8
WIND_DIRN = 20
FLIGHT_SPD_INSECTS = 1.924
FLIGHT_SPD_BIRDS = 6.101
WIND_OFFSET_INSECTS = -8
WIND_OFFSET_BIRDS = 14

signal_func = lambda x, t: x[0] * np.sin(2 * np.pi * (1 / 360) * t + x[1])

def GenerateVAD(speed, dirn, dirn_grid):
    phase = (90 - dirn) * np.pi / 180
    return signal_func([speed, phase], dirn_grid)

def GenerateNoiseNormal(mu=0, sdev=1, num_samples=100):
    return sdev * np.random.randn(num_samples) + mu

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


def GenerateWinds(wind_spd, wind_dirn, wind_offset_insects, num_samples, noise_mean=0, noise_sdev=1,
                  show_debug_plot=False):
    mig_dirn_insects = wind_dirn + wind_offset_insects
    mig_dirn_birds = wind_dirn + WIND_OFFSET_BIRDS

    # Search for missing wind components.
    alpha_flight_insects, mig_spd_insects = SearchAlphaFlightAndMagMig(alpha_mig_true=mig_dirn_insects,
                                                                       spd_flight=FLIGHT_SPD_INSECTS,
                                                                       spd_wind=wind_spd, alpha_wind=wind_dirn)
    alpha_flight_birds, mig_spd_birds = SearchAlphaFlightAndMagMig(alpha_mig_true=mig_dirn_birds,
                                                                   spd_flight=FLIGHT_SPD_BIRDS,
                                                                   spd_wind=wind_spd, alpha_wind=wind_dirn)

    print("Wind. speed: {} mps. direction: {} degrees.".format(wind_spd, wind_dirn))
    print("Birds. speed: {} mps. direction: {} degrees.".format(mig_spd_birds, mig_dirn_birds))
    print("Insects. speed: {} mps. direction: {} degrees.".format(mig_spd_insects, mig_dirn_insects))

    # Generate bird and insect vads.
    alpha_grid = np.linspace(0, 360, num_samples)
    noise = GenerateNoiseNormal(mu=noise_mean, sdev=noise_sdev, num_samples=num_samples)

    vad_winds = GenerateVAD(speed=wind_spd, dirn=wind_dirn, dirn_grid=alpha_grid)
    vad_birds = GenerateVAD(speed=mig_spd_birds, dirn=mig_dirn_birds, dirn_grid=alpha_grid) + noise
    vad_insects = GenerateVAD(speed=mig_spd_insects, dirn=mig_dirn_insects, dirn_grid=alpha_grid) + noise

    if show_debug_plot:
        title_str_base = '$wind: ({}\,mps, {} ^\circ). birds: ({}\,mps, {} ^\circ). insects: ({}\,mps, {} ^\circ).$'
        title_str = title_str_base.format(wind_spd, wind_dirn, round(mig_spd_birds, 1),
                                          round(mig_dirn_birds, 1),
                                          round(mig_spd_insects, 1),
                                          round(mig_dirn_insects, 1))
        fig, ax = plt.subplots()
        ax.plot(alpha_grid, vad_winds, c='black', alpha=0.2, label="wind")
        ax.plot(alpha_grid, vad_birds, c='blue', alpha=0.5, label="birds")
        ax.plot(alpha_grid, vad_insects, c='red', alpha=0.5, label="insects")
        ax.set_xlim(0, 360)
        ax.set_xlabel(r'$\Phi (^\circ)$')
        ax.set_ylabel('Speed (mps)')
        ax.grid(True)
        ax.set_title(title_str)
        plt.legend(loc='best')
        plt.show()

    return {'vad_winds': vad_winds, 'vad_birds': vad_birds, 'vad_insects': vad_insects}


# TODO(pjatau)
# 1. Work on cross contamination

def Main():

    num_samples = 1080
    wind_spd = WIND_SPD
    wind_dirn = 90 #WIND_DIRN
    wind_offset_insects = WIND_OFFSET_INSECTS #-4

    winds = GenerateWinds(wind_spd=wind_spd, wind_dirn=wind_dirn, wind_offset_insects=wind_offset_insects,
                          num_samples=num_samples, noise_mean=0, noise_sdev=1, show_debug_plot=False)

    insect_ratio_bio = 0.5
    num_birds_true = int((1-insect_ratio_bio)*num_samples)
    num_insects_true = int(insect_ratio_bio*num_samples)
    print(num_birds_true)
    print(num_insects_true)

    # pred_speed, pred_dirn, vad_fit = fitVAD(pred_var=alpha_grid, resp_var=vad_gen, signal_func=signal_func,
    #                                         showDebugPlot=True, description='')
    # print("Desired. speed: {} mps. direction: {} degrees.".format(mig_spd_insects, desired_dirn))
    # print("Predicted. speed: {} mps. direction: {} degrees.".format(pred_speed, pred_dirn))

    return


if __name__ == '__main__':
    Main()
