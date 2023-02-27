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

# Classifier metrics
TPR = 0.862
TNR = 0.811

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

    return {'vad_winds': vad_winds, 'vad_birds': vad_birds, 'vad_insects': vad_insects}, alpha_grid


def MixVADWinds(az_grid, vad_birds, vad_insects, insect_ratio_bio=0.5, show_plot=False):
    num_samples = len(az_grid)
    idxs_az = [i for i in range(len(az_grid))]
    num_insects_true = int(insect_ratio_bio * num_samples)
    num_birds_true = num_samples - num_insects_true

    # Generate true vad mixture
    is_birds_true = np.zeros(az_grid.shape, dtype=bool)
    idx_birds = np.random.choice(a=idxs_az, size=num_birds_true, replace=False)
    is_birds_true[idx_birds] = 1
    idx_insects = list(set(idxs_az) - set(idx_birds))
    vad_measured = np.where(is_birds_true, vad_birds, vad_insects)

    if show_plot:
        col_true = np.where(is_birds_true, 'blue', 'red')
        fig, ax = plt.subplots()
        ax.scatter(az_grid, vad_measured, c=col_true, alpha=0.2)
        plt.show()

    return vad_measured, is_birds_true, idx_birds, idx_insects


def SimulateBIPredictions(tpr=0.95, tnr=0.95, idx_birds=[], idx_insects=[]):
    num_birds_true = len(idx_birds)
    num_insects_true = len(idx_insects)
    tpn = round(tpr * num_birds_true)
    fpn = round((1 - tnr) * num_insects_true)

    tp_idxs = np.random.choice(a=idx_birds, size=tpn, replace=False)
    fp_idxs = np.random.choice(a=idx_insects, size=fpn, replace=False)
    idx_birds_pred = np.concatenate([tp_idxs, fp_idxs])
    is_birds_pred = np.zeros((num_birds_true + num_insects_true,), dtype=bool)
    is_birds_pred[idx_birds_pred] = 1
    return is_birds_pred


# TODO(pjatau)
# 1. Calculate airspeeds for true and predicted classes.
# 2. Explore 1 for different insect_ratio_bio.
def Main():
    num_samples = 2160
    wind_spd = WIND_SPD
    wind_dirn = 90  # WIND_DIRN
    wind_offset_insects = WIND_OFFSET_INSECTS  # -4

    winds, az_grid = GenerateWinds(wind_spd=wind_spd, wind_dirn=wind_dirn, wind_offset_insects=wind_offset_insects,
                                   num_samples=num_samples, noise_mean=0, noise_sdev=1, show_debug_plot=False)

    # Mix bird and insect vads
    vad_measured, is_birds_true, idx_birds, idx_insects = MixVADWinds(az_grid=az_grid, vad_birds=winds['vad_birds'],
                                                                      vad_insects=winds['vad_insects'],
                                                                      insect_ratio_bio=0.5,
                                                                      show_plot=False)

    # Simulate model predictions
    # num. predicted birds = TPR * nbirds + (1-TNR)*ninsects
    tpr, tnr = 0.95, 0.95
    is_birds_pred = SimulateBIPredictions(tpr=tpr, tnr=tnr, idx_birds=idx_birds, idx_insects=idx_insects)

    # pred_speed, pred_dirn, vad_fit = fitVAD(pred_var=alpha_grid, resp_var=vad_gen, signal_func=signal_func,
    #                                         showDebugPlot=True, description='')
    # print("Desired. speed: {} mps. direction: {} degrees.".format(mig_spd_insects, desired_dirn))
    # print("Predicted. speed: {} mps. direction: {} degrees.".format(pred_speed, pred_dirn))

    return


if __name__ == '__main__':
    Main()
