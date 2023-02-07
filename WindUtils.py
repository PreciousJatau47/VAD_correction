import numpy as np
import pickle


def Polar2CartesianComponentsDf(spd, dirn):
    """
     Converts 2D polar wind to cartesian components.
    :param spd:
    :param dirn: wind direction measured in degrees from the positive y axis.
    :return:
    """
    u = spd * np.sin(dirn * np.pi / 180)
    v = spd * np.cos(dirn * np.pi / 180)
    return u, v


def Cartesian2PolarComponentsDf(u, v):
    """
    Converts 2D cartesian wind components to polar components.
    :param u:
    :param v:
    :return:
    """
    spd = np.sqrt(u ** 2 + v ** 2)
    dirn = np.rad2deg(np.arctan2(u, v))
    dirn = (dirn + 360) % 360
    return spd, dirn


def TestWindUtils():

    # Test for sounding.
    with open('./expected_results/sample_balloon_sounding.pkl', 'rb') as p_in:
        sounding_df = pickle.load(p_in)
    p_in.close()

    exp_u = sounding_df["windU"]
    exp_v = sounding_df["windV"]
    exp_spd = sounding_df["SMPS"]
    exp_dirn = sounding_df["DRCT"]

    pred_spd, pred_dirn = Cartesian2PolarComponentsDf(exp_u, exp_v)
    pred_u, pred_v = Polar2CartesianComponentsDf(exp_spd, exp_dirn)

    assert round(exp_spd).equals(round(pred_spd)), "predicted speed does not match true speed."
    assert round(exp_dirn).equals(round(pred_dirn)), "predicted direction does not match true direction."
    assert round(exp_u).equals(round(pred_u)), "predicted U-wind does not match true U-wind."
    assert round(exp_v).equals(round(pred_v)), "predicted V-wind does not match true V-wind."


    # Test for VAD.
    with open('./expected_results/sample_vad_profile_bio.pkl', 'rb') as p_in:
        vad_bio_df = pickle.load(p_in)
    p_in.close()

    exp_u_vad = vad_bio_df["wind_U"]
    exp_v_vad = vad_bio_df["wind_V"]
    exp_spd_vad = vad_bio_df["wind_speed"]
    exp_dirn_vad = vad_bio_df["wind_direction"]

    pred_spd_vad, pred_dirn_vad = Cartesian2PolarComponentsDf(exp_u_vad, exp_v_vad)
    pred_u_vad, pred_v_vad = Polar2CartesianComponentsDf(exp_spd_vad, exp_dirn_vad)

    assert round(exp_spd_vad).equals(round(pred_spd_vad)), "predicted speed does not match true speed."
    assert round(exp_dirn_vad).equals(round(pred_dirn_vad)), "predicted direction does not match true direction."
    assert round(exp_u_vad).equals(round(pred_u_vad)), "predicted U-wind does not match true U-wind."
    assert round(exp_v_vad).equals(round(pred_v_vad)), "predicted V-wind does not match true V-wind."


    # exp_airspeed_path = './expected_results/KOHX_20180503_test_data\hca_weather_corrected\KOHX_20180503_test_data_rap_130.pkl'
    # with open(exp_airspeed_path, 'rb') as p_in:
    #     wind_error_df = pickle.load(p_in)
    # p_in.close()

    return

TestWindUtils()
