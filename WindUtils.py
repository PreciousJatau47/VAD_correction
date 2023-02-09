import numpy as np
import pickle
import pandas as pd


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

def CalcCartesianDiffVec(u1, v1, u2, v2):
    u_diff = u1 - u2
    v_diff = v1 - v2
    return u_diff, v_diff

def CalcPolarDiffVec(spd1, dirn1, spd2, dirn2):
    u1, v1 = Polar2CartesianComponentsDf(spd1, dirn1)
    u2, v2 = Polar2CartesianComponentsDf(spd2, dirn2)
    u_diff, v_diff = CalcCartesianDiffVec(u1, v1, u2, v2)
    spd_diff, dirn_diff = Cartesian2PolarComponentsDf(u_diff, v_diff)
    return spd_diff, dirn_diff

def TestWindUtils():
    # Test for CalcCartesianDiffVec
    # v1 = [0 1], v2 = [1,0], result = [-1 1]
    exp_diff = [-1, 1]
    actual_diff = CalcCartesianDiffVec(u1=0, v1=1, u2=1, v2=0)
    assert np.array_equal(exp_diff, actual_diff)

    # Test for CalcPolarDiffVec
    # v1 = (4,0), v2 = (3,90), result = (5, 324]
    exp_pol_diff = (5, 323.13)
    actual_diff = CalcPolarDiffVec(spd1=4, dirn1=0, spd2=3, dirn2=90)
    assert (exp_pol_diff[0] - actual_diff[0]) < 0.5
    assert (exp_pol_diff[1] - actual_diff[1]) < 2

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

    # Test for flight vector
    scan_name_base = "KOHX20180503_180336_V06"

    exp_airspeed_path = './expected_results/KOHX_20180503_test_data\hca_weather_corrected\KOHX_20180503_test_data_rap_130.pkl'
    with open(exp_airspeed_path, 'rb') as p_in:
        exp_airspeed_df_full = pickle.load(p_in)
    p_in.close()

    # wind_error_path = './batch_analysis_logs\KOHX_20180503_test_data_launched_202327_15/KOHX_20180503_test_data.pkl'
    wind_error_path = './batch_analysis_logs\KOHX_20180503_test_data_launched_202328_16/KOHX_20180503_test_data.pkl'
    with open(wind_error_path, 'rb') as p_in:
        wind_error_df_full = pickle.load(p_in)[0]
    p_in.close()

    # Expected flight speeds
    scan_name = "".join([scan_name_base, "_wind"])
    idx = exp_airspeed_df_full['file_name'] == scan_name
    exp_airspeed_df = exp_airspeed_df_full.loc[idx, ['height_m','airspeed_birds','airspeed_insects']]
    exp_airspeed_df = exp_airspeed_df.groupby(["height_m"], as_index=False).mean()
    exp_airspeed_df = exp_airspeed_df.sort_values(by=["height_m"])

    # Predicted flight speeds.
    idx = wind_error_df_full["file_name"] == scan_name_base
    wind_error_df = wind_error_df_full.loc[idx,
                    ['height_m', 'wind_speed', 'wind_direction', 'birds_speed', 'birds_direction','insects_direction','insects_speed']]
    wind_error_df = wind_error_df.groupby(["height_m"], as_index=False).mean()
    wind_error_df = wind_error_df.sort_values(by=["height_m"])
    wind_error_df["fspeed_birds"], _ = CalcPolarDiffVec(spd1=wind_error_df['birds_speed'],
                                                        dirn1=wind_error_df[
                                                            'birds_direction'],
                                                        spd2=wind_error_df['wind_speed'],
                                                        dirn2=wind_error_df[
                                                            'wind_direction'])
    wind_error_df["fspeed_insects"], _ = CalcPolarDiffVec(spd1=wind_error_df['insects_speed'],
                                                          dirn1=wind_error_df[
                                                              'insects_direction'],
                                                          spd2=wind_error_df['wind_speed'],
                                                          dirn2=wind_error_df[
                                                              'wind_direction'])

    combined_df = exp_airspeed_df.copy()
    combined_df = pd.merge(combined_df, wind_error_df.loc[:,["height_m", "fspeed_birds", 'fspeed_insects']], on="height_m", how='left')

    idx_finite_birds = np.isfinite(combined_df["airspeed_birds"])
    combined_df = combined_df.loc[idx_finite_birds, :]

    assert np.logical_and.reduce(np.isclose(a=combined_df["airspeed_birds"], b=combined_df["fspeed_birds"], rtol=0.5,
                                            atol=0.5)), "predicted bird airspeeds differ from expected airspeeds"
    assert np.logical_and.reduce(np.isclose(a=combined_df["airspeed_insects"], b=combined_df["fspeed_insects"], rtol=0.5,
                                            atol=0.5)), "predicted insect airspeeds differ from expected airspeeds"

    return

TestWindUtils()
