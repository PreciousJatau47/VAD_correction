import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from VADMaskEnum import VADMask, GetVADMaskDescription
from WindUtils import CalcPolarDiffVec
from TrueWindEnum import *


def WindError(x1, y1, x2, y2, error_fn, reduce_fn):
    mapper = {}
    for j in range(len(x2)):
        mapper[x2[j]] = j

    x_scores = x1
    scores = []
    for i in range(len(x1)):
        if x1[i] in mapper.keys():
            ind = mapper[x1[i]]
            error = error_fn(y1[i], y2[ind])
            scores.append(error)
        else:
            scores.append(np.nan)

    reduced_error = reduce_fn(scores)
    return reduced_error, scores, x_scores



def GetVelocitiesScan(wind_file, vad, sounding_df, echo_dist, figure_dir, debug_plots=False):
    # Sounding
    vel_profiles = sounding_df.loc[:, ["HGHT", "DRCT", "SMPS"]]
    vel_profiles.rename(columns={"HGHT": "height_m", "DRCT": "wind_direction", "SMPS": "wind_speed"}, inplace=True)
    wind_file_no_ext = os.path.splitext(wind_file)[0]
    vel_profiles['file_name'] = wind_file_no_ext
    vel_profiles = vel_profiles.drop_duplicates(subset='height_m', keep='last')
    vel_profiles['height_m'] = round(vel_profiles['height_m'])

    # VAD
    vad_vel_cols_base = ["height", "wind_speed", "wind_direction", "num_samples", "num_samples_50", "coverage_perc"]
    new_cols_base = ["height_m", "{}_speed", "{}_direction", "num_{}_height", "num_{}_height_50", "{}_coverage_perc"]

    for echo in vad:
        if echo == VADMask.weather:
            continue

        vad_vel_cols = vad_vel_cols_base.copy()
        new_cols = new_cols_base.copy()
        new_cols = [new_cols[i] if i < 1 else new_cols[i].format(GetVADMaskDescription(echo)) for i in
                    range(len(new_cols))]

        if echo == VADMask.biological:
            vad_vel_cols.insert(1, 'mean_ref')
            vad_vel_cols.insert(2, 'mean_prob')
            new_cols.insert(1, '_'.join(['mean_ref', GetVADMaskDescription(echo)]))
            new_cols.insert(2, '_'.join(['mean_prob', GetVADMaskDescription(echo)]))

        echo_df = vad[echo].loc[:, vad_vel_cols]
        echo_df.rename(columns=dict(zip(vad_vel_cols, new_cols)), inplace=True)
        echo_df = echo_df.drop_duplicates(subset='height_m', keep='last')
        echo_df['height_m'] = round(echo_df['height_m'])

        vel_profiles = pd.merge(vel_profiles, echo_df, on="height_m", how="outer")

    vel_profiles['prop_birds'] = echo_dist['bird']
    vel_profiles['prop_insects'] = echo_dist['insects']
    vel_profiles['prop_weather'] = echo_dist['weather']
    vel_profiles['file_name'] = wind_file_no_ext

    return vel_profiles

def extract_features_from_ppi_name(s):
    radar_name = s[:4]
    yyyy = s[4:8]
    month = float(s[8:10])
    day = float(s[10:12])
    hh = float(s[13:15])
    mm = float(s[15:17])
    ss = float(s[17:19])
    time_hour = float(hh) + float(mm) / 60 + float(ss) / 3600
    return radar_name, yyyy, month, day, time_hour


def UpdateWindError(wind_error_df, target_file, vad_profiles, sounding_df, echo_dist_VAD, error_fn, reduce_fn,
                    ground_truth_source, figure_dir, batch_folder, n_birds, n_insects, n_weather):
    if sounding_df is None or vad_profiles is None or target_file == '':
        return wind_error_df

    for echo_type in vad_profiles:
        if vad_profiles[echo_type].empty and echo_type != VADMask.external_l3_vad_profile:
            return wind_error_df

    airspeed_scan = GetVelocitiesScan(wind_file=target_file, vad=vad_profiles, sounding_df=sounding_df,
                                      echo_dist=echo_dist_VAD, figure_dir=figure_dir, debug_plots=True)

    # Calculate proportion of weather echoes for the whole scan.
    airspeed_scan["prop_weather_scan"] = n_weather * 100 / (
            n_birds + n_insects + n_weather)

    # Calculate proportion of insects relative to biological echoes.
    bird_ratio_bio = airspeed_scan['prop_birds']
    insects_ratio_bio = airspeed_scan['prop_insects']
    total = bird_ratio_bio + insects_ratio_bio
    insects_ratio_bio = np.divide(insects_ratio_bio, total)
    airspeed_scan['insect_prop_bio'] = insects_ratio_bio * 100
    # airspeed_scan = airspeed_scan.sort_values(by=['insect_prop_bio'])

    radar_name, yyyy, month, day, time_hour = extract_features_from_ppi_name(target_file)
    airspeed_scan["radar"] = radar_name
    airspeed_scan["year"] = yyyy
    airspeed_scan["month"] = month
    airspeed_scan["day"] = day
    airspeed_scan["time_hour"] = time_hour

    # Update airspeed table.
    if airspeed_scan is not None:
        wind_error_df = wind_error_df.append(airspeed_scan, ignore_index=True)

    return wind_error_df