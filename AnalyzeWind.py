from InterpolateData import *
from RadarHCAUtils import *
from ClfUtils import classify_echoes
from SoundingDataUtils import *
from RapUtils import *
from NexradUtils import *
from VADMaskEnum import VADMask
from TrueWindEnum import *
from VADUtils import VADWindProfile
import GeneralUtils as gu
from collections import Counter

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 11}
plt.rc('font', **font)

# TODO
# need check for if sounding is available.
# Might need to tune for best availability threshold.

RAP_FILE_BASE = "rapanl_130_{}{}{}{}.g2.tar"


def VisualizeWinds(vad_profiles_job, sounding_wind_df, max_height, description_jobs, title_str, prop_str,
                   output_folder, figure_prefix, save_plots, figure_suffix=''):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    figure_suffix = figure_suffix.replace(' ', '_')

    # TODO add biological echoes to colour wheel
    red_color_wheel = {VADMask.weather: "gold", VADMask.insects: "tomato", VADMask.birds: "lime",
                       VADMask.biological: "fuchsia"}
    blue_color_wheel = {VADMask.weather: "deepskyblue", VADMask.insects: "blueviolet", VADMask.birds: "cornflowerblue",
                        VADMask.biological: "darkslategray"}

    # Plot for speed and direction.
    fig, ax = plt.subplots(1, 2)

    # Radar speed and direction.
    for job_id in vad_profiles_job.keys():
        wind_profile_vad = vad_profiles_job[job_id]
        vad_height_idx = wind_profile_vad['height'] < max_height

        ax[0].scatter(wind_profile_vad["wind_direction"][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                      color=blue_color_wheel[job_id], marker=description_jobs[job_id][1], alpha=0.35)
        ax[0].plot(wind_profile_vad["wind_direction"][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                   color=blue_color_wheel[job_id], label=description_jobs[job_id][0] + " dir")
        ax[1].scatter(wind_profile_vad["wind_speed"][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                      color=red_color_wheel[job_id], marker=description_jobs[job_id][1], alpha=0.35)
        ax[1].plot(wind_profile_vad["wind_speed"][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                   color=red_color_wheel[job_id], label=description_jobs[job_id][0] + " spd")

    # Sounding speed and direction.
    sounding_height_idx = sounding_wind_df['HGHT'] < max_height
    sounding_plot_idx = np.isfinite(sounding_wind_df["DRCT"])
    sounding_plot_idx = np.logical_and(sounding_plot_idx, sounding_height_idx)

    ax[0].scatter(sounding_wind_df["DRCT"][sounding_plot_idx], sounding_wind_df['HGHT'][sounding_plot_idx],
                  marker='o', color="blue", alpha=0.4)
    ax[0].plot(sounding_wind_df["DRCT"][sounding_plot_idx], sounding_wind_df['HGHT'][sounding_plot_idx],
               label="wind dir", color="blue", linestyle='dashed', alpha=0.8)
    ax[1].scatter(sounding_wind_df["SMPS"][sounding_plot_idx], sounding_wind_df['HGHT'][sounding_plot_idx],
                  color="red", marker='o', alpha=0.4)
    ax[1].plot(sounding_wind_df["SMPS"][sounding_plot_idx], sounding_wind_df['HGHT'][sounding_plot_idx],
               label="wind spd", color="red", linestyle='dashed', alpha=0.8)

    ax[0].set_xlim(0, 360)
    ax[0].set_ylim(0, 1.4 * max_height)
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_xlim(0, 20)
    ax[1].set_ylim(0, 1.4 * max_height)
    ax[1].grid(True)
    ax[1].legend()
    fig.suptitle(title_str)
    if save_plots:
        plt.savefig(
            os.path.join(output_folder, "".join([figure_prefix, "_wind_comparison_spherical_", figure_suffix, ".png"])),
            dpi=200)
    # plt.close(fig)

    # Plot for U and V wind components.
    fig_comp, ax_comp = plt.subplots()
    ax_comp_sec = ax_comp.twinx()

    # Mean reflectivity vs height.
    wind_profile_vad = vad_profiles_job[VADMask.insects]
    vad_height_idx = wind_profile_vad['height'] < max_height
    ax_comp.plot(wind_profile_vad['mean_ref'][vad_height_idx], wind_profile_vad['height'][vad_height_idx], color='black',
             label="Z")

    for job_id in vad_profiles_job.keys():
        wind_profile_vad = vad_profiles_job[job_id]
        vad_height_idx = wind_profile_vad['height'] < max_height

        # Radar wind components.
        ax_comp.scatter(wind_profile_vad['wind_U'][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                    color=blue_color_wheel[job_id], marker=description_jobs[job_id][1], alpha=0.35)
        ax_comp.plot(wind_profile_vad['wind_U'][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                 color=blue_color_wheel[job_id], alpha=0.8, label=description_jobs[job_id][0] + " vad_U")
        ax_comp.scatter(wind_profile_vad['wind_V'][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                    color=red_color_wheel[job_id], marker=description_jobs[job_id][1], alpha=0.35)
        ax_comp.plot(wind_profile_vad['wind_V'][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                 color=red_color_wheel[job_id], alpha=0.8, label=description_jobs[job_id][0] + " vad_V")

    # Sounding wind components.
    sounding_height_idx = sounding_wind_df['HGHT'] < max_height
    ax_comp.scatter(sounding_wind_df['windU'][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
                marker='o', color="blue", alpha=0.1)
    ax_comp.scatter(sounding_wind_df['windV'][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
                color="red", alpha=0.1)
    ax_comp.plot(sounding_wind_df['windU'][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
             label="wind_U", color="blue", linestyle='dashed')
    ax_comp.plot(sounding_wind_df['windV'][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
             label="wind_V", color="red", linestyle='dashed')

    # ax_comp.set_ylim(0, 1.4 * max_height)
    ax_comp.set_ylim(0, 1.2 * max_height)
    # ax_comp.set_xlim(-16, 23)
    ax_comp.set_xlim(-2, 20)
    ax_comp.grid(True)
    ax_comp.set_xlabel("Wind components [mps]")
    ax_comp.set_ylabel("Height [m]")
    # ax_comp_sec.set_ylim(0, 1.4 * max_height)
    ax_comp_sec.set_ylim(0, 1.2 * max_height)
    # ax_comp_sec.set_xlim(-16, 23)
    ax_comp_sec.set_xlim(-2, 20)
    ax_comp_sec.set_ylabel("Height [m]")

    ax_comp.set_title(title_str + '\n' + prop_str)
    ax_comp.legend(loc = 'upper center', ncol=3, fontsize = 10.5)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_folder,
                                 "".join([figure_prefix, "_wind_comparison_components_", figure_suffix, ".png"])),
                    dpi=200)
    plt.show()
    plt.close(fig_comp)
    plt.close('all')

    return


def InterpolateWindComponents(windU, windV, height_grid, height_grid_interp, max_height_diff, max_height,
                              show_debug_plot=False):
    height_grid_interp = height_grid_interp[height_grid_interp < max_height]
    windU_interp = Interpolate(x=height_grid, y=windU, x_interp=height_grid_interp, max_delta_x=max_height_diff)
    windV_interp = Interpolate(x=height_grid, y=windV, x_interp=height_grid_interp, max_delta_x=max_height_diff)

    # Impose max height.
    # TODO Use idx_height for plots directly instead.
    idx_height = height_grid < max_height
    height_grid = height_grid[idx_height]
    windU = windU[idx_height]
    windV = windV[idx_height]

    idx_height_vad = height_grid_interp < max_height
    idx_valid = np.logical_not(np.logical_and(np.isnan(windU_interp), np.isnan(windV_interp)))
    idx_valid = np.logical_and(idx_valid, idx_height_vad)

    height_grid_interp = height_grid_interp[idx_valid]
    windU_interp = windU_interp[idx_valid]
    windV_interp = windV_interp[idx_valid]

    if show_debug_plot:
        # plot.
        plt.figure()
        plt.scatter(windU, height_grid, color='blue', alpha=0.5)
        plt.scatter(windU_interp, height_grid_interp, color='red', alpha=0.5)
        plt.plot(windU, height_grid, color='blue')
        plt.ylim(0, max_height)
        plt.title("U")

        plt.figure()
        plt.scatter(windV, height_grid, color='blue', alpha=0.5)
        plt.scatter(windV_interp, height_grid_interp, color='red', alpha=0.5)
        plt.plot(windV, height_grid, color='blue')
        plt.ylim(0, max_height)
        plt.title("V")

        plt.figure()
        plt.scatter(windU_interp, height_grid_interp, color='blue', label='U')
        plt.scatter(windV_interp, height_grid_interp, color='red', label='V')
        plt.legend()

        plt.show()

    return windU_interp, windV_interp, height_grid_interp


"""
Interpolate wind U and V components from sounding and rap 130.
"""
def InterpolateSoundingWind(sounding_df, height_grid_interp, max_height_diff, max_height):
    height_grid = sounding_df['HGHT']
    windU = sounding_df['windU']
    windV = sounding_df['windV']

    windU_interp, windV_interp, height_grid_interp = InterpolateWindComponents(windU=windU, windV=windV,
                                                                               height_grid=height_grid,
                                                                               height_grid_interp=height_grid_interp,
                                                                               max_height_diff=max_height_diff,
                                                                               max_height=max_height)
    idx_height = height_grid < max_height
    height_grid = height_grid[idx_height]
    windU = windU[idx_height]
    windV = windV[idx_height]

    idx_height_interp = idx_height[idx_height]
    idx_height_interp = idx_height_interp.append(pd.Series([False for i in range(len(height_grid_interp))]),
                                                 ignore_index=True)
    height_grid_interp = height_grid.append(height_grid_interp, ignore_index=True)
    windU_interp = windU.append(pd.Series(windU_interp), ignore_index=True)
    windV_interp = windV.append(pd.Series(windV_interp), ignore_index=True)
    df_interp = pd.DataFrame(
        {'HGHT': height_grid_interp, 'windU': windU_interp, 'windV': windV_interp})

    remnant_variables = set(sounding_df.columns) - set(df_interp.columns)
    for remnant_var in remnant_variables:
        df_interp[remnant_var] = np.nan
        tmp = sounding_df.loc[idx_height, remnant_var]
        tmp.reset_index(drop=True, inplace=True)
        df_interp.loc[idx_height_interp, remnant_var] = tmp

    spd_interp, dirn_interp = Cartesian2PolarComponentsDf(u=df_interp["windU"], v=df_interp["windV"])
    idx_finite_smps = np.isfinite(df_interp['SMPS'])
    idx_infinite_smps = np.logical_not(idx_finite_smps)
    df_interp.loc[idx_infinite_smps, "SMPS"] = spd_interp[idx_infinite_smps]
    df_interp.loc[idx_infinite_smps, "DRCT"] = dirn_interp[idx_infinite_smps]

    # TODO(pjatau) remove this after a few runs
    assert round(df_interp['SMPS']).equals(round(spd_interp))
    assert round(df_interp['DRCT']).equals(round(dirn_interp))

    df_interp = df_interp.sort_values(by=['HGHT'])

    return df_interp


def InterpolateVADWind(vad_df, height_grid_interp, max_height_diff, max_height):
    height_grid = vad_df['height']
    windU = vad_df['wind_U']
    windV = vad_df['wind_V']

    windU_interp, windV_interp, height_grid_interp = InterpolateWindComponents(windU=windU, windV=windV,
                                                                               height_grid=height_grid,
                                                                               height_grid_interp=height_grid_interp,
                                                                               max_height_diff=max_height_diff,
                                                                               max_height=max_height,
                                                                               show_debug_plot=False)
    idx_height = height_grid < max_height
    height_grid = height_grid[idx_height]
    windU = windU[idx_height]
    windV = windV[idx_height]

    height_grid_interp = height_grid.append(height_grid_interp, ignore_index=True)
    windU_interp = windU.append(pd.Series(windU_interp), ignore_index=True)
    windV_interp = windV.append(pd.Series(windV_interp), ignore_index=True)
    df_interp = pd.DataFrame(
        {'height': height_grid_interp, 'wind_U': windU_interp, 'wind_V': windV_interp})

    spd_interp, dirn_interp = Cartesian2PolarComponentsDf(u=df_interp["wind_U"], v=df_interp["wind_V"])

    df_interp['wind_speed'] = spd_interp
    df_interp.loc[range(len(vad_df)), 'wind_speed'] = vad_df['wind_speed']
    df_interp['wind_direction'] = dirn_interp
    df_interp.loc[range(len(vad_df)), 'wind_direction'] = vad_df['wind_direction']
    df_interp['num_samples'] = np.nan
    df_interp.loc[range(len(vad_df)), 'num_samples'] = vad_df['num_samples']
    df_interp['mean_ref'] = np.nan
    df_interp.loc[range(len(vad_df)), 'mean_ref'] = vad_df['mean_ref']

    # TODO(pjatau) remove this after a few runs
    assert round(df_interp['wind_speed']).equals(round(spd_interp))
    assert round(df_interp['wind_direction']).equals(round(dirn_interp))

    df_interp = df_interp.sort_values(by=['height'])

    return df_interp


def PrepareAnalyzeWindInputs(radar_data_file, batch_folder, radar_data_folder, hca_data_folder, clf_file, is_batch,
                             norm_stats_file=None, correct_hca_weather=False, biw_norm_stats_file=None,
                             biw_clf_file=None, allowed_el_hca = None):
    if is_batch:
        start_day = int(radar_data_file[10:12])
        stop_day = start_day

        date_pattern = "*{}_V06*".format("".join([radar_data_file[:10], "{}*"]))
        hca_data_folder = os.path.join(hca_data_folder, batch_folder)
        batch_folder_path_radar = os.path.join(radar_data_folder, batch_folder)
        radar_scans_day = GetFileListRadar(batch_folder_path_radar, start_day=start_day, stop_day=stop_day,
                                           date_pattern=date_pattern)
        l3_files_dic = GetFileListL3(hca_data_folder)

        # find subpath for radar file
        radar_subpath = None
        for key in radar_scans_day.keys():
            for element in radar_scans_day[key]:
                if element.endswith(radar_data_file):
                    radar_subpath = element
                    break

        data_table, radar_obj, hca_vol = PrepareDataTable(batch_folder_path_radar,
                                                          radar_subpath,
                                                          hca_data_folder, l3_files_dic,
                                                          max_range=400,
                                                          clf_file=clf_file, norm_stats_file=norm_stats_file,
                                                          correct_hca_weather=correct_hca_weather,
                                                          max_height_VAD=1000, biw_norm_stats_file=biw_norm_stats_file,
                                                          biw_clf_file=biw_clf_file, allowed_el_hca=allowed_el_hca)

        target_folder = os.path.split(radar_subpath)
        radar_data_file = target_folder[1]
        radar_data_folder = os.path.join(radar_data_folder, batch_folder, target_folder[0])
        return radar_data_file, radar_data_folder, data_table, radar_obj, hca_vol
    else:
        radar_data_folder = os.path.join(radar_data_folder, batch_folder)
        return radar_data_file, radar_data_folder, None, None, None


def AnalyzeWind(radar_data_file, radar_data_folder, hca_data_folder, radar_t_sounding, sounding_log_dir,
                norm_stats_file, clf_file, vad_jobs, figure_dir, max_range=300, max_height_VAD=1000,
                match_radar_and_sounding_grid=True, save_wind_figure=False, radar=None, hca_vol=None, data_table=None,
                l3_filelist=None, vad_debug_params=None, ground_truth_source=WindSource.sounding, rap_folder=None,
                log_dir_rap='./atmospheric_model_data/UV_wind_logs',
                log_file_base_rap='{}_windcomponents_lat_{}_lon_{}.pkl', correct_hca_weather=True,
                biw_norm_stats_file=None, biw_clf_file=None, allowed_el_hca=None, use_vad_weights=False):
    radar_data_file_no_ext = os.path.splitext(radar_data_file)[0]
    radar_name, year, month, day, hh, mm, ss = read_info_from_radar_name(radar_data_file)

    # Log options to command line
    print()
    print("Correct hca weather: \t {}".format(correct_hca_weather))

    # Sounding.
    station_id = str(radar_t_sounding[radar_name][0][0])

    sounding_url_base = "http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR={}&MONTH={}&FROM={}&TO={}&STNM={}"

    gt_desc = GetWindSourceDescription(ground_truth_source)

    if not os.path.isdir(sounding_log_dir):
        os.makedirs(sounding_log_dir)

    # Read radar data.
    if radar is None:
        radar = pyart.io.read(os.path.join(radar_data_folder, radar_data_file))

    location_radar = {"latitude": radar.latitude['data'][0],
                      "longitude": radar.longitude['data'][0],
                      "height": radar.altitude['data'][0]}

    # Sounding wind profile.
    if ground_truth_source == WindSource.sounding:
        print("Ground truth wind is sounding.")
        year_sounding, month_sounding, gt_wind_ddhh = GetSoundingDateTimeFromRadarFile(radar_data_file)
        gt_wind_df, gt_wind_location, sounding_url = GetSoundingWind(sounding_url_base, radar_data_file,
                                                                     location_radar,
                                                                     station_id, sounding_log_dir,
                                                                     showDebugPlot=False, log_sounding_data=True,
                                                                     force_website_download=False)

        # TODO(pjatau) use more intuitive(relevant) description for ground truth.
        # gt_loc_desc = station_infos[radar_t_sounding[radar_name]][1]
        gt_loc_desc = str(radar_t_sounding[radar_name][0][0])
        if gt_wind_df is None:
            print("Sounding data is not available for this scan. Aborting wind analysis ...")
            return None, None, None
    elif ground_truth_source == WindSource.rap_130:
        print("Ground truth wind is rap 130.")
        rap_file = RAP_FILE_BASE.format(year, month, day, hh)
        gt_wind_df, gt_wind_location = GetRapWindProfileRelativeToRadar(location_radar['latitude'],
                                                                        location_radar['longitude'], location_radar,
                                                                        rap_folder, rap_file, log_dir_rap,
                                                                        log_file_base_rap, show_fig=False,
                                                                        force_update=False,
                                                                        save_wind_profile=True)
        # TODO(pjatau) move magnitude/direction calculation into GetRapWindProfileRelativeToRadar
        spd_rap, dirn_rap = Cartesian2PolarComponentsDf(u=gt_wind_df["windU"], v=gt_wind_df["windV"])
        gt_wind_df['SMPS'] = spd_rap
        gt_wind_df['DRCT'] = dirn_rap

        gt_wind_ddhh = ''.join([day, hh])
        gt_loc_desc = "Lat {}{}, Lon {}{}".format(round(location_radar['latitude'], 2), '$^{\circ}$',
                                                  round(location_radar['longitude'], 2),
                                                  '$^{\circ}$')
    else:
        print("Unknown ground truth source specified. Terminating program ...")
        quit()

    distance_radar_sounding = GetHaverSineDistance(location_radar["latitude"], location_radar["longitude"],
                                                   gt_wind_location["latitude"],
                                                   gt_wind_location["longitude"])
    distance_radar_sounding = round(distance_radar_sounding / 1000, 2)

    # Read HCA data.
    if l3_filelist is None:
        radar_base = radar_data_file[:12]
        radar_data_folder = os.path.join(radar_data_folder, radar_base)
        hca_data_folder = os.path.join(hca_data_folder, radar_base)

    if hca_vol is None:
        if l3_filelist is None:
            hca_vol = GetHcaVol(hca_data_folder, radar_data_file_no_ext, allowed_el_hca)
        else:
            hca_vol = GetHcaVolFromFileList(hca_data_folder, radar_data_file_no_ext, l3_filelist, allowed_el_hca)

    # Data table.
    if data_table is None:
        data_table = MergeRadarAndHCAUpdate(radar, hca_vol, max_range)
        data_table["height"] = data_table["range"] * np.sin(data_table["elevation"] * np.pi / 180)

        if correct_hca_weather:
            X_biw = data_table.loc[:, ['differential_reflectivity', 'differential_phase', 'cross_correlation_ratio']]
            X_biw.rename(columns={"differential_reflectivity": "ZDR"}, inplace=True)
            X_biw.rename(columns={"differential_phase": "pdp"}, inplace=True)
            X_biw.rename(columns={"cross_correlation_ratio": "RHV"}, inplace=True)
            data_table['BIWClass'] = -1
            biw_class, _ = classify_echoes(X_biw, biw_clf_file, norm_stats_path=biw_norm_stats_file)
            data_table.loc[:, 'BIWClass'] = biw_class
            # data_table.loc[:, 'BIWClass'] = classify_echoes(X_biw, biw_clf_file, norm_stats_path=biw_norm_stats_file)

            # Correct HCA's misclassification of birds as weather within VAD region.
            # if weather hca, and non-weather biw, and within collection region, set to biological
            correction_msk = data_table["height"] < max_height_VAD
            weather_hca = np.logical_and(data_table["hca"] >= 30.0, data_table["hca"] <= 100.0)
            non_weather_biw = data_table['BIWClass'] != 3
            correction_msk = np.logical_and(correction_msk, weather_hca)
            correction_msk = np.logical_and(correction_msk, non_weather_biw)
            data_table.loc[correction_msk, "hca"] = 10.0

        # TODO Precious. Get original ZDR mask.
        data_table["mask_differential_reflectivity"] = data_table["differential_reflectivity"] > -8.0
        data_table["hca_bio"] = data_table["hca"] == 10.0
        data_table["hca_weather"] = np.logical_and(data_table["hca"] >= 30.0, data_table["hca"] <= 100.0)

        height_binsize = 0.04  # height bins in km
        data_table["height_bin_meters"] = (np.floor(
            data_table["height"] / height_binsize) + 1) * height_binsize - height_binsize / 2
        data_table["height_bin_meters"] *= 1000

        # Apply bird-insect classifier. -1 is non-bio, 1 is bird and 0 is insects.
        echo_mask = np.logical_and(data_table["mask_differential_reflectivity"], data_table["hca_bio"])
        X = data_table.loc[
            echo_mask, ['differential_reflectivity', 'differential_phase', 'cross_correlation_ratio']]
        X.rename(columns={"differential_reflectivity": "ZDR"}, inplace=True)
        X.rename(columns={"differential_phase": "pdp"}, inplace=True)
        X.rename(columns={"cross_correlation_ratio": "RHV"}, inplace=True)
        data_table['BIClass'], data_table['BIProb'] = -1, -1
        if not X.empty:
            bi_class, bi_probs = classify_echoes(X, clf_file, norm_stats_file)
            data_table.loc[echo_mask, 'BIClass'] = bi_class
            data_table.loc[echo_mask, 'BIProb'] = bi_probs[:, 1]
            # data_table.loc[echo_mask, 'BIClass'] = classify_echoes(X, clf_file, norm_stats_path=norm_stats_file)


    # VAD wind profile
    signal_func = lambda x, t: x[0] * np.sin(2 * np.pi * (1 / 360) * t + x[1])

    if vad_debug_params:
        vad_heights = vad_debug_params['vad_heights']
        show_vad_plot = vad_debug_params['show_plot']
    else:
        vad_heights = np.arange(80, max_height_VAD, 40)
        # vad_heights = np.array([480])
        show_vad_plot = False

    vad_profiles_job = {}
    for vad_mask in vad_jobs:
        wind_profile_vad = VADWindProfile(signal_func, vad_heights, vad_mask, data_table,
                                          showDebugPlot=show_vad_plot, use_weights=use_vad_weights)
        vad_profiles_job[vad_mask] = wind_profile_vad

    # Interpolate wind components.
    max_height = 1.1 * max_height_VAD  # meters.
    max_height_diff = 250  # meters.
    height_grid_interp = wind_profile_vad['height']
    height_grid_interp = height_grid_interp[height_grid_interp < max_height]

    # Match, interpolate VAD and sounding/rap grid
    if match_radar_and_sounding_grid:
        gt_wind_df_interp = InterpolateSoundingWind(sounding_df=gt_wind_df,
                                                    height_grid_interp=height_grid_interp,
                                                    max_height_diff=max_height_diff,
                                                    max_height=max_height)
        height_grid_interp = gt_wind_df['HGHT']
        height_grid_interp = height_grid_interp[height_grid_interp < max_height]
        vad_profiles_job_interp = {}
        for vad_mask in vad_jobs:
            vad_profiles_job_interp[vad_mask] = InterpolateVADWind(vad_df=vad_profiles_job[vad_mask],
                                                                   height_grid_interp=height_grid_interp,
                                                                   max_height_diff=max_height_diff,
                                                                   max_height=max_height)

    # Plots.
    height_msk = data_table["height_bin_meters"] < max_height_VAD
    total_echoes = np.sum(np.logical_or(data_table["hca_bio"][height_msk], data_table["hca_weather"][height_msk]))

    prop_birds = np.sum(data_table['BIClass'][height_msk] == 1) / total_echoes
    prop_birds = round(prop_birds * 100)
    prop_insects = np.sum(data_table['BIClass'][height_msk] == 0) / total_echoes
    prop_insects = round(prop_insects * 100)
    prop_weather = np.sum(data_table['hca_weather'][height_msk] == 1) / total_echoes
    prop_weather = round(prop_weather * 100)
    prop_str = "{}% birds, {}% insects, {}% weather".format(prop_birds, prop_insects, prop_weather)
    echo_dist_VAD = {'bird': prop_birds, 'insects': prop_insects, 'weather': prop_weather}

    title_str = "{}, {}/{}/{}, {}:{}:{} UTC.\n{} km from {} UTC, {} {}.".format(
        radar_name, year, month,
        day, hh, mm, ss,
        distance_radar_sounding,
        gt_wind_ddhh[2:],
        gt_loc_desc, gt_desc)

    description_jobs = {VADMask.biological: ("bio", "."), VADMask.insects: ("ins", "2"),
                        VADMask.weather: ("wea", "d"), VADMask.birds: ("bir", "^")}
    figure_prefix = os.path.splitext(radar_data_file)[0]

    if match_radar_and_sounding_grid:
        VisualizeWinds(vad_profiles_job_interp, gt_wind_df_interp, 1100, description_jobs, title_str, prop_str,
                       figure_dir, figure_prefix, save_wind_figure)
        return vad_profiles_job_interp, gt_wind_df_interp, echo_dist_VAD

    VisualizeWinds(vad_profiles_job, gt_wind_df, 1100, description_jobs, title_str, prop_str,
                   figure_dir, figure_prefix, save_wind_figure, figure_suffix=gt_desc)
    return vad_profiles_job, gt_wind_df, echo_dist_VAD
