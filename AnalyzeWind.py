from InterpolateData import *
from RadarHCAUtils import *
from SoundingDataUtils import *
from VADMaskEnum import VADMask
from VADUtils import VADWindProfile

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 11}
plt.rc('font', **font)


# TODO
# need check for if sounding is available.
# Might need to tune for best availability threshold.
def VisualizeWinds(vad_profiles_job, sounding_wind_df, max_height, description_jobs, title_str, prop_str,
                   output_folder, figure_prefix, save_plots):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    red_color_wheel = {VADMask.weather: "gold", VADMask.insects: "tomato", VADMask.birds: "lime"}
    blue_color_wheel = {VADMask.weather: "deepskyblue", VADMask.insects: "blueviolet", VADMask.birds: "cornflowerblue"}

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
               label="sound dir", color="blue", linestyle='dashed', alpha=0.8)
    ax[1].scatter(sounding_wind_df["SMPS"][sounding_plot_idx], sounding_wind_df['HGHT'][sounding_plot_idx],
                  color="red", marker='o', alpha=0.4)
    ax[1].plot(sounding_wind_df["SMPS"][sounding_plot_idx], sounding_wind_df['HGHT'][sounding_plot_idx],
               label="Sound spd", color="red", linestyle='dashed', alpha=0.8)

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
        plt.savefig(os.path.join(output_folder, "".join([figure_prefix, "_wind_comparison_spherical.png"])))

    # Plot for U and V wind components.
    plt.figure()

    # Mean reflectivity vs height.
    wind_profile_vad = vad_profiles_job[VADMask.insects]
    vad_height_idx = wind_profile_vad['height'] < max_height
    plt.plot(wind_profile_vad['mean_ref'][vad_height_idx], wind_profile_vad['height'][vad_height_idx], color='black',
             label="ref")

    for job_id in vad_profiles_job.keys():
        wind_profile_vad = vad_profiles_job[job_id]
        vad_height_idx = wind_profile_vad['height'] < max_height

        # Radar wind components.
        plt.scatter(wind_profile_vad['wind_U'][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                    color=blue_color_wheel[job_id], marker=description_jobs[job_id][1], alpha=0.35)
        plt.plot(wind_profile_vad['wind_U'][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                 color=blue_color_wheel[job_id], alpha=0.8, label=description_jobs[job_id][0] + " vad_U")
        plt.scatter(wind_profile_vad['wind_V'][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                    color=red_color_wheel[job_id], marker=description_jobs[job_id][1], alpha=0.35)
        plt.plot(wind_profile_vad['wind_V'][vad_height_idx], wind_profile_vad['height'][vad_height_idx],
                 color=red_color_wheel[job_id], alpha=0.8, label=description_jobs[job_id][0] + " vad_V")

    # Sounding wind components.
    sounding_height_idx = sounding_wind_df['HGHT'] < max_height
    plt.scatter(sounding_wind_df['windU'][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
                marker='o', color="blue", alpha=0.1)
    plt.scatter(sounding_wind_df['windV'][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
                color="red", alpha=0.1)
    plt.plot(sounding_wind_df['windU'][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
             label="wind_U", color="blue", linestyle='dashed')
    plt.plot(sounding_wind_df['windV'][sounding_height_idx], sounding_wind_df['HGHT'][sounding_height_idx],
             label="wind_V", color="red", linestyle='dashed')

    plt.ylim(0, 1.4 * max_height)
    plt.xlim(-16, 23)
    plt.grid(True)
    plt.xlabel("Wind components [mps]")
    plt.ylabel("Height [m]")
    plt.title(title_str + '\n' + prop_str)
    plt.legend(ncol=3)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_folder, "".join([figure_prefix, "_wind_comparison_components.png"])))
    # plt.show()
    plt.close()

    return


def classify_echoes(X, bi_clf_path, bi_norm_stats=None):
    # TODO load directly from bi_norm_stats
    norm_stats_bi = {
        'mean': pd.Series(data={'ZDR': 2.880010, 'pdp': 112.129741, 'RHV': 0.623049}, index=['ZDR', 'pdp', 'RHV']),
        'standard deviation': pd.Series(data={'ZDR': 2.936261, 'pdp': 52.774116, 'RHV': 0.201977},
                                        index=['ZDR', 'pdp', 'RHV'])}

    # Load bird-insect classifier.
    pin = open(bi_clf_path, 'rb')
    bi_clf = pickle.load(pin)
    pin.close()

    # Normalize X for bi.
    X_bi = X - norm_stats_bi['mean']
    X_bi = X.div(norm_stats_bi['standard deviation'], axis=1)
    X_bi = np.array(X_bi)
    y_bi = bi_clf['model'].predict(X_bi)

    return y_bi


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
    idx_height_interp = idx_height_interp.append(pd.Series([False for i in range(len(height_grid_interp))]), ignore_index=True)
    height_grid_interp = height_grid.append(height_grid_interp, ignore_index=True)
    windU_interp = windU.append(pd.Series(windU_interp), ignore_index=True)
    windV_interp = windV.append(pd.Series(windV_interp), ignore_index=True)
    df_interp = pd.DataFrame(
        {'HGHT': height_grid_interp, 'windU': windU_interp, 'windV': windV_interp})

    df_interp['TEMP'] = np.nan
    df_interp['TEMP'][idx_height_interp] = sounding_df['TEMP'][idx_height]
    df_interp['DRCT'] = np.nan
    df_interp['DRCT'][idx_height_interp] = sounding_df['DRCT'][idx_height]
    df_interp['SMPS'] = np.nan
    df_interp['SMPS'][idx_height_interp] = sounding_df['SMPS'][idx_height]
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

    df_interp['wind_speed'] = np.nan
    df_interp.loc[range(len(vad_df)), 'wind_speed'] = vad_df['wind_speed']
    df_interp['wind_direction'] = np.nan
    df_interp.loc[range(len(vad_df)), 'wind_direction'] = vad_df['wind_direction']
    df_interp['num_samples'] = np.nan
    df_interp.loc[range(len(vad_df)), 'num_samples'] = vad_df['num_samples']
    df_interp['mean_ref'] = np.nan
    df_interp.loc[range(len(vad_df)), 'mean_ref'] = vad_df['mean_ref']
    df_interp = df_interp.sort_values(by=['height'])

    return df_interp


def AnalyzeWind(radar_data_file, radar_data_folder, hca_data_folder, radar_t_sounding, station_infos, sounding_log_dir,
                norm_stats_file, clf_file, vad_jobs, figure_dir, max_range=300, max_height_VAD=1000,
                match_radar_and_sounding_grid=True, save_wind_figure=False, radar=None, hca_vol=None, data_table=None,
                l3_filelist=None):
    radar_data_file_no_ext = os.path.splitext(radar_data_file)[0]

    # Sounding.
    radar_name, year, month, day, hh, mm, ss = read_info_from_radar_name(radar_data_file)
    station_id = station_infos[radar_t_sounding[radar_name]][0]
    station_desc = station_infos[radar_t_sounding[radar_name]][1]
    sounding_url_base = "http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR={}&MONTH={}&FROM={}&TO={}&STNM={}"

    if not os.path.isdir(sounding_log_dir):
        os.makedirs(sounding_log_dir)

    # Read radar data.
    if radar is None:
        radar = pyart.io.read(os.path.join(radar_data_folder, radar_data_file))

    location_radar = {"latitude": radar.latitude['data'][0],
                      "longitude": radar.longitude['data'][0],
                      "height": radar.altitude['data'][0]}

    # Sounding wind profile.
    year_sounding, month_sounding, ddhh_sounding = GetSoundingDateTimeFromRadarFile(radar_data_file)
    sounding_wind_df, sounding_location, sounding_url = GetSoundingWind(sounding_url_base, radar_data_file,
                                                                        location_radar,
                                                                        station_id, sounding_log_dir,
                                                                        showDebugPlot=False, log_sounding_data=True,
                                                                        force_website_download=False)
    if sounding_wind_df is None:
        print("Sounding data is not available for this scan. Aborting wind analysis ...")
        return None, None, None

    distance_radar_sounding = GetHaverSineDistance(location_radar["latitude"], location_radar["longitude"],
                                                   sounding_location["latitude"],
                                                   sounding_location["longitude"])
    distance_radar_sounding = round(distance_radar_sounding / 1000, 2)

    # Read HCA data.
    if l3_filelist is None:
        radar_base = radar_data_file[:12]
        radar_data_folder = os.path.join(radar_data_folder, radar_base)
        hca_data_folder = os.path.join(hca_data_folder, radar_base)

    if hca_vol is None:
        if l3_filelist is None:
            hca_vol = GetHcaVol(hca_data_folder, radar_data_file_no_ext)
        else:
            hca_vol = GetHcaVolFromFileList(hca_data_folder, radar_data_file_no_ext, l3_filelist)

    # Data table.
    if data_table is None:
        data_table = MergeRadarAndHCAUpdate(radar, hca_vol, max_range)

    # TODO Precious. Get original ZDR mask.
    data_table["mask_differential_reflectivity"] = data_table["differential_reflectivity"] > -8.0
    data_table["hca_bio"] = data_table["hca"] == 10.0
    data_table["hca_weather"] = np.logical_and(data_table["hca"] >= 30.0, data_table["hca"] <= 100.0)
    data_table["height"] = data_table["range"] * np.sin(data_table["elevation"] * np.pi / 180)

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
    data_table['BIClass'] = -1
    data_table.loc[echo_mask, 'BIClass'] = classify_echoes(X, clf_file)

    # Visualize data table.
    color_map = GetDataTableColorMap()
    # VisualizeDataTable(data_table, color_map, figure_dir, scan_name = radar_data_file, title_suffix = "", combine_plots=True)

    # VAD wind profile
    signal_func = lambda x, t: x[0] * np.sin(2 * np.pi * (1 / 360) * t + x[1])
    vad_heights = np.arange(80, max_height_VAD, 40)
    # vad_heights = np.array([480])

    vad_profiles_job = {}
    for vad_mask in vad_jobs:
        wind_profile_vad = VADWindProfile(signal_func, vad_heights, vad_mask, data_table, showDebugPlot=False)
        vad_profiles_job[vad_mask] = wind_profile_vad

    # Interpolate wind components.
    max_height = 1.1 * max_height_VAD  # meters.
    max_height_diff = 250  # meters.
    height_grid_interp = wind_profile_vad['height']
    height_grid_interp = height_grid_interp[height_grid_interp < max_height]

    # Match, interpolate VAD and sounding grid
    if match_radar_and_sounding_grid:
        sounding_wind_df_interp = InterpolateSoundingWind(sounding_df=sounding_wind_df,
                                                          height_grid_interp=height_grid_interp,
                                                          max_height_diff=max_height_diff,
                                                          max_height=max_height)
        height_grid_interp = sounding_wind_df['HGHT']
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

    title_str = "{}, {}/{}/{}, {}:{}:{} UTC.\n{} km from {} UTC {} sounding.".format(
        radar_name, year, month,
        day, hh, mm, ss,
        distance_radar_sounding,
        ddhh_sounding[2:],
        station_desc)

    description_jobs = {VADMask.biological: ("bio", "."), VADMask.insects: ("ins", "2"),
                        VADMask.weather: ("wea", "d"), VADMask.birds: ("bir", "^")}
    figure_prefix = os.path.splitext(radar_data_file)[0]

    if match_radar_and_sounding_grid:
        VisualizeWinds(vad_profiles_job_interp, sounding_wind_df_interp, 1100, description_jobs, title_str, prop_str,
                       figure_dir, figure_prefix, save_wind_figure)
        return vad_profiles_job_interp, sounding_wind_df_interp, echo_dist_VAD

    VisualizeWinds(vad_profiles_job, sounding_wind_df, 1100, description_jobs, title_str, prop_str,
                   figure_dir, figure_prefix, save_wind_figure)
    return vad_profiles_job, sounding_wind_df, echo_dist_VAD


def Main():
    # Radar/Sounding.
    radar_data_file = "KOHX20180502_020640_V06"  # "KOHX20180506_231319_V06"
    radar_data_folder = "./radar_data"
    hca_data_folder = "./hca_data"
    radar_t_sounding = {'KHTX': 'BNA', 'KTLX': 'LMN', 'KOHX': 'BNA'}
    station_infos = {'LMN': ('74646', 'Lamont, Oklahoma'), 'BNA': ('72327', 'Nashville, Tennessee')}
    sounding_log_dir = "./sounding_logs"

    # Model.
    norm_stats_file = "./models/ridge_bi/mean_std_for_normalization_1.pkl"
    clf_file = "./models/ridge_bi/RidgeRegModels_SGD_1.pkl"

    # VAD options
    vad_jobs = [VADMask.birds, VADMask.insects, VADMask.weather]
    # vad_jobs = [VADMask.insects]

    figure_dir = os.path.join('./figures', radar_data_file[:12])

    vad_profiles, sounding_df, echo_dist = AnalyzeWind(radar_data_file, radar_data_folder, hca_data_folder,
                                                       radar_t_sounding, station_infos, sounding_log_dir,
                                                       norm_stats_file, clf_file, vad_jobs, figure_dir=figure_dir,
                                                       match_radar_and_sounding_grid=True,
                                                       save_wind_figure=False)

# Main()
