import random
import numpy as np
import pickle
import math
import os
from VADMaskEnum import VADMask
import matplotlib.pyplot as plt

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 12}
plt.rc('font', **font)


def GetTimeHourUTC(some_str: str) -> float:
    idx_timestart = some_str.find('_') + 1
    hh = some_str[idx_timestart:idx_timestart + 2]
    mm = some_str[idx_timestart + 2:idx_timestart + 4]
    ss = some_str[idx_timestart + 4:idx_timestart + 6]
    time_hour = float(hh) + float(mm) / 60 + float(ss) / 3600
    return time_hour


def GetTimeHourLocal(some_str: str, utc_t_local: int) -> float:
    return GetTimeHourUTC(some_str) + utc_t_local


def GetMonthNumberToName():
    return {'01': 'January',
            '02': 'February',
            '03': 'March',
            '04': 'April',
            '05': 'May',
            '06': 'June',
            '07': 'July',
            '08': 'August',
            '09': 'September',
            '10': 'October',
            '11': 'November',
            '12': 'December'}


def VisualizeEchoDistributionForOneDay(day, file_base, log_dir, normalize_counts, save_fig, output_figure_dir):
    day = '0' + str(day) if day < 10 else str(day)
    log_file = file_base.format(day)
    output_figure_name = log_file.split('_')[0]
    output_figure_name_ext = ''.join([output_figure_name, '.png'])

    # load results from analysis. Result is a tuple, (filelist, echo distribution dictionary)
    log_file_path = os.path.join(log_dir, log_file)
    with open(log_file_path, "rb") as p_in:
        result = pickle.load(p_in)
    p_in.close()

    time_hour = [GetTimeHourUTC(some_str) for some_str in result[0]]

    # Convert counts to fraction
    bird_count_scan = np.array(result[1][VADMask.birds])
    insect_count_scan = np.array(result[1][VADMask.insects])
    weather_count_scan = np.array(result[1][VADMask.weather])

    if normalize_counts:
        total_count_scan = bird_count_scan + insect_count_scan + weather_count_scan
        bird_count_scan = np.divide(bird_count_scan, total_count_scan)
        insect_count_scan = np.divide(insect_count_scan, total_count_scan)
        weather_count_scan = np.divide(weather_count_scan, total_count_scan)

    plt.figure()
    plt.plot(time_hour, weather_count_scan, color='green', label="weather", linestyle='dashed')
    plt.plot(time_hour, insect_count_scan, color='red', label="insects")
    plt.plot(time_hour, bird_count_scan, color='blue', label="birds")
    # plt.xticks(np.arange(1,24))
    plt.xlim(0, 24)
    if normalize_counts:
        plt.ylim(0, 1.2)
        plt.ylabel('Relative proportion [no unit]')
    else:
        plt.ylabel("Echo count [no unit]")
    plt.xlabel("UTC Time [hrs]")
    plt.title(output_figure_name)
    plt.grid(True)
    plt.legend()
    if save_fig:
        plt.savefig(os.path.join(output_figure_dir, output_figure_name_ext), dpi=200)
    # plt.show()
    plt.close()


def VisualizeEchoDistributionForMultipleDays(start_day, stop_day, file_base, log_dir, normalize_counts, save_fig,
                                             output_figure_dir):
    for day in range(start_day, stop_day + 1):
        print("Visualizing day ", str(day), '....')
        VisualizeEchoDistributionForOneDay(day, file_base, log_dir, normalize_counts, save_fig, output_figure_dir)


def AccumulateResults(start_day, stop_day, file_base, log_dir, normalize_counts):
    # Define analysis grid (day * time hour)
    time_hour_step = 0.5
    time_hour_grid = np.arange(0, 24 + time_hour_step, time_hour_step)
    day_grid = list(range(start_day, stop_day + 1))

    # Initialize echo counts for batch analysis
    echo_counts_batch = {}
    echo_counts_batch[VADMask.insects] = np.empty((len(day_grid), len(time_hour_grid)))
    echo_counts_batch[VADMask.insects][:] = np.nan
    echo_counts_batch[VADMask.birds] = np.empty((len(day_grid), len(time_hour_grid)))
    echo_counts_batch[VADMask.birds][:] = np.nan
    echo_counts_batch[VADMask.weather] = np.empty((len(day_grid), len(time_hour_grid)))
    echo_counts_batch[VADMask.weather][:] = np.nan

    # Invalidate 0 bucket. All time in the range (0,0.499) are mapped to the 0.5 bucket.
    echo_counts_batch[VADMask.insects][:, 0] = np.nan
    echo_counts_batch[VADMask.birds][:, 0] = np.nan
    echo_counts_batch[VADMask.weather][:, 0] = np.nan

    for i in range(len(day_grid)):
        # Load result.
        day = day_grid[i]
        day = '0' + str(day) if day < 10 else str(day)
        log_file = file_base.format(day)

        log_file_path = os.path.join(log_dir, log_file)
        with open(log_file_path, "rb") as p_in:
            result = pickle.load(p_in)
        p_in.close()

        # Get timestamps.
        time_hour = [GetTimeHourUTC(some_str) for some_str in result[0]]

        # Accumulate results for time period defined by day grid.
        prev_idx = -1
        idx_count = 0
        for j in range(len(time_hour)):

            # Skip nan entries.
            if math.isnan(result[1][VADMask.insects][j]):
                continue

            idx = int(math.ceil(time_hour[j] / 0.5))

            if prev_idx == -1 or idx == prev_idx:
                idx_count += 1
            else:   # New bucket encountered.
                # Calculate average for the past 0.5 hour.
                echo_counts_batch[VADMask.insects][i][prev_idx] /= idx_count
                echo_counts_batch[VADMask.birds][i][prev_idx] /= idx_count
                echo_counts_batch[VADMask.weather][i][prev_idx] /= idx_count
                idx_count = 1

            echo_total = result[1][VADMask.insects][j] + result[1][VADMask.birds][j] + result[1][VADMask.weather][j] if normalize_counts else 1

            if math.isnan(echo_counts_batch[VADMask.insects][i][idx]):  # Bucket encountered for the first time.
                echo_counts_batch[VADMask.insects][i][idx] = result[1][VADMask.insects][j] / echo_total
                echo_counts_batch[VADMask.birds][i][idx] = result[1][VADMask.birds][j] / echo_total
                echo_counts_batch[VADMask.weather][i][idx] = result[1][VADMask.weather][j] / echo_total
            else:
                echo_counts_batch[VADMask.insects][i][idx] = echo_counts_batch[VADMask.insects][i][idx] + \
                                                             result[1][VADMask.insects][j] / echo_total
                echo_counts_batch[VADMask.birds][i][idx] = echo_counts_batch[VADMask.birds][i][idx] + \
                                                           result[1][VADMask.birds][j] / echo_total
                echo_counts_batch[VADMask.weather][i][idx] = echo_counts_batch[VADMask.weather][i][idx] + \
                                                             result[1][VADMask.weather][j] / echo_total
            prev_idx = idx

        # Calculate average for last 0.5 hour.
        echo_counts_batch[VADMask.insects][i][prev_idx] /= idx_count
        echo_counts_batch[VADMask.birds][i][prev_idx] /= idx_count
        echo_counts_batch[VADMask.weather][i][prev_idx] /= idx_count

    return echo_counts_batch, day_grid, time_hour_grid


def Main():
    batch_folder = 'KOHX_20180501_20180515'
    log_dir = './analysis_output_logs'
    radar_name = 'KOHX'
    file_base = 'KOHX{}{}{}_echo_count.pkl'
    month = 5
    year = 2018
    start_day = 1
    stop_day = 15
    normalize_counts = True
    save_fig = False
    output_figure_dir = './figures'

    log_dir = os.path.join(log_dir, batch_folder)
    output_figure_dir = os.path.join(output_figure_dir, batch_folder, 'summary')

    month_num_t_name = GetMonthNumberToName()

    month_str = ''.join(['0', str(month)]) if month < 10 else str(month)
    file_base = ''.join([radar_name, str(year), month_str, '{}_echo_count.pkl'])
    plot_title_str = ''.join(
        ['Average for ', month_num_t_name[month_str], ', ', str(start_day), ' - ', str(stop_day), ', ', str(year), '.'])

    if not os.path.isdir(output_figure_dir):
        os.makedirs(output_figure_dir)

    VisualizeEchoDistributionForMultipleDays(start_day, stop_day, file_base, log_dir, normalize_counts, save_fig,
                                             output_figure_dir)

    echo_counts_batch, day_grid, time_hour_grid = AccumulateResults(start_day, stop_day, file_base, log_dir,
                                                                    normalize_counts)

    bird_stats_reduced = np.nanmean(echo_counts_batch[VADMask.birds], axis=0)
    insect_stats_reduced = np.nanmean(echo_counts_batch[VADMask.insects], axis=0)
    weather_stats_reduced = np.nanmean(echo_counts_batch[VADMask.weather], axis=0)

    plt.figure()

    # Plots for each day.
    for i in range(len(day_grid)):
        plt.plot(time_hour_grid, echo_counts_batch[VADMask.weather][i], color='green', linestyle='dashed', alpha=0.15)
        plt.plot(time_hour_grid, echo_counts_batch[VADMask.birds][i], color='blue', alpha=0.15)
        plt.plot(time_hour_grid, echo_counts_batch[VADMask.insects][i], color='red', alpha=0.15)

    # Averages
    plt.plot(time_hour_grid, weather_stats_reduced, color='green', label='weather', linestyle='dashed', linewidth=2.5)
    plt.plot(time_hour_grid, bird_stats_reduced, color='blue', label='birds', linewidth=2.5)
    plt.plot(time_hour_grid, insect_stats_reduced, color='red', label='insects', linewidth=2.5)

    plt.xlim(0, 24)
    if normalize_counts:
        plt.ylim(0, 1.2)
    plt.xlabel('UTC Time [hrs]')
    plt.ylabel('Relative proportion [no unit]')
    plt.title(plot_title_str)
    plt.grid(True)
    plt.legend(loc="upper right")
    if save_fig:
        plt.savefig(os.path.join(output_figure_dir, 'Average.png'), dpi=200)
    plt.show()


Main()
