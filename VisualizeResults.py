import random
import numpy as np
import pickle
import math
import os
from VADMaskEnum import VADMask
import matplotlib.pyplot as plt

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 11}
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
    plt.plot(time_hour, bird_count_scan, color='blue', label="bird")
    plt.plot(time_hour, insect_count_scan, color='red', label="insect")
    plt.plot(time_hour, weather_count_scan, color='green', label="weather")
    # plt.xticks(np.arange(1,24))
    plt.xlim(0, 24)
    if normalize_counts:
        plt.ylim(0, 1.2)
        plt.ylabel('Relative proportion')
    else:
        plt.ylabel("Echo count")
    plt.xlabel("UTC Time [hrs]")
    plt.title(output_figure_name)
    plt.grid(True)
    plt.legend()
    if save_fig:
        plt.savefig(os.path.join(output_figure_dir, output_figure_name_ext), dpi=200)
    # plt.show()
    plt.close()


# TODO
# monthly average
def Main():
    log_dir = './analysis_output_logs'
    file_base = 'KOHX201805{}_echo_count.pkl'
    start_day = 1
    stop_day = 15
    normalize_counts = True
    save_fig = True
    output_figure_dir = './figures/KOHX_20180501_20180515'

    if not os.path.isdir(output_figure_dir):
        os.makedirs(output_figure_dir)

    for day in range(start_day, stop_day+1):
        print("Visualizing day ", str(day),'....')
        VisualizeEchoDistributionForOneDay(day, file_base, log_dir, normalize_counts, save_fig, output_figure_dir)


Main()
