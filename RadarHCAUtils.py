import sys
import os
import numpy as np
import pandas as pd
import pyart
import matplotlib.pyplot as plt
from matplotlib import colors
from PreciousFunctions import PreciousCmap

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 15}
plt.rc('font', **font)


def read_info_from_radar_name(radar_file):
    """
    Finds the corresponding hca PPI name given a radar PPI name.
    :param radar_file:
    :return:
    """
    radar_name = radar_file[:4]
    year = radar_file[4:8]
    month = radar_file[8:10]
    day = radar_file[10:12]
    hh = radar_file[13:15]
    mm = radar_file[15:17]
    ss = radar_file[17:19]
    return radar_name, year, month, day, hh, mm, ss


def GetHcaPathFromFileList(hca_day_folder, radar_file, hca_el, el_desc_hca, filelist):
    suffix = "".join(["{}_", el_desc_hca[hca_el], "{}_{}{}{}{}"])

    sdus_id = next(iter(filelist))[:6]
    radar_name = radar_file[:4]
    year = radar_file[4:8]
    month = radar_file[8:10]
    day = radar_file[10:12]
    hhmm = radar_file[13:17]
    suffix = suffix.format(sdus_id, radar_name[1:], year, month, day, hhmm)

    if suffix in filelist.keys():
        target_folder = os.path.join(hca_day_folder, filelist[suffix])
        return target_folder
    return None  # file not found.

def GetHcaPath(hca_day_folder, radar_file, hca_el, el_desc_hca):
    suffix = "".join(["SDUS84_", el_desc_hca[hca_el], "{}_{}{}{}{}"])
    hca_subfolder = el_desc_hca[hca_el]

    radar_name = radar_file[:4]
    year = radar_file[4:8]
    month = radar_file[8:10]
    day = radar_file[10:12]
    hhmm = radar_file[13:17]
    suffix = suffix.format(radar_name[1:], year, month, day, hhmm)

    target_folder = os.path.join(hca_day_folder, hca_subfolder)
    for hca_file in os.listdir(target_folder):
        if hca_file.endswith(suffix):
            return os.path.join(target_folder, hca_file)
    return None  # file not found.

def GetHcaVolFromFileList(hca_day_folder, radar_file, filelist, el_desc_hca):
    # el_desc_hca = {0.5: "N0H", 1.5: "N1H", 2.5: "N2H", 3.5: "N3H"}
    volume_hca = {}
    for el_hca in el_desc_hca.keys():
        volume_hca[el_hca] = pyart.io.read_nexrad_level3(
            GetHcaPathFromFileList(hca_day_folder, radar_file, el_hca, el_desc_hca, filelist))
    return volume_hca


def GetHcaVol(hca_day_folder, radar_file, el_desc_hca):
    # el_desc_hca = {0.5: "N0H", 1.5: "N1H", 2.5: "N2H", 3.5: "N3H"}
    volume_hca = {}
    for el_hca in el_desc_hca.keys():
        volume_hca[el_hca] = pyart.io.read_nexrad_level3(GetHcaPath(hca_day_folder, radar_file, el_hca, el_desc_hca))
    return volume_hca


def ReadRadarSliceUpdate(radar, slice_idx):
    radar_range = radar.range['data'] / 1000  # in km
    sweep_ind = radar.get_slice(slice_idx)
    radar_az_deg = radar.azimuth['data'][sweep_ind]  # in degrees
    radar_el = radar.elevation['data'][sweep_ind]

    placeholder_matrix = np.empty(radar.fields["reflectivity"]['data'][sweep_ind].shape)
    placeholder_matrix[:] = np.nan
    radar_mask = placeholder_matrix

    data_slice = []
    labels_slice = list(radar.fields.keys())
    labels_slice.sort()
    mask_slice = []
    for radar_product in labels_slice:
        if np.sum(radar.fields[radar_product]['data'][sweep_ind].mask == False) > 0:
            data_slice.append(radar.fields[radar_product]['data'][sweep_ind].data)
            mask_slice.append(True)
            if radar_product == 'velocity':
                radar_mask = radar.fields[radar_product]['data'][sweep_ind].mask
        else:
            data_slice.append(placeholder_matrix)
            mask_slice.append(False)

    return radar_range, radar_az_deg, radar_el, data_slice.copy(), mask_slice.copy(), labels_slice, radar_mask


def MatchGates(arr, key_arr):
    """
    :param arr:
    :param key_arr:
    :return:
    """
    match_idxs = [np.nan for _ in range(len(key_arr))]
    for i in range(len(key_arr)):
        diff = list(np.abs(arr - key_arr[i]))
        match_idxs[i] = diff.index(min(diff))
    return match_idxs


def GetDataTableColorMapInfo():
    """
    Keys are variable names, Values is a tuple containing the associated color map, map range, and number of bins.
    :return:
    """
    return {'differential_phase': ('Theodore16', [0, 180], 10),
            'reflectivity': ('NWSRef', [-25, 75], 10),
            'differential_reflectivity': ('RefDiff', [-7.9, 7.9], 10),
            'cross_correlation_ratio': ('RRate11', [0.2, 1.05], 10),
            'velocity': ('NWSVel', [-37, 37], 10),
            'hca': ('viridis', [0, 150], 16),
            'hca_bio': ('viridis', [0, 1], 2),
            'hca_weather': ('Spectral', [0, 1], 2),
            'BIClass': ('custom', [0, 1], 3)}


def GetDataTableColorMap():
    cmap_info = GetDataTableColorMapInfo()
    cmap = {}
    pyart_products = {'differential_phase', 'reflectivity', 'differential_reflectivity', 'cross_correlation_ratio',
                      'velocity'}

    for key in cmap_info.keys():
        if key in pyart_products:
            cmap[key] = (pyart.graph.cm._generate_cmap(cmap_info[key][0], cmap_info[key][2]), cmap_info[key][1])
        elif key == 'BIClass':
            _, pCmap = PreciousCmap(256)
            cmap[key] = (colors.ListedColormap([pCmap[0], pCmap[-1]]), cmap_info[key][1])
        else:
            cmap[key] = (plt.cm.get_cmap(cmap_info[key][0], cmap_info[key][2]), cmap_info[key][1])
    return cmap


def VisualizeDataTable(data_table, color_map, output_folder, scan_name=None, title_suffix=None, combine_plots=True,
                       correct_hca_weather=False):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    az_bins = np.unique(data_table["azimuth"])
    range_bins = np.unique(data_table["range"])
    elevation_slices = np.unique(data_table["elevation"])

    num_az = len(az_bins)
    num_range = len(range_bins)

    az_grid = np.array(data_table["azimuth"][data_table["elevation"] == 0.5])
    az_grid = az_grid.reshape(num_az, num_range)
    range_grid = np.array(data_table["range"][data_table["elevation"] == 0.5])
    range_grid = range_grid.reshape(num_az, num_range)

    radar_name, year, month, day, hh, mm, ss = read_info_from_radar_name(scan_name)
    title_str = '{}, {}/{}/{}, {}:{}:{} UTC.'.format(radar_name, year, month, day, hh, mm, ss)
    title_bi = ''.join([title_str, '\n', title_suffix])

    products = list(color_map.keys())
    for product in products:
        out_product_name = "".join([product, '_corrected']) if (
                    product in {"hca_weather", "hca", "hca_bio", "BIClass"} and correct_hca_weather) else product
        print(out_product_name)

        product_folder = os.path.join(output_folder, out_product_name)
        if not os.path.isdir(product_folder):
            os.makedirs(product_folder)

        plot_folder = product_folder
        plot_scan_name = scan_name

        if combine_plots:
            if scan_name:
                output_path = os.path.join(plot_folder, ''.join([scan_name, '_', '.png']))
            else:
                output_path = plot_folder + "\\" + product + ".png"
        else:
            plot_folder = os.path.join(plot_folder, 'individual_cuts')
            if not os.path.isdir(plot_folder):
                os.makedirs(plot_folder)

            if scan_name:
                output_path_base = os.path.join(plot_folder, ''.join([scan_name, '_{}', '.png']))
            else:
                output_path_base = plot_folder + "\\" + product + "_{}.png"

        if combine_plots:
            fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
            count_subplot = 0
        # else:
        #     # TODO handle individual plots.
        #     sys.exit("VisualizeDataTable: No definition for individual plots.")

        for curr_el in elevation_slices:
            x = np.multiply(range_grid * np.cos(curr_el * np.pi / 180), np.sin(az_grid * np.pi / 180))
            y = np.multiply(range_grid * np.cos(curr_el * np.pi / 180), np.cos(az_grid * np.pi / 180))
            elev_mask = data_table["elevation"] == curr_el
            data_mask = data_table["mask_velocity"][elev_mask]  # TODO dp

            data_grid = data_table[product][elev_mask]
            data_grid[np.logical_not(data_mask)] = np.nan
            data_grid = np.array(data_grid).reshape(num_az, num_range)

            if combine_plots:
                im = ax[count_subplot // 2, count_subplot % 2].pcolor(x, y, data_grid, cmap=color_map[product][0],
                                                                      vmin=color_map[product][1][0],
                                                                      vmax=color_map[product][1][1])
                ax[count_subplot // 2, count_subplot % 2].set_title(str(curr_el) + "$^{\circ}$.")
                count_subplot += 1
            else:
                curr_el_str = str(curr_el)
                curr_el_str = curr_el_str.replace('.','p')
                output_path = output_path_base.format(curr_el_str)

                fig, ax = plt.subplots()
                cax = ax.pcolor(x, y, data_grid, cmap=color_map[product][0], vmin=color_map[product][1][0],
                           vmax=color_map[product][1][1])
                ax.set_xlim(-270, 270)
                ax.set_ylim(-270, 270)
                ax.set_xticks((-200, -100, 0, 100, 200))
                ax.set_yticks((-200, -100, 0, 100, 200))
                ax.set_xlabel('X [km]')
                ax.set_ylabel('Y [km]')
                ax.set_title(title_str + " \n Elevation: " + str(curr_el) + "$^{\circ}$")
                cbar = fig.colorbar(cax)
                plt.tight_layout()
                plt.savefig(output_path, dpi=200)
                # plt.show()
                plt.close()

        if combine_plots:
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel('X [km]')
            plt.ylabel('Y [km]')
            if product == "BIClass":
                fig.suptitle(title_bi)
            else:
                fig.suptitle(title_str)
            plt.tight_layout()
            fig.subplots_adjust(right=0.8)
            fig.subplots_adjust(top=0.84)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            plt.show()
            plt.savefig(output_path, dpi=200)
            plt.close(fig)
    return


# TODO mask_dp
def MergeRadarAndHCAUpdate(radar, hca_volume, maxRange):
    # Define common grid
    minRange = 10
    range_common = np.arange(minRange, maxRange, 0.25)
    az_common = np.arange(0, 360, 0.5)

    # Number of radar sweeps.
    n_sweeps = radar.nsweeps
    slice_store = dict(zip(hca_volume.keys(), [None for i in range(len(hca_volume))]))

    for radar_sweep_idx in range(n_sweeps):
        elevs_slice = radar.elevation['data'][radar.get_slice(radar_sweep_idx)]
        if elevs_slice.size == 0:
            print("Empty elevation slice. Skipping to next iteration")
            continue

        radar_el_default = np.mean(elevs_slice)
        radar_el = round(radar_el_default / 0.5) * 0.5

        # build radar + hca table if radar elevation matches hca elevation
        if radar_el in hca_volume.keys():
            radar_range, radar_az_deg, _, data_slice, mask_slice, labels_slice, radar_mask = ReadRadarSliceUpdate(radar,
                                                                                                                  radar_sweep_idx)

            # interpolate to common grid
            range_idxs = MatchGates(radar_range, range_common)
            az_idxs = MatchGates(radar_az_deg, az_common)

            for i in range(len(data_slice)):
                data_slice[i] = data_slice[i][np.ix_(az_idxs, range_idxs)]
                data_slice[i] = data_slice[i].reshape(-1, 1)
            radar_mask = radar_mask[np.ix_(az_idxs, range_idxs)]
            radar_mask = np.logical_not(radar_mask.reshape(-1, 1))

            # store slice.
            if slice_store[radar_el] == None:
                slice_store[radar_el] = [data_slice, mask_slice, radar_mask]
            else:  # update existing data slice
                print("slice already exists")
                for product_idx in range(len(labels_slice)):
                    if not slice_store[radar_el][1][product_idx]:
                        slice_store[radar_el][0][product_idx] = data_slice[product_idx]
                        slice_store[radar_el][1][product_idx] = mask_slice[product_idx]
                        if labels_slice[product_idx] == 'velocity':
                            slice_store[radar_el][2] = radar_mask

        else:  # invalid elevation for radar-hca fusion.
            print("invalid elevation. {} degrees".format(radar_el))

    # Merge tables from different elevations.
    data_tables = []
    for hca_el in hca_volume.keys():
        if slice_store[hca_el] == None:
            continue

        has_all_variables = np.logical_and.reduce(slice_store[hca_el][1])
        if not has_all_variables:
            continue

        # Define common grid
        range_common = np.arange(minRange, maxRange, 0.25)
        az_common = np.arange(0, 360, 0.5)

        # Get radar scan.
        data_table = slice_store[hca_el][0]
        columns = labels_slice.copy()

        data_table.append(slice_store[hca_el][2])
        columns.append("mask_velocity")

        # Read HCA.
        print("Reading HCA. el = {} degrees.".format(hca_el))
        hca = hca_volume[hca_el]
        hca_az_deg = hca.azimuth['data']  # in degrees
        hca_range = hca.range['data'] / 1000  # in km
        hca_mask = hca.fields['radar_echo_classification']['data'].mask
        hca_data = hca.fields['radar_echo_classification']['data'].data

        # Interpolate.
        range_idxs_hca = MatchGates(hca_range, range_common)
        az_idxs_hca = MatchGates(hca_az_deg, az_common)
        hca_data = hca_data[np.ix_(az_idxs_hca, range_idxs_hca)]
        hca_data = hca_data.reshape(-1, 1)
        hca_mask = hca_mask[np.ix_(az_idxs_hca, range_idxs_hca)]
        hca_mask = np.logical_not(hca_mask.reshape(-1, 1))

        # Add HCA to data table.
        data_table.append(hca_data)
        columns.append("hca")
        data_table.append(hca_mask)
        columns.append("hca_mask")

        # Azimuth, range and elevation.
        range_common, az_common = np.meshgrid(range_common, az_common)

        az_common = az_common.reshape(-1, 1)
        data_table.append(az_common)
        columns.append("azimuth")

        range_common = range_common.reshape(-1, 1)
        data_table.append(range_common)
        columns.append("range")

        data_table = np.concatenate(data_table, axis=1)
        data_table = pd.DataFrame(data_table, columns=columns)
        data_table["elevation"] = hca_el
        # print(data_table.shape)
        data_tables.append(data_table)

    # Merge tables
    if data_tables:
        data_table = pd.concat(data_tables, axis=0)
    else:
        data_table = pd.DataFrame(columns=['cross_correlation_ratio', 'differential_phase',
                                           'differential_reflectivity', 'reflectivity', 'spectrum_width',
                                           'velocity', 'mask_velocity', 'hca', 'hca_mask', 'azimuth', 'range',
                                           'elevation'])

    print(data_table.shape)
    return data_table


def ReadRadarCutAsTableUpdate(radar, slice_idx):
    """
    Assumes elevation is 0.5 degrees.
    :param radar:
    :param radar_products_slice:
    :param slice_idx:
    :return:
    """
    radar_range, radar_az_deg, radar_el, data_slice, labels_slice, radar_mask = ReadRadarSliceUpdate(radar, slice_idx)

    for i in range(len(data_slice)):
        data_slice[i] = data_slice[i].reshape(-1, 1)
    print("ReadRadarCutAsTableUpdate ", data_slice[0].shape)

    columns = labels_slice
    columns.extend(["mask", "range", "azimuth", "elevation"])
    data_slice.append(radar_mask.reshape(-1, 1))
    print(radar_mask.shape)
    print(radar_mask[3])

    # Range.
    range_grid, az_grid = np.meshgrid(radar_range, radar_az_deg)
    range_grid = range_grid.reshape(-1, 1)
    data_slice.append(range_grid)

    # Azimuth.
    az_grid = az_grid.reshape(-1, 1)
    data_slice.append(az_grid)

    # Elevation.
    _, el_grid = np.meshgrid(radar_range, radar_el)
    el_grid = el_grid.reshape(-1, 1)
    data_slice.append(el_grid)

    radar_data_table = np.concatenate(data_slice, axis=1)
    radar_data_table = pd.DataFrame(radar_data_table, columns=columns)

    return radar_data_table
