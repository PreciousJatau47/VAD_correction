import numpy as np
from matplotlib.colors import ListedColormap


"""
This function creates a custom colormap with blue for high values, green for intermediate values and red for low values
inputs: nPts
outputs: myMap
"""
def PreciousCmap(nPts):
    # initialize map array
    myMap = np.zeros((nPts, 4))

    # red channel
    myMap[range(nPts), 0] = np.linspace(1, 0.3, nPts)

    # green channel
    mu = .5
    sig = .2
    x = np.linspace(0, 1, nPts)
    myMap[range(nPts), 1] = (1 / np.sqrt(2 * np.pi * sig ** 2)) * np.exp(-((x - mu) ** 2) / (2 * sig ** 2))
    myMap[range(nPts), 1] = 0.99 * myMap[range(nPts), 1] / max(myMap[range(nPts), 1])

    # blue channel
    myMap[range(nPts), 2] = np.linspace(.3, 1, nPts)

    # alpha channel
    myMap[range(nPts), 3] = 1

    if nPts <= 3:
        myMap = myMap[[0,nPts-1],:]


    # convert to listed color map object
    myMapLC = ListedColormap(myMap)

    return myMapLC, myMap

""" 
This function extracts the cartesian coordinates for a radar matrix 
inputs: elevation, azBins, rangeGates
outputs: x, y, z
"""
def cartesian_coord_for_radar_ppi(elevation, azBins, rangeGates):
    # convert elevation and azimuth to rads
    elevation = elevation / 180 * np.pi
    azBins = azBins / 180 * np.pi

    # convert radar matrix to be in cartesian coordinates
    rangeGrid, azGrid = np.meshgrid(rangeGates, azBins)
    x = np.multiply(rangeGrid * np.cos(elevation), np.sin(azGrid))
    y = np.multiply(rangeGrid * np.cos(elevation), np.cos(azGrid))
    z = rangeGrid * np.sin(elevation)

    return x, y, z


# convert to radar format
# inputs: nGates, nAzim, delR, inData
# output: radarMatrix, elevation, ppiTime
def df_to_radar_matrix(nGates, nAzim, delR, inData, outputType):
    elevation = inData['elevation'][0]
    if len(np.unique(inData['elevation'])) > 1:
        print('CHECK! More than 2 unique elevations in the current PPI.')

    ppiTime = inData['time'][0]
    if len(np.unique(inData['time'])) > 1:
        print('CHECK! More than 2 unique times in the current PPI.')

    # create range, azimuth bins
    rangeGates = np.arange(0, nGates, 1) * delR

    azBins = np.linspace(min(inData['az']), max(inData['az']), nAzim)

    # create matrix in radar format
    radarMatrix = np.full((nGates, nAzim), np.nan)

    # fill in matrix
    for ind in range(inData.shape[0]):
        indRange = rangeGates == inData.loc[ind, 'range']
        indAz = np.argmin(np.absolute(azBins - inData.loc[ind, 'az']))
        radarMatrix[indRange, indAz] = inData.loc[ind, outputType]

    return radarMatrix, rangeGates, azBins, elevation, ppiTime

"""
radar_data is a dictionary containing radar range, azimuth and data matrix.
hca_map contains similar information for a hca map
"""
def match_with_hca(radar_data, hca_map):

    range_gates_hca = hca_map['range gates']
    az_hca = hca_map['az']

    range_gates_radar = radar_data['range gates']
    az_radar = radar_data['az']

    range_idx = [None] * len(range_gates_radar)
    az_idx = [None] * len(az_radar)

    for ir in range(len(range_idx)):
        range_idx[ir] = find_closest_element_position(range_gates_radar[ir],range_gates_hca)

    for iaz in range(len(az_idx)):
        az_idx[iaz] = find_closest_element_position(az_radar[iaz],az_hca)

    return range_idx, az_idx


def find_closest_element_position(key, arr):
    diff = [np.abs(arr[i] - key) for i in range(len(arr))]
    return diff.index(min(diff))

# dilate binary image using a 3x3 window
def dilateimage(im):
    nrows, ncols = im.shape
    result = np.copy(im)

    for row in range(1,nrows-1):
        for col in range(1,ncols-1):

            # dilate
            targetfound = im[row][col]
            if not targetfound:
                i = -1
                j = -1
                while not targetfound and j<=1:
                    targetfound = targetfound or im[row+i][col+j]
                    #print(im[row+i][col+j], targetfound)

                    if i ==1:
                        j += 1
                        i = -1
                    else:
                        i += 1
                result[row, col] = targetfound


    return result




