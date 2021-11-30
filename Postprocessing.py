import os
import pickle
import numpy as np
from VADMaskEnum import VADMask
import matplotlib.pyplot as plt
import pandas as pd

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 11}
plt.rc('font', **font)

"""
TODO. 
1. Define error (done), reduce function(done) and pass down WindError(). (done)
2. Error -> Absolute difference. Reduce -> Mean. 
3. GetAirSpeedsForScan
"""


# GetWindErrorBatch() -> GetWindErrorForScan() -> WindError().
def WindError(x1, y1, x2, y2, error_fn, reduce_fn):
    mapper = {}
    for j in range(len(x2)):
        mapper[x2[j]] = j

    x_scores = []
    scores = []
    for i in range(len(x1)):
        if x1[i] in mapper.keys():
            ind = mapper[x1[i]]
            error = error_fn(y1[i], y2[ind])
            scores.append(error)
            x_scores.append(x2[ind])

    reduced_error = reduce_fn(scores)
    return reduced_error, scores, x_scores


def GetWindErrorForScan(wind_file, wind_files_parent, error_fn, reduce_fn):
    wind_file_no_ext = os.path.splitext(wind_file)[0]
    with open(os.path.join(wind_files_parent, wind_file), 'rb') as w_in:
        wind_result = pickle.load(w_in)
    w_in.close()

    vad = wind_result['VAD']
    sounding_df = wind_result['Sounding']
    echo_dist = wind_result['echo_dist']

    if sounding_df is None:
        return None

    # Sounding.
    x_sound = np.array(sounding_df['HGHT'])
    y_sound_U = np.array(sounding_df['windU'])
    y_sound_V = np.array(sounding_df['windV'])

    # Birds.
    x_birds = np.array(vad[VADMask.birds]['height'])
    y_birds_U = np.array(vad[VADMask.birds]['wind_U'])
    y_birds_V = np.array(vad[VADMask.birds]['wind_V'])

    # Insects
    x_insects = np.array(vad[VADMask.insects]['height'])
    y_insects_U = np.array(vad[VADMask.insects]['wind_U'])
    y_insects_V = np.array(vad[VADMask.insects]['wind_V'])

    # Sounding v bird
    err_sound_birds_U, scores_sound_birds_U, x_sound_birds_U = WindError(x_sound, y_sound_U, x_birds, y_birds_U,
                                                                         error_fn, reduce_fn)
    err_sound_birds_V, scores_sound_birds_V, x_sound_birds_V = WindError(x_sound, y_sound_V, x_birds, y_birds_V,
                                                                         error_fn, reduce_fn)

    # Sounding v insects
    err_sound_insects_U, scores_sound_insects_U, x_sound_insects_U = WindError(x_sound, y_sound_U, x_insects,
                                                                               y_insects_U, error_fn, reduce_fn)
    err_sound_insects_V, scores_sound_insects_V, x_sound_insects_V = WindError(x_sound, y_sound_V, x_insects,
                                                                               y_insects_V, error_fn, reduce_fn)

    # Birds v insects.
    err_birds_insects_U, scores_birds_insects_U, x_birds_insects_U = WindError(x_birds, y_birds_U, x_insects,
                                                                               y_insects_U, error_fn, reduce_fn)
    err_birds_insects_V, scores_birds_insects_V, x_birds_insects_V = WindError(x_birds, y_birds_V, x_insects,
                                                                               y_insects_V, error_fn, reduce_fn)

    return {'file_name': wind_file_no_ext, 'err_sounding_birds_U': err_sound_birds_U,
            'err_sounding_birds_V': err_sound_birds_V,
            'err_sounding_insects_U': err_sound_insects_U, 'err_sounding_insects_V': err_sound_insects_V,
            'err_birds_insects_U': err_birds_insects_U, 'err_birds_insects_V': err_birds_insects_V,
            'prop_birds': echo_dist['bird'], 'prop_insects': echo_dist['insects'],
            'prop_weather': echo_dist['weather']}


def GetWindErrorBatch(wind_files_parent, error_fn, reduce_fn):
    wind_files = os.listdir(wind_files_parent)
    wind_error_df = pd.DataFrame(columns=['file_name', 'err_sounding_birds_U',
                                          'err_sounding_birds_V',
                                          'err_sounding_insects_U', 'err_sounding_insects_V',
                                          'err_birds_insects_U', 'err_birds_insects_V',
                                          'prop_birds', 'prop_insects',
                                          'prop_weather'])

    # Get wind here.
    for wind_file in wind_files:
        entry = GetWindErrorForScan(wind_file, wind_files_parent, error_fn, reduce_fn)
        if entry is not None:
            wind_error_df = wind_error_df.append(entry, ignore_index=True)
    return wind_error_df


def Main():
    wind_dir = './vad_sounding_comparison_logs'
    batch_folder_1 = "KOHX_20180501_20180515"
    batch_folder_2 = "KOHX_20180516_20180531"
    error_fn = lambda yTrue, yPred: (yPred - yTrue) ** 2
    reduce_fn = lambda scores: np.sqrt(np.nanmean(scores))

    wind_error_1 = GetWindErrorBatch(os.path.join(wind_dir, batch_folder_1), error_fn, reduce_fn)
    wind_error_2 = GetWindErrorBatch(os.path.join(wind_dir, batch_folder_2), error_fn, reduce_fn)
    wind_error = pd.concat([wind_error_1, wind_error_2], ignore_index=True)

    bird_ratio_bio = wind_error['prop_birds']
    insects_ratio_bio = wind_error['prop_insects']
    total = bird_ratio_bio + insects_ratio_bio
    bird_ratio_bio = np.divide(bird_ratio_bio, total)
    insects_ratio_bio = np.divide(insects_ratio_bio, total)
    wind_error['insect_prop_bio'] = insects_ratio_bio * 100
    wind_error = wind_error.sort_values(by=['insect_prop_bio'])

    MAX_WEATHER_PROP = 10
    idx_low_weather = wind_error['prop_weather'] < MAX_WEATHER_PROP

    c = wind_error['insect_prop_bio'][idx_low_weather]
    c = (c - min(c)) / (max(c) - min(c))
    s = (c * 100) + 10

    # Sounding, bird/insect U.
    plt.figure()
    plt.plot([0, 10], [0, 10], linestyle='dashed', alpha=0.6)
    plt.scatter(x=wind_error['err_sounding_insects_U'][idx_low_weather],
                y=wind_error['err_sounding_birds_U'][idx_low_weather], s=s, c=c, alpha=0.3, cmap='jet')

    plt.xlim(0, 8)
    plt.ylim(0, 8)
    plt.grid(True)
    plt.colorbar()
    plt.xlabel(r"$RMSE (sounding_U, insects_U)$")
    plt.ylabel(r"$RMSE (sounding_U, birds_U)$")
    plt.title("Comparison of U component for Sounding and VAD winds.")
    # plt.savefig("sounding_bio_U.png", dpi = 200)

    # Sounding, bird/insect V.
    plt.figure()
    plt.plot([0, 10], [0, 10], linestyle='dashed', alpha=0.6)
    plt.scatter(x=wind_error['err_sounding_insects_V'][idx_low_weather],
                y=wind_error['err_sounding_birds_V'][idx_low_weather], s=s, c=c, alpha=0.3, cmap='jet')
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    plt.grid(True)
    plt.colorbar()
    plt.xlabel(r"$RMSE (sounding_V, insects_V)$")
    plt.ylabel(r"$RMSE (sounding_V, birds_V)$")
    plt.title("Comparison of V component for Sounding and VAD winds.")
    # plt.savefig("sounding_bio_V.png", dpi = 200)

    plt.show()


Main()
