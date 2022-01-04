import pandas as pd
import numpy as np
import pickle

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
