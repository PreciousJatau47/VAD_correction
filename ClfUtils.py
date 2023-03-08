import pandas as pd
import os
import numpy as np
import pickle


def classify_echoes(X, clf_path, norm_stats_path=None):
    # TODO(pjatau) Remove hardcoded values.
    # norm_stats = {
    #     'mean': pd.Series(data={'ZDR': 2.880010, 'pdp': 112.129741, 'RHV': 0.623049}, index=['ZDR', 'pdp', 'RHV']),
    #     'standard deviation': pd.Series(data={'ZDR': 2.936261, 'pdp': 52.774116, 'RHV': 0.201977},
    #                                     index=['ZDR', 'pdp', 'RHV'])}

    # Load norm-stats.
    pin = open(norm_stats_path, 'rb')
    norm_stats = pickle.load(pin)
    pin.close()

    # Load bird-insect classifier.
    pin = open(clf_path, 'rb')
    bi_clf = pickle.load(pin)
    pin.close()

    # Normalize X for bi.
    X_bi = X - norm_stats['mean']
    X_bi = X_bi.div(norm_stats['standard deviation'], axis=1)
    X_bi = np.array(X_bi)
    y_bi = bi_clf['model'].predict(X_bi)
    y_probs = bi_clf['model'].predict_proba(X_bi)

    return y_bi, y_probs


def Main():
    return


if __name__ == "__main__":
    Main()
