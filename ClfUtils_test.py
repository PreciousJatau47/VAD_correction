import unittest
import pickle
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class TestClifUtils(unittest.TestCase):
    def test_crosscheck_model_predictions(self):
        norm_stats_file = "./models/ridge_bi/mean_std_for_normalization_2.pkl"
        clf_file = "./models/ridge_bi/RidgeRegModels_SGD_1.pkl"

        # Load normalization statistics
        pin = open(norm_stats_file, 'rb')
        norm_stats = pickle.load(pin)
        pin.close()

        # Load bird-insect classifier.
        pin = open(clf_file, 'rb')
        bi_clf = pickle.load(pin)
        pin.close()

        # Weights and biases.
        coefs = dict(zip(['ZDR', 'pdp', 'RHV'], bi_clf['model'].coef_[0]))
        intercept = bi_clf['model'].intercept_[0]

        meas_dict = {'ZDR': [0, 1, 2, 0.5], 'pdp': [0, 60, 100, 240], 'RHV': [0, 0.5, 0.99, 0.7]}
        X = pd.DataFrame(meas_dict)

        # Expected scores and probs.
        # score = W*(x - mu)/sdev + b.
        # prob = sigmoid(score)
        exp_score = intercept
        for var in meas_dict:
            exp_score += coefs[var] * (X[var] - norm_stats['mean'][var]) / norm_stats['standard deviation'][var]
        exp_score = np.array(exp_score)
        exp_probs = sigmoid(exp_score)

        # Predicted scores and probs.
        X = X - norm_stats['mean']
        X = X.div(norm_stats['standard deviation'], axis=1)
        pred_score = bi_clf['model'].decision_function(X)
        pred_probas = bi_clf['model'].predict_proba(X)[:, 1]

        assert np.logical_and.reduce(
            np.isclose(a=exp_score, b=pred_score, rtol=0.05,
                       atol=0.05)), "predicted scores differ from expected scores"
        assert np.logical_and.reduce(
            np.isclose(a=exp_probs, b=pred_probas, rtol=0.05,
                       atol=0.05)), "predicted scores differ from expected probabilities"
        return