import unittest
from Simulations import *


class TestSimulations(unittest.TestCase):
    def test_SearchAlphaFlightAndMagMig(self):
        # Test 1.
        # vr = (4,0), vw = (3,90), vf = (5, 324]. Result 324, 4
        alpha_mig_true = 0
        spd_flight = 5
        spd_wind = 3
        alpha_wind = 90

        exp_alpha_flight = 324
        exp_mag_mig = 4

        pred_alpha_flight, pred_mag_mig = SearchAlphaFlightAndMagMig(alpha_mig_true=alpha_mig_true,
                                                                     spd_flight=spd_flight,
                                                                     spd_wind=spd_wind, alpha_wind=alpha_wind)
        self.assertTrue(exp_alpha_flight - pred_alpha_flight < 2)
        self.assertTrue(exp_mag_mig - pred_mag_mig < 2)

        # Test 2.
        # vr = (5,143.13), vw = (3,90), vf = (4, 180). Result 180, 4
        alpha_mig_true = 143.13
        spd_flight = 4
        spd_wind = 3
        alpha_wind = 90

        exp_alpha_flight = 180
        exp_mag_mig = 5

        pred_alpha_flight, pred_mag_mig = SearchAlphaFlightAndMagMig(alpha_mig_true=alpha_mig_true,
                                                                     spd_flight=spd_flight,
                                                                     spd_wind=spd_wind, alpha_wind=alpha_wind)
        self.assertTrue(exp_alpha_flight - pred_alpha_flight < 2)
        self.assertTrue(exp_mag_mig - pred_mag_mig < 2)

        # Test 3.
        # vr = (5,233.13), vw = (3,180), vf = (4, 270). Result 270, 5
        alpha_mig_true = 233.13
        spd_flight = 4
        spd_wind = 3
        alpha_wind = 180

        exp_alpha_flight = 270
        exp_mag_mig = 5

        pred_alpha_flight, pred_mag_mig = SearchAlphaFlightAndMagMig(alpha_mig_true=alpha_mig_true,
                                                                     spd_flight=spd_flight,
                                                                     spd_wind=spd_wind, alpha_wind=alpha_wind)
        self.assertTrue(exp_alpha_flight - pred_alpha_flight < 2)
        self.assertTrue(exp_mag_mig - pred_mag_mig < 2)

    def test_MixVADWinds(self):
        x = np.array([0, 1, 2, 3])
        y_pos = np.array([0, 2, 4, 6])
        y_neg = np.array([1, 3, 5, 7])

        num_samples = len(x)
        neg_prop = 0.2
        exp_num_neg = int(neg_prop * num_samples)
        exp_num_pos = num_samples - exp_num_neg
        mixture, is_pos, idx_pos, idx_neg = MixVADWinds(az_grid=x, vad_birds=y_pos, vad_insects=y_neg,
                                                        insect_ratio_bio=neg_prop,
                                                        show_plot=False)
        self.assertTrue(len(idx_pos) == exp_num_pos)
        self.assertTrue(len(idx_neg) == exp_num_neg)
        self.assertTrue(np.array_equal(np.sort(np.where(is_pos))[0], np.sort(idx_pos)))
        self.assertTrue(np.array_equal(np.sort(np.where(is_pos == False))[0], np.sort(idx_neg)))
        self.assertTrue(np.array_equal(mixture[is_pos], y_pos[is_pos]))
        self.assertTrue(np.array_equal(mixture[is_pos == False], y_neg[is_pos == False]))

    def test_SimulateBIPredictions(self):
        tpr, tnr = 0.80, 0.90
        idx_pos = [i for i in range(10)]
        idx_neg = [i for i in range(10, 20)]
        num_samples = len(idx_pos) + len(idx_neg)
        is_bird = np.zeros((num_samples,), dtype=bool)
        is_bird[idx_pos] = True
        is_birds_pred = SimulateBIPredictions(tpr=tpr, tnr=tnr, idx_birds=idx_pos, idx_insects=idx_neg)

        num_pos = np.sum(is_bird)
        num_neg = num_samples - num_pos
        tpn = np.sum(np.logical_and(is_bird, is_birds_pred))
        tpr_pred = tpn / num_pos
        tnn = np.sum(np.logical_and(is_bird == False, is_birds_pred == False))
        tnr_pred = tnn / num_neg

        self.assertAlmostEqual(first=tpr, second=tpr_pred, places=1, msg="Expected and predicted tpr are not equal.")
        self.assertAlmostEqual(first=tnr, second=tnr_pred, places=1, msg="Expected and predicted tnr are not equal.")


if __name__ == '__main__':
    unittest.main()
