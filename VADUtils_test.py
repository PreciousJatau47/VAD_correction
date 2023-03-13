import unittest
from VADUtils import *
import GeneralUtils as gu


class TestVADUtils(unittest.TestCase):
    def test_VADMask(self):
        # load data table
        p_in = open('./test_files/data_table.pkl', 'rb')
        data_table = pickle.load(p_in)
        p_in.close()

        # biological echoes.
        exp_bio_mask = gu.logical_and(data_table['mask_velocity'], data_table['hca_bio'])
        pred_bio_mask = GetVADMask(data_table, VADMask.biological)
        assert np.logical_and.reduce(exp_bio_mask == pred_bio_mask)

        # weather echoes.
        exp_wea_mask = gu.logical_and(data_table['mask_velocity'], data_table['hca_weather'])
        pred_wea_mask = GetVADMask(data_table, VADMask.weather)
        assert np.logical_and.reduce(exp_wea_mask == pred_wea_mask)

        # tail threshold = 0.5
        tail_threshold = 0.5
        exp_bird_mask = gu.logical_and(data_table['mask_velocity'], data_table['mask_differential_reflectivity'],
                                       data_table['hca_bio'], data_table['BIClass'] == 1)
        pred_bird_mask = GetVADMask(data_table, VADMask.birds, tail_threshold)
        print("pred. birds: ", np.sum(pred_bird_mask))
        assert np.logical_and.reduce(exp_bird_mask == pred_bird_mask)

        exp_ins_mask = gu.logical_and(data_table['mask_velocity'], data_table['mask_differential_reflectivity'],
                                      data_table["hca_bio"], data_table['BIClass'] == 0)
        pred_ins_mask = GetVADMask(data_table, VADMask.insects, tail_threshold)
        print("pred. insects: ", np.sum(pred_ins_mask))
        assert np.logical_and.reduce(exp_ins_mask == pred_ins_mask)

        # tail threshold = 0.8
        tail_threshold = 0.2
        exp_bird_mask = gu.logical_and(data_table['mask_velocity'], data_table['mask_differential_reflectivity'],
                                       data_table['hca_bio'], data_table['BIProb'] >= (1 - tail_threshold))
        pred_bird_mask = GetVADMask(data_table, VADMask.birds, tail_threshold)
        print("pred. birds: ", np.sum(pred_bird_mask))
        assert np.logical_and.reduce(exp_bird_mask == pred_bird_mask)

        exp_ins_mask = gu.logical_and(data_table['mask_velocity'], data_table['mask_differential_reflectivity'],
                                      data_table["hca_bio"], data_table['BIProb'] < tail_threshold)
        pred_ins_mask = GetVADMask(data_table, VADMask.insects, tail_threshold)
        print("pred. insects: ", np.sum(pred_ins_mask))
        assert np.logical_and.reduce(exp_ins_mask == pred_ins_mask)

    def test_MixVADs(self):
        N = 720 * 2
        t = np.linspace(0, 360, N)

        # Generate VADs
        true_wind_speed = 15
        true_wind_dir = 90
        vad_main = GenerateVAD(speed=true_wind_speed, dirn=true_wind_dir, dirn_grid=t) + GenerateNoiseNormal(
            num_samples=N)
        vad_sec = GenerateVAD(speed=11, dirn=68, dirn_grid=t) + GenerateNoiseNormal(num_samples=N)

        # MixVADs
        vad_mix, is_main_vad, a_idxs, mix_idxs, b_idxs, weights_mix = MixVADs(vad_main=vad_main, vad_sec=vad_sec,
                                                                              a=1 / 20, mix_r=9 / 10)

        self.assertTrue(np.array_equal(vad_mix[a_idxs], vad_main[a_idxs]))
        self.assertTrue(np.array_equal(vad_mix[b_idxs], vad_sec[b_idxs]))
        exp_mix = vad_mix[mix_idxs]
        act_mix = weights_mix * vad_main[mix_idxs] + (1 - weights_mix) * vad_sec[mix_idxs]
        self.assertTrue(np.array_equal(exp_mix, act_mix))

        self.assertTrue(np.array_equal(is_main_vad[mix_idxs], weights_mix))
        self.assertTrue(np.logical_and.reduce(is_main_vad[a_idxs] == 1))
        self.assertTrue(np.logical_and.reduce(is_main_vad[b_idxs] == 0))

        return

    def test_fitVAD(self):
        N = 720 * 2
        t = np.linspace(0, 360, N)

        # Unweighted VAD
        true_wind_speed = 11
        true_wind_dir = 60
        data = GenerateVAD(speed=true_wind_speed, dirn=true_wind_dir, dirn_grid=t) + GenerateNoiseNormal(num_samples=N)
        wind_speed, wind_dir, vad_fit = fitVAD(t, data, signal_func, showDebugPlot=False, description='')
        self.assertTrue(np.isclose(a=true_wind_speed, b=wind_speed, rtol=1, atol=1))
        self.assertTrue(np.isclose(a=true_wind_dir, b=wind_dir, rtol=5, atol=5))

        # Weighted VAD
        true_wind_speed = 15
        true_wind_dir = 90
        vad_main = GenerateVAD(speed=true_wind_speed, dirn=true_wind_dir, dirn_grid=t) + GenerateNoiseNormal(
            num_samples=N)
        vad_sec = GenerateVAD(speed=11, dirn=68, dirn_grid=t) + GenerateNoiseNormal(num_samples=N)


        vad_mix, is_main_vad, a_idxs, mix_idxs, b_idxs, weights_mix = MixVADs(vad_main=vad_main, vad_sec=vad_sec,
                                                                              a=1 / 20, mix_r=9 / 10)

        wind_speed, wind_dir, vad_fit = fitVAD(t, vad_main, signal_func, showDebugPlot=False, description='')
        print()
        print("Fit on pure VAD")
        print(wind_speed)
        print(wind_dir)
        print()

        wind_speed, wind_dir, vad_fit = fitVAD(t, vad_mix, signal_func, showDebugPlot=False, description='')
        print("Weighted fit on contaminated VAD")
        print(wind_speed)
        print(wind_dir)
        print()

        wind_speed, wind_dir, vad_fit = fitVAD(t, vad_mix, signal_func, showDebugPlot=False, description='',
                                               weights=is_main_vad)
        print("Unweighted fit on contaminated VAD")
        print(wind_speed)
        print(wind_dir)
        print()

    def test_MinMaxNormalization(self):
        x = np.array([0.1, 0.5, 0.9])
        x_norm = MinMaxNormalization(x)
        self.assertAlmostEqual(first=x_norm[0], second=(0.1 - 0.1) / .8, delta=0.005)
        self.assertAlmostEqual(first=x_norm[1], second=(0.5 - 0.1) / .8, delta=0.005)
        self.assertAlmostEqual(first=x_norm[2], second=(0.9 - 0.1) / .8, delta=0.005)

        min_val, max_val = -5, 10
        x = np.array([i for i in range(min_val, max_val + 1)])
        x_norm = MinMaxNormalization(x)
        self.assertTrue(np.min(x_norm) >= 0)
        self.assertTrue(np.max(x_norm) <= 1)

    def test_GetVADWeights(self):
        # Birds
        bi_score = np.array([0.6, 0.7, 0.99])
        exp_score = (bi_score - 0.5) / (0.99 - 0.5)
        pred_score = GetVADWeights(bi_scores_cut=bi_score, echo_type=VADMask.birds)
        self.assertTrue(np.logical_and.reduce(np.isclose(a=exp_score, b=pred_score, atol=0.1, rtol=0.1)))

        # Insects
        bi_score = np.array([0.4, 0.3, 0.01])
        bi_score_inv = 1 - bi_score
        exp_score = (bi_score_inv - 0.5) / (0.99 - 0.5)
        pred_score = GetVADWeights(bi_scores_cut=bi_score, echo_type=VADMask.insects)
        self.assertTrue(np.logical_and.reduce(np.isclose(a=exp_score, b=pred_score, atol=0.1, rtol=0.1)))

        # Biological
        bi_score = np.array([0.5, 0.3, 0.01])
        bi_score_inv = 1 - bi_score
        exp_score = (bi_score_inv - 0) / (1 - 0)
        pred_score = GetVADWeights(bi_scores_cut=bi_score, echo_type=VADMask.biological)
        self.assertTrue(np.logical_and.reduce(np.isclose(a=exp_score, b=pred_score, atol=0.1, rtol=0.1)))

        # Weather
        bi_score = np.array([0.6, 0.7, 0.99])
        pred_score = GetVADWeights(bi_scores_cut=bi_score, echo_type=VADMask.weather)
        self.assertTrue(pred_score == 1)

if __name__ == "__main__":
    unittest.main()