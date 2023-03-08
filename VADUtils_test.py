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
