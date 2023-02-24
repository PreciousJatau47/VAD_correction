import unittest
import numpy as np
import pandas as pd
from WindUtils import *


class TestWindUtils(unittest.TestCase):
    def test_CalcSmallAngleDirDiff(self):
        # dirn1 = 10, dirn2 = 350, result = 20
        self.assertEqual(CalcSmallAngleDirDiff(10, 350), 20)
        # dirn1 = 350, dirn2 = 10, result = -20
        self.assertEqual(CalcSmallAngleDirDiff(350, 10), -20)
        return

    def test_CalcSmallAngleDirDiffDf(self):
        df = pd.DataFrame({"dir_b": [10, 350, 20, 100], "dir_w": [350, 10, 100, 20]})
        exp_small_ang_diff = pd.Series([20, -20, -80, 80])
        pred_small_ang_diff = CalcSmallAngleDirDiffDf(df["dir_b"], df["dir_w"])
        self.assertTrue(np.array_equal(exp_small_ang_diff, pred_small_ang_diff))

    def test_CalcPolarDiffVec(self):
        # Can be modelled as vr - vw = vf

        # Test 1
        # v1 = (4,0), v2 = (3,90), result = (5, 324)
        exp_pol_diff = (5, 323.13)
        actual_diff = CalcPolarDiffVec(spd1=4, dirn1=0, spd2=3, dirn2=90)
        self.assertTrue((exp_pol_diff[0] - actual_diff[0]) < 0.5)
        self.assertTrue((exp_pol_diff[1] - actual_diff[1]) < 2)

        # Test 2
        # v1 = (5,143.13), v2 = (3,90), result = (4, 180)
        exp_pol_diff = (4, 180)
        actual_diff = CalcPolarDiffVec(spd1=5, dirn1=143.13, spd2=3, dirn2=90)
        self.assertTrue((exp_pol_diff[0] - actual_diff[0]) < 0.5)
        self.assertTrue((exp_pol_diff[1] - actual_diff[1]) < 2)

        # Test 3
        # v1 = (5,233.13), v2 = (3,180), result = (4, 270)
        exp_pol_diff = (4, 270)
        actual_diff = CalcPolarDiffVec(spd1=5, dirn1=233.13, spd2=3, dirn2=180)
        self.assertTrue((exp_pol_diff[0] - actual_diff[0]) < 0.5)
        self.assertTrue((exp_pol_diff[1] - actual_diff[1]) < 2)


if __name__ == "__main__":
    unittest.main()
