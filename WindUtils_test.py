import unittest
import numpy as np
import pandas as pd
from WindUtils import *


class TestWindUtils(unittest.TestCase):
    def test_CalcSmallAngleDirDiffDf(self):
        df = pd.DataFrame({"dir_b": [10, 350, 20, 100], "dir_w": [350, 10, 100, 20]})
        exp_small_ang_diff = pd.Series([20, -20, -80, 80])
        print((exp_small_ang_diff + 360) % 360)
        pred_small_ang_diff = CalcSmallAngleDirDiffDf(df["dir_b"], df["dir_w"])
        assert np.array_equal(exp_small_ang_diff, pred_small_ang_diff)


if __name__ == "__main__":
    unittest.main()
