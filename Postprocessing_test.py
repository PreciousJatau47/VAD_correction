import unittest
import numpy as np
import pandas as pd
from Postprocessing import prepare_pcolor_grid_from_series, prepare_weekly_data_for_pcolor_plot

class TestPostprocessing(unittest.TestCase):
    def test_prepare_pcolor_grid_from_series(self):
        df = pd.DataFrame({'x': [0, 0, 1, 1], 'y': [0, 1, 0, 1], 'z': [1, 2, 3, 4]})
        uniqueX = [0, 1]
        uniqueY = [0, 1]
        z_dict = {"item1": df['z']}
        zGrid = {"item1": np.array([[1, 2], [3, 4]])}

        uX, uY, zG = prepare_pcolor_grid_from_series(x=df['x'], y=df['y'], z=z_dict, uniqueX=uniqueX,
                                                     uniqueY=uniqueY)
        self.assertEqual(uniqueX, uX)
        self.assertEqual(uniqueY, uY)
        for key in zGrid:
            self.assertTrue(np.array_equal(zGrid[key], zG[key]))

    def test_prepare_weekly_data_for_pcolor_plot(self):
        base_df = pd.DataFrame({'week': [0, 0, 0, 0], 'x': [0, 0, 1, 1], 'y': [0, 1, 0, 1], 'z': [0, 1, 2, 3]})
        dfs = []

        for i in range(3):
            df = base_df.copy()
            df["week"] += i
            df["z"] += i
            dfs.append(df)
        weekly_df = pd.concat(dfs, axis=0)

        base_arr = np.array([[0, 1], [2, 3]])
        exp_weekly_data = {"week_{}".format(i): {'z': base_arr + i} for i in range(3)}

        exp_ux = [0, 1]
        exp_uy = [0, 1]
        month = 5
        noon_s_mid = [0, 24]
        exp_xlab = [["{}/{}".format(month, week * 7 + time_hr // 24 + 1) for time_hr in noon_s_mid] for week in
                    [0, 1, 2]]

        uX, uY, wD, xLabs = prepare_weekly_data_for_pcolor_plot(key_cols='z', x_col_name='x', y_col_name='y',
                                                                in_data=weekly_df, month=month,
                                                                noon_s_midnight=noon_s_mid,
                                                                uniqueX=exp_ux, uniqueY=exp_uy)

        self.assertEqual(exp_ux, uX)
        self.assertEqual(exp_uy, uY)
        for key in exp_weekly_data:
            self.assertTrue(key in wD.keys())
            self.assertTrue('z' in wD[key])
            pred = wD[key]['z'].astype(int)
            self.assertTrue(np.array_equal(exp_weekly_data[key]['z'], pred))
            self.assertEqual(exp_xlab, xLabs)




if __name__ == "__main__":
    unittest.main()
