import unittest
import numpy as np
import pandas as pd
from Postprocessing import prepare_pcolor_grid_from_series

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


if __name__ == "__main__":
    unittest.main()
