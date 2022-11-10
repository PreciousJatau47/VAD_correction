import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def AlmostEqual(x, y, epsilon):
    return abs(x - y) <= epsilon

p_in = open("./batch_analysis_logs/KOHX_20180503_test_data.pkl","rb")
wind_error_df = pickle.load(p_in)
p_in.close()
wind_error_df = wind_error_df.sort_values(by=['file_name', 'height_m'])
print(wind_error_df.columns)

p_in = open("./expected_results/KOHX_20180503_test_data/hca_weather_corrected/KOHX_20180503_test_data_rap_130.pkl", "rb")
wind_error_expected = pickle.load(p_in)
p_in.close()
wind_error_expected = wind_error_expected.sort_values(by=['file_name','height_m'])

# Compare dataframes.
tolerance = 1e-5
columns = wind_error_df.columns


# Compare numerical variables. File names are not expected to be equal, so this comparison can be skipped.
# TODO(pjatau) This test needs to be updated.
for col in columns[1:]:
    #TODO(pjatau) add test for airspeed bio(?)
    if col == 'airspeed_bio':
        continue

    status = AlmostEqual(wind_error_expected[col].max(), wind_error_df[col].max(), tolerance)
    print(status)

# TODO
## Averaging test ##
airspeed_log_path = r".\batch_analysis_logs\KOHX_20180503_test_data_launched_20221023_19\KOHX_20180503_test_data.pkl"
airspeed_avg_log_path = r".\batch_analysis_logs\KOHX_20180503_test_data_launched_20221023_19\KOHX_20180503_test_data_averaged_0.5.pkl"

p_in = open(airspeed_log_path, 'rb')
wind_error, id_log = pickle.load(p_in)
p_in.close()

p_in = open(airspeed_avg_log_path, 'rb')
wind_error_avg, id_avg_log = pickle.load(p_in)
p_in.close()

idx_avg = wind_error_avg['file_name'] == "KOHX20180504_051042_V06"
idx = np.logical_and(wind_error['file_name'] >= "KOHX20180504_050000_V06", wind_error['file_name'] <= "KOHX20180504_053000_V06")

plt.figure()
plt.scatter(wind_error['height_m'][idx], wind_error['airspeed_insects'][idx], color= 'blue')
plt.plot(wind_error_avg['height_m'][idx_avg], wind_error_avg['airspeed_insects'][idx_avg], color= 'blueviolet')
plt.scatter(wind_error['height_m'][idx], wind_error['airspeed_birds'][idx], color= 'red')
plt.plot(wind_error_avg['height_m'][idx_avg], wind_error_avg['airspeed_birds'][idx_avg], color= 'tomato')
plt.show()


