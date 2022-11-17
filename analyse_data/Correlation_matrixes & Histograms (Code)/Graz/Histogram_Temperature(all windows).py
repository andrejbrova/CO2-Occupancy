import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_excel('C:/Users/Brova/occupancy_data/Graz_occupants.xlsx', engine='openpyxl')   # , index_col='Date_Time', na_values=-999, parse_dates=True

df_Temperature_left = dataset['Temperature (WL)']
df_Temperature_middle = dataset['Temperature (WM)']
df_Temperature_right = dataset['Temperature (WR)']
#df_CO2_test = df_CO2_test[(df_CO2_test >= 0)]
#df_CO2_train = df_CO2_train[(df_CO2_train >= 0)]
print(df_Temperature_left)
print(df_Temperature_middle)
print(df_Temperature_right)

plt.hist([df_Temperature_left, df_Temperature_middle, df_Temperature_right], color=['Blue', 'Green', 'Yellow'], label=['Window Left', 'Window Middle', 'Window Right'])
plt.title('Temperature histogram for Graz dataset')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.legend()
plt.show()
