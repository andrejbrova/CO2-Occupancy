import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_excel('C:/Users/Brova/occupancy_data/Graz_occupants.xlsx', engine='openpyxl')   # , index_col='Date_Time', na_values=-999, parse_dates=True

df_CO2_left = dataset['CO2 (WL)']
df_CO2_middle = dataset['CO2 (WM)']
df_CO2_right = dataset['CO2 (WR)']
#df_CO2_test = df_CO2_test[(df_CO2_test >= 0)]
#df_CO2_train = df_CO2_train[(df_CO2_train >= 0)]
print(df_CO2_left)
print(df_CO2_middle)
print(df_CO2_right)

plt.hist([df_CO2_left, df_CO2_middle, df_CO2_right], color=['Blue', 'Green', 'Yellow'], label=['Window Left', 'Window Middle', 'Window Right'])
plt.title('CO2 histogram for Graz dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.legend()
plt.show()
