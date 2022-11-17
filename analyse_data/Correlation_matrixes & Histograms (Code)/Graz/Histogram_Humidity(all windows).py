import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_excel('C:/Users/Brova/occupancy_data/Graz_occupants.xlsx', engine='openpyxl')   # , index_col='Date_Time', na_values=-999, parse_dates=True

df_Humidity_left = dataset['Humidity (WL)']
df_Humidity_middle = dataset['Humidity (WM)']
df_Humidity_right = dataset['Humidity (WR)']
#df_CO2_test = df_CO2_test[(df_CO2_test >= 0)]
#df_CO2_train = df_CO2_train[(df_CO2_train >= 0)]
print(df_Humidity_left)
print(df_Humidity_middle)
print(df_Humidity_right)

plt.hist([df_Humidity_left, df_Humidity_middle, df_Humidity_right], color=['Blue', 'Green', 'Yellow'], label=['Window Left', 'Window Middle', 'Window Right'])
plt.title('Humidity histogram for Graz dataset')
plt.xlabel('Humidity')
plt.ylabel('Frequency')
plt.legend()
plt.show()