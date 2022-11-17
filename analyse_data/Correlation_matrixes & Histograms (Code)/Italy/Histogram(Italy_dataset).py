import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/Brova/Downloads/occupancy_data/Italy/Indoor_Measurement_Study10.csv', index_col='Date_Time', na_values=-999, parse_dates=True)
testData = dataset.loc['6/21/2016  12:00:00 AM':'9/21/2016  11:59:00 PM', :]
trainData = dataset.drop(testData.index)

df_CO2_test = testData['Indoor_CO2[ppm]']
df_CO2_test = df_CO2_test[(df_CO2_test >= 0)]
df_CO2_train = trainData['Indoor_CO2[ppm]']
df_CO2_train = df_CO2_train[(df_CO2_train >= 0)]
print(df_CO2_train)
print(df_CO2_test)

plt.hist([df_CO2_train, df_CO2_test], color=['Blue', 'Yellow'], label=['Other seasons', 'Summer season'])
plt.title('CO2 histogram for Italy dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.legend()
plt.show()
