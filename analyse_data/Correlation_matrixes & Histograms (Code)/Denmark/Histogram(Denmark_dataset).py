import pandas as pd
import matplotlib.pyplot as plt

# Denmark dataset has 3 rooms

dataset = pd.read_csv('/Users/Brova/Downloads/occupancy_data/Denmark/Indoor_Measurement_Study4.csv', index_col='Date_Time', na_values=-999, parse_dates=True)
testData = dataset.loc['6/21/2018  12:00:00 AM':'9/14/2018  11:59:00 PM', :]
trainData = dataset.drop(testData.index)

# df_Room1 = dataset[dataset['Room_ID'] == 1]
# df_Room2 = dataset[dataset['Room_ID'] == 2]
# df_Room3 = dataset[dataset['Room_ID'] == 3]
# df_CO2_Room1 = df_Room1['Indoor_CO2[ppm]']
# df_CO2_Room2 = df_Room2['Indoor_CO2[ppm]']
# df_CO2_Room3 = df_Room3['Indoor_CO2[ppm]']
# fig1 = plt.figure()
# ax1 = fig1.gca()
# df_CO2_Room1.hist(ax=ax1)

df_CO2_test = testData['Indoor_CO2[ppm]']
df_CO2_train = trainData['Indoor_CO2[ppm]']
plt.hist([df_CO2_train, df_CO2_test], color=['Blue', 'Yellow'], label=['Other seasons', 'Summer season'])
# plt.title('CO2 histogram for Denmark Room 1 dataset')
plt.title('CO2 histogram for Denmark dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

# fig2 = plt.figure()
# ax2 = fig2.gca()
# df_CO2_Room2.hist(ax=ax2)
# plt.title('CO2 histogram for Denmark Room 2 dataset')
# plt.xlabel('CO2')
# plt.ylabel('Frequency')
# plt.show()
#
# fig3 = plt.figure()
# ax3 = fig3.gca()
# df_CO2_Room3.hist(ax=ax3)
# plt.title('CO2 histogram for Denmark Room 3 dataset')
# plt.xlabel('CO2')
# plt.ylabel('Frequency')
# plt.show()