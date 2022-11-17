import pandas as pd
import matplotlib.pyplot as plt

# Australia dataset has 16 rooms

dataset = pd.read_csv('/Users/Brova/Downloads/occupancy_data/Australia/Indoor_Measurement_Study7.csv')
df_Room1 = dataset[dataset['Room_ID'] == 1]
df_Room2 = dataset[dataset['Room_ID'] == 2]
df_Room3 = dataset[dataset['Room_ID'] == 3]
df_Room4 = dataset[dataset['Room_ID'] == 4]
df_Room5 = dataset[dataset['Room_ID'] == 5]
df_Room6 = dataset[dataset['Room_ID'] == 6]
df_Room7 = dataset[dataset['Room_ID'] == 7]
df_Room8 = dataset[dataset['Room_ID'] == 8]
df_Room9 = dataset[dataset['Room_ID'] == 9]
df_Room10 = dataset[dataset['Room_ID'] == 10]
df_Room11 = dataset[dataset['Room_ID'] == 11]
df_Room12 = dataset[dataset['Room_ID'] == 12]
df_Room13 = dataset[dataset['Room_ID'] == 13]
df_Room14 = dataset[dataset['Room_ID'] == 14]
df_Room15 = dataset[dataset['Room_ID'] == 15]
df_Room16 = dataset[dataset['Room_ID'] == 16]
df_CO2_Room1 = df_Room1['Indoor_CO2[ppm]']
df_CO2_Room2 = df_Room2['Indoor_CO2[ppm]']
df_CO2_Room3 = df_Room3['Indoor_CO2[ppm]']
df_CO2_Room4 = df_Room4['Indoor_CO2[ppm]']
df_CO2_Room5 = df_Room5['Indoor_CO2[ppm]']
df_CO2_Room6 = df_Room6['Indoor_CO2[ppm]']
df_CO2_Room7 = df_Room7['Indoor_CO2[ppm]']
df_CO2_Room8 = df_Room8['Indoor_CO2[ppm]']
df_CO2_Room9 = df_Room9['Indoor_CO2[ppm]']
df_CO2_Room10 = df_Room10['Indoor_CO2[ppm]']
df_CO2_Room11 = df_Room11['Indoor_CO2[ppm]']
df_CO2_Room12 = df_Room12['Indoor_CO2[ppm]']
df_CO2_Room13 = df_Room13['Indoor_CO2[ppm]']
df_CO2_Room14 = df_Room14['Indoor_CO2[ppm]']
df_CO2_Room15 = df_Room15['Indoor_CO2[ppm]']
df_CO2_Room16 = df_Room16['Indoor_CO2[ppm]']

fig1 = plt.figure()
ax1 = fig1.gca()
df_CO2_Room1.hist(ax=ax1)
plt.title('CO2 histogram for Australia Room 1 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig2 = plt.figure()
ax2 = fig2.gca()
df_CO2_Room2.hist(ax=ax2)
plt.title('CO2 histogram for Australia Room 2 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig3 = plt.figure()
ax3 = fig3.gca()
df_CO2_Room3.hist(ax=ax3)
plt.title('CO2 histogram for Australia Room 3 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig4 = plt.figure()
ax4 = fig4.gca()
df_CO2_Room4.hist(ax=ax4)
plt.title('CO2 histogram for Australia Room 4 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig5 = plt.figure()
ax5 = fig5.gca()
df_CO2_Room5.hist(ax=ax5)
plt.title('CO2 histogram for Australia Room 5 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig6 = plt.figure()
ax6 = fig6.gca()
df_CO2_Room6.hist(ax=ax6)
plt.title('CO2 histogram for Australia Room 6 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig7 = plt.figure()
ax7 = fig7.gca()
df_CO2_Room7.hist(ax=ax7)
plt.title('CO2 histogram for Australia Room 7 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig8 = plt.figure()
ax8 = fig8.gca()
df_CO2_Room8.hist(ax=ax8)
plt.title('CO2 histogram for Australia Room 8 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig9 = plt.figure()
ax9 = fig9.gca()
df_CO2_Room9.hist(ax=ax9)
plt.title('CO2 histogram for Australia Room 9 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig10 = plt.figure()
ax10 = fig10.gca()
df_CO2_Room10.hist(ax=ax10)
plt.title('CO2 histogram for Australia Room 10 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig11 = plt.figure()
ax11 = fig11.gca()
df_CO2_Room11.hist(ax=ax11)
plt.title('CO2 histogram for Australia Room 11 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig12 = plt.figure()
ax12 = fig12.gca()
df_CO2_Room12.hist(ax=ax12)
plt.title('CO2 histogram for Australia Room 12 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig13 = plt.figure()
ax13 = fig13.gca()
df_CO2_Room13.hist(ax=ax13)
plt.title('CO2 histogram for Australia Room 13 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig14 = plt.figure()
ax14 = fig14.gca()
df_CO2_Room14.hist(ax=ax14)
plt.title('CO2 histogram for Australia Room 14 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig15 = plt.figure()
ax15 = fig15.gca()
df_CO2_Room15.hist(ax=ax15)
plt.title('CO2 histogram for Australia Room 15 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()

fig16 = plt.figure()
ax16 = fig16.gca()
df_CO2_Room16.hist(ax=ax16)
plt.title('CO2 histogram for Australia Room 16 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()