import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import matplotlib.pyplot as plt

dataset_Australia = pd.read_csv('/Users/Brova/Downloads/occupancy_data/Australia/Indoor_Measurement_Study7.csv', index_col='Date_Time')

# CO2 = dataset_Australia["Indoor_CO2[ppm]"]
# Temperature = dataset_Australia["Indoor_Temp[C]"]
# HumidityRatio = dataset_Australia["Indoor_RH[%]"]
# rooms_Australia = dataset_Australia['Room_ID']
# array = dataset_Australia.to_numpy()
#
# matrix = np.asmatrix(dataset_Australia)
#
# matrix_rooms = np.asmatrix([rooms_Australia])
# matrix_CO2 = np.asmatrix([CO2])
#
# rooms = np.asarray(rooms_Australia).reshape(1, -1)
# co2 = np.asarray(CO2).reshape(1, -1)
# temperature = np.asarray(Temperature).reshape(1, -1)
# humidity_ratio = np.asarray(HumidityRatio).reshape(1, -1)

dataset = pd.DataFrame(dataset_Australia)
dataset = dataset.drop('Building_ID', axis=1)

# for i in range(1, 17):
room1 = dataset.loc[dataset['Room_ID'] == 1]
room2 = dataset.loc[dataset['Room_ID'] == 2]
room3 = dataset.loc[dataset['Room_ID'] == 3]
room4 = dataset.loc[dataset['Room_ID'] == 4]
room5 = dataset.loc[dataset['Room_ID'] == 5]
room6 = dataset.loc[dataset['Room_ID'] == 6]
room7 = dataset.loc[dataset['Room_ID'] == 7]
room8 = dataset.loc[dataset['Room_ID'] == 8]
room9 = dataset.loc[dataset['Room_ID'] == 9]
room10 = dataset.loc[dataset['Room_ID'] == 10]
room11 = dataset.loc[dataset['Room_ID'] == 11]
room12 = dataset.loc[dataset['Room_ID'] == 12]
room13 = dataset.loc[dataset['Room_ID'] == 13]
room14 = dataset.loc[dataset['Room_ID'] == 14]
room15 = dataset.loc[dataset['Room_ID'] == 15]
room16 = dataset.loc[dataset['Room_ID'] == 16]

room1array = np.array(room1)
room2array = np.array(room2)
room3array = np.array(room3)
room4array = np.array(room4)
room5array = np.array(room5)
room6array = np.array(room6)
room7array = np.array(room7)
room8array = np.array(room8)
room9array = np.array(room9)
room10array = np.array(room10)
room11array = np.array(room11)
room12array = np.array(room12)
room13array = np.array(room13)
room14array = np.array(room14)
room15array = np.array(room15)
room16array = np.array(room16)

# list0 = []
# for i in range(1,17):
#     room = dataset.loc[dataset['Room_ID'] == i]
#     room_array = np.array(room)
#     list0.append(room_array)

print(room1array)
print(room2array)
cs = cosine_similarity(room1array, room2array)
print(cs)
print(len(cs))
# sn.heatmap(cosine_similarity(room1array, room2array), annot=True)
# plt.show()
#print(cosine_similarity(room1array, room2array, room3array, room4array, room5array, room6array, room7array, room8array, room9array, room10array, room11array, room12array, room13array, room14array, room15array, room16array))
