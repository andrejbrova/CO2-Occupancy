import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import matplotlib.pyplot as plt

from get_data import get_readings_data

def plot_cosine_similarity(location):
    dataset = get_readings_data(location, data_cleaning=False)

    rooms = []
    for room_id in dataset['Room_ID'].unique():
        room = dataset.loc[dataset['Room_ID'] == room_id].drop(['Room_ID'], axis='columns')
        rooms.append(np.array(room))

    """room1array = np.array(room1)
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
    room16array = np.array(room16)"""

    # list0 = []
    # for i in range(1,17):
    #     room = dataset.loc[dataset['Room_ID'] == i]
    #     room_array = np.array(room)
    #     list0.append(room_array)

    row_similarities = []
    for room_1_row, room_2_row in zip(rooms[0], rooms[1]):
        cs = cosine_similarity([room_1_row], [room_2_row])
        row_similarities.append(cs)
    room_similarity = np.mean(row_similarities)
    print(room_similarity)
    # sn.heatmap(cosine_similarity(room1array, room2array), annot=True)
    # plt.show()
    #print(cosine_similarity(room1array, room2array, room3array, room4array, room5array, room6array, room7array, room8array, room9array, room10array, room11array, room12array, room13array, room14array, room15array, room16array))
