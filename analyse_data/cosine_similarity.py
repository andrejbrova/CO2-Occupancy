from pathlib import Path

ROOT_DIR = str(Path(__file__).parents[1])
ANALYSIS_DIR = ROOT_DIR + '/analyse_data/'

import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from get_data import get_readings_data


def plot_cosine_similarity(location):
    dataset = get_readings_data(location, data_cleaning=False)

    room_ids = dataset['Room_ID'].unique()
    number_rooms = len(room_ids)
    rooms = []
    for room_id in range(1, number_rooms + 1):
        room = dataset.loc[dataset['Room_ID'] == room_id].loc[:, ['Temperature', 'CO2']]
        rooms.append(np.array(room))

    similarity_values = np.zeros((number_rooms, number_rooms))
    for room_1_id in range(number_rooms):
        for room_2_id in range(room_1_id + 1):
            row_similarities = []
            row_similarities = cosine_similarity(rooms[room_1_id], rooms[room_2_id])
            room_similarity = np.mean(row_similarities)
            similarity_values[room_1_id, room_2_id] = room_similarity

    mask = np.triu(np.ones_like(similarity_values, dtype=np.bool), k=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    sn.heatmap(similarity_values, mask=mask, annot=True, fmt='.4f', annot_kws=dict(fontsize=6))
    plt.xticks(plt.xticks()[0], labels=np.arange(1, number_rooms+1))
    plt.xlabel('First Room')
    plt.yticks(plt.yticks()[0], labels=np.arange(1, number_rooms+1))
    plt.ylabel('Second Room')
    plt.suptitle(f'Cosine Similarity between Rooms for the {location} dataset')

    plt.savefig(f'{ANALYSIS_DIR}cosine_similarity/cosine_similarity_{location}.png')
    plt.show()

def cosine_similarity(x, y):
    """
    Computes Cosine Similarity between two vectors
    """
    return np.sum(x*y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))