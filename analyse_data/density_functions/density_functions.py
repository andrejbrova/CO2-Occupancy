from pathlib import Path

ROOT_DIR = str(Path(__file__).parents[2])
SAVE_DIR = ROOT_DIR + '/analyse_data/density_functions/'

import matplotlib.pyplot as plt
import numpy as np

from get_data import load_dataset


def plot_probability_density_function(location):
    if location in ['Australia', 'Denmark']:
        pdf_with_rooms(location)
    elif location == 'Graz':
        pdf_graz(location)
    else:
        pdf_without_rooms(location)

def pdf_without_rooms(location):
    X, y = load_dataset(location, split_data=False, data_cleaning=False)

    columns = X.select_dtypes(exclude=['category', 'string', 'object']).columns

    colormap = {
        'Temperature': 'blue',
        'CO2': 'red'
    }

    fig, axes = plt.subplots(1, len(columns), figsize=(8, 5))

    for it, column in enumerate(columns):
        ax = axes[it]
        X[column].loc[y['Occupancy']==1].plot.kde(ax=ax, label='Occupied', color=colormap[column], style=['-'])
        X[column].loc[y['Occupancy']==0].plot.kde(ax=ax, label='Not Occupied', color=colormap[column], alpha=0.7, style=['--'])
        ax.set_title(column)
        ax.set_xlabel('Feature Value')
        ax.set_xlim(left=X[column].min(), right=X[column].max())
        ax.grid()
        ax.legend(loc='upper right')
    plt.suptitle(f'Probability Density Functions of the {location} dataset')
    plt.tight_layout()

    plt.savefig(f'{SAVE_DIR}pdf_{location}.png')
    plt.show()

def pdf_graz(location):
    features = [
        'CO2',
        'Humidity',
        'Temperature'
        ]

    X, y = load_dataset(location, split_data=False, data_cleaning=False)

    columns = X.select_dtypes(exclude=['category', 'string', 'object']).columns

    fig, axes = plt.subplots(1, len(features), figsize=(10, 5))

    for it, column in enumerate(features):
        ax = axes[it]
        X[column + ' (WL)'].loc[y['Occupancy']==1].plot.kde(ax=ax, label='WL', color='red', style=['-'])
        X[column + ' (WL)'].loc[y['Occupancy']==0].plot.kde(ax=ax, label='_nolegend_', color='red', alpha=0.7, style=['--'])
        X[column + ' (WM)'].loc[y['Occupancy']==1].plot.kde(ax=ax, label='WM', color='blue', style=['-'])
        X[column + ' (WM)'].loc[y['Occupancy']==0].plot.kde(ax=ax, label='_nolegend_', color='blue', alpha=0.7, style=['--'])
        X[column + ' (WR)'].loc[y['Occupancy']==1].plot.kde(ax=ax, label='WR', color='green', style=['-'])
        X[column + ' (WR)'].loc[y['Occupancy']==0].plot.kde(ax=ax, label='_nolegend_', color='green', alpha=0.7, style=['--'])
        ax.set_title(column)
        ax.set_xlabel('Feature Value')
        #ax.set_xlim(left=X[column].min(), right=X[column].max())
        ax.grid()
        ax.legend(loc='upper right')
    plt.suptitle(f'Probability Density Functions of the {location} dataset')
    plt.tight_layout()

    plt.savefig(f'{SAVE_DIR}pdf_{location}.png')
    plt.show()

def pdf_with_rooms(location):
    X, y = load_dataset(location, split_data=False, data_cleaning=False)

    columns = X.select_dtypes(exclude=['category', 'string', 'object']).columns

    colors = [
        'gray', 'indianred', 'blue', 'turquoise', 'red', 'sienna',
        'olive', 'yellow', 'forestgreen', 'lime', 'cyan', 'gold',
        'orange', 'navy', 'violet', 'darkorchid', 'magenta', 'greenyellow'
        ]

    fig, axes = plt.subplots(1, len(columns), figsize=(8, 5))

    for it, column in enumerate(columns):
        ax = axes[it]
        for room_id in range(1, len(X['Room_ID'].unique())+1):
            room_data_X = X.loc[X['Room_ID'] == room_id]
            room_data_y = y.loc[X['Room_ID'] == room_id]
            room_data_X[column].loc[room_data_y['Occupancy']==1].plot.kde(ax=ax, label=room_id, color=colors[room_id-1], style=['-'])
            room_data_X[column].loc[room_data_y['Occupancy']==0].plot.kde(ax=ax, label='_nolegend_', color=colors[room_id-1], alpha=0.7, style=['--'])
        ax.set_title(column)
        ax.set_xlabel('Feature Value')
        ax.set_xlim(left=X[column].min(), right=X[column].max())
        ax.grid()
        ax.legend(loc='upper right')
    plt.suptitle(f'Probability Density Functions of the {location} dataset')
    plt.tight_layout()

    plt.savefig(f'{SAVE_DIR}pdf_{location}.png')
    plt.show()