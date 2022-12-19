from pathlib import Path

ROOT_DIR = str(Path(__file__).parents[2])
SAVE_DIR = ROOT_DIR + '/analyse_data/density_functions/'

import matplotlib.pyplot as plt
import numpy as np

from get_data import load_dataset


def plot_probability_density_function(location):
    X, y = load_dataset(location, split_data=False, data_cleaning=False)

    columns = X.select_dtypes(exclude=['category', 'string', 'object']).columns

    colormap = {
        'Temperature': 'blue',
        'CO2': 'red'
    }

    fig, axes = plt.subplots(1, len(columns), figsize=(8, 5))
    #fig.subplots_adjust(hspace=0.5)

    for it, column in enumerate(columns):
        ax = axes[it]
        X[column].loc[y['Occupancy']==1].plot.kde(ax=ax, label='Occupied', color=colormap[column], style=['-'])
        X[column].loc[y['Occupancy']==0].plot.kde(ax=ax, label='Not Occupied', color=colormap[column], alpha=0.7, style=['--'])
        ax.set_title(column)
        ax.set_xlabel('Feature Value')
        ax.grid()
        ax.legend(loc='upper right')
    plt.suptitle(f'Probability Density Functions of the {location} dataset')
    plt.tight_layout()

    plt.savefig(f'{SAVE_DIR}pdf_{location}.png')
    plt.show()