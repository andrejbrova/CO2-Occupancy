import pathlib
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

directory = pathlib.Path(__file__).parent
sys.path.append(str(directory.parent))

from utils import load_dataset, get_feature_list


def main():
    plot_correlation()
    plot_hourly_distributions()

def plot_correlation():
    X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined = load_dataset()
    X = pd.concat([X_train, X_test_1, X_test_2])
    Y = pd.concat([y_train, y_test_1, y_test_2])
    Y = Y.astype('category')
    data = X.join(Y)

    occupancy = data[data['Occupancy'] == 1]
    non_occupancy = data[data['Occupancy'] == 0]

    fig, ax = plt.subplots()
    plt.scatter(x=occupancy['Temperature'], y=occupancy['Light'], color='blue', label='Occupancy')
    plt.scatter(x=non_occupancy['Temperature'], y=non_occupancy['Light'], color='red', label='Non-Occupancy')
    plt.title('Correlation between Light and Temperature')
    plt.xlabel('Temperature [°C]')
    plt.ylabel('Light')
    plt.legend(loc='upper left')

    plt.savefig(str(directory) + '/correlation_light_temp.png')
    plt.show()

def plot_hourly_distributions():
    X_train, X_test_1, X_test_2, *_ = load_dataset()

    features = [
        'Temperature',
        'Light',
        'CO2',
        'Humidity'
    ]
    labels = [
        '°C',
        '',
        '',
        ''
    ]

    X = pd.concat([X_train, X_test_1, X_test_2])

    X['hour'] = X.index.hour
    X = X.set_index([X.index, 'hour'])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.3)

    for it, feature in enumerate(features):
        if it < 2:
            x = it
            y = 0
        else:
            x = it-2
            y = 1

        X_hourly = X[feature].unstack(level=1)
        X_list = []
        for x_hour in X_hourly.columns:
            X_list.append(X_hourly[x_hour].dropna().to_numpy())

        bplot = axs[x,y].boxplot(x=X_list, widths=0.8, patch_artist=True)
        axs[x,y].set_ylabel(labels[it])

        cmap = plt.cm.ScalarMappable(cmap='rainbow')
        test_mean = [x for x in range(len(X_list))]
        for patch, color in zip(bplot['boxes'], cmap.to_rgba(test_mean)):
            patch.set_facecolor(color)
    
    axs[0, 1].set_xlabel('Hour')
    axs[1, 1].set_xlabel('Hour')
    fig.suptitle('Hourly distributions of Temperature, CO2, Light and Humidity')

    plt.savefig(str(directory) + '/hourly_distribution.png', dpi=144)
    plt.show()

if __name__ == '__main__':
    main()