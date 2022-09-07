import pathlib
import sys
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm

directory = pathlib.Path(__file__).parent
sys.path.append(str(directory.parent))

from utils import load_dataset, get_feature_list


def main():
    plot_hourly_distributions()

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

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.3)

    for it, feature in enumerate(features):
        if it < 2:
            x = it
            y = 0
        else:
            x = it-2
            y = 1
        axs[x,y].set_ylabel(labels[it])
        X.boxplot(column=feature, by='hour', ax=axs[x,y], widths=0.8)
    
    axs[0, 1].set_xlabel('Hour')
    axs[1, 1].set_xlabel('Hour')
    fig.suptitle('Hourly distributions of Temperature, CO2, Light and Humidity')
    plt.savefig(str(directory) + '/hourly_distribution.png', dpi=144)
    plt.show()

if __name__ == '__main__':
    main()