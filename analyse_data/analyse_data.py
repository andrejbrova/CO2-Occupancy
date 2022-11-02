import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).parents[1])
ANALYSIS_DIR = ROOT_DIR + '/analyse_data/'
sys.path.append(ROOT_DIR)

import warnings
import pandas as pd
import pandera as pa
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import normaltest

from utils import load_dataset, load_dataset_graz, get_feature_list


def main():
    dataset = 'Graz'

    #plot_correlation()
    plot_hourly_distributions(dataset)
    #dataset_validation()

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

    plt.savefig(ROOT_DIR + '/correlation_light_temp.png')
    plt.show()

def plot_hourly_distributions(dataset):
    X, _ = load_dataset(
        dataset=dataset,
        feature_set='full',
        historical_co2=False,
        normalize=False,
        embedding=False,
        shaped=False,
        split_data=False
        )

    X = X.select_dtypes(include=np.number)

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

    X['hour'] = X.index.hour
    X = X.set_index([X.index, 'hour'])

    number_features = X.shape[-1]
    number_cols = 2 if number_features <= 6 else 3

    fig, axs = plt.subplots(number_cols, 2, figsize=(12, 8))
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
        axs[x,y].set_title(features[it])
        axs[x,y].set_ylabel(labels[it])

        cmap = plt.cm.ScalarMappable(cmap='rainbow')
        test_mean = [x for x in range(len(X_list))]
        for patch, color in zip(bplot['boxes'], cmap.to_rgba(test_mean)):
            patch.set_facecolor(color)
    
    axs[1, 0].set_xlabel('Hour')
    axs[1, 1].set_xlabel('Hour')
    fig.suptitle('Hourly distributions of Temperature, CO2, Light and Humidity')

    plt.savefig(ANALYSIS_DIR + dataset + '.png', dpi=144)
    plt.show()

def dataset_validation():
    # References:
    # https://pandera.readthedocs.io/en/stable/hypothesis.html
    # https://pandera.readthedocs.io/en/v0.6.5/generated/pandera.hypotheses.Hypothesis.html?highlight=pa.Hypothesis
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html

    dataset = load_dataset_graz()

    normal_check = pa.Hypothesis(
        test=normaltest,
        relationship=lambda stat, pvalue, alpha=0.05: pvalue >= alpha,
        error='normality test',
        raise_warning=True,
    )

    columns_to_check = {
        'CO2 (WL)': pa.Column(int, normal_check),
        'CO2 (WM)': pa.Column(int, normal_check),
        'CO2 (WR)': pa.Column(int, normal_check),
        'Humidity (WL)': pa.Column(float, normal_check),
        'Humidity (WM)': pa.Column(float, normal_check),
        'Humidity (WR)': pa.Column(float, normal_check),
        'Temperature (WL)': pa.Column(float, normal_check),
        'Temperature (WM)': pa.Column(float, normal_check),
        'Temperature (WR)': pa.Column(float, normal_check),
    }

    schema = pa.DataFrameSchema(
        columns=columns_to_check,
    )

    schema.validate(dataset)

if __name__ == '__main__':
    main()