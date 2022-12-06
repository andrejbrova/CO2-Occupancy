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
from matplotlib.colors import ListedColormap
from scipy.stats import normaltest
import seaborn as sn
import pickle as pkl

from get_data import load_dataset, load_dataset_uci, load_dataset_brick, load_dataset_graz, get_feature_list


def main():
    dataset_name = 'Australia'

    describe_dataset(dataset_name)
    #plot_correlation()
    #plot_timeline(dataset_name, data_cleaning=False)
    #plot_correlation_matrix(dataset_name)
    #plot_hourly_distributions(dataset_name)
    #dataset_validation()

def describe_dataset(dataset_name):
    if dataset_name == 'uci':
        datasets = load_dataset_uci()
    elif dataset_name in ['Denmark', 'Australia', 'Italy']:
        dataset = load_dataset_brick(dataset_name)
        datasets = [dataset]
    else:
        X, y = load_dataset(
            dataset=dataset_name,
            feature_set='full',
            historical_co2=False,
            normalize=False,
            embedding=False,
            shaped=False,
            split_data=False
        )

    for dataset in datasets:
        occupancy_count = dataset['Occupancy'].value_counts()
        dataset_properties = pd.DataFrame(data={
            'number rows': [dataset.shape[0]],
            'number occupancies': [occupancy_count[1]],
            'number non-occupancies': [occupancy_count[0]],
            'share occupancies': [round((occupancy_count[1] / np.sum(occupancy_count)) * 100, 1)],
            'rows with nodata values': [len(dataset[dataset.isna().any(axis=1)])],
            'temporal resolution': ['?'],
            'date from': [dataset.index.min()],
            'date to': [dataset.index.max()]

        }).T
        print(dataset_properties)

        descriptive_stats = dataset.describe()
        print(descriptive_stats)

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

def plot_timeline(dataset, data_cleaning=True):
    """
    Plots the features of a dataset with time on the x axis and the feature values on the y axis
    """
    #if dataset != 'Graz':
     #   raise Exception('This function only works with Graz dataset so far!')

    lim = False
    date = pd.Timestamp('2022-07-26')
    n_days = 7

    X, y = load_dataset(
        dataset=dataset,
        feature_set='full',
        historical_co2=False,
        normalize=False,
        embedding=False,
        shaped=False,
        split_data=False,
        data_cleaning=data_cleaning
        )

    if dataset == 'Graz':
        features = [
            'CO2',
            'Humidity',
            'Temperature'
        ]

        fig, ax = plt.subplots(len(features), 1, figsize=(6, 3*len(features)))

        for row, column in enumerate(features):
            ax[row].set_title(column)
            X[column + ' (WL)'].plot(ax=ax[row], color='red', label='Window Left', xlabel=None)
            X[column + ' (WM)'].plot(ax=ax[row], color='blue', label='Window Middle', xlabel=None)
            X[column + ' (WR)'].plot(ax=ax[row], color='green', label='Window Right', xlabel=None)
            ax[row].set_ylabel('Feature value')
            if lim:
                ax[row].set_xlim(left=date, right=date + pd.DateOffset(days=7))
            ax[row].grid()
            ax[row].legend()
    
    else:
        number_columns = len(X.columns)
        fig, ax = plt.subplots(number_columns, 1, figsize=(6, 2*number_columns), sharex=True)
        fig.subplots_adjust(top=0.9)
        fig.add_gridspec(3, hspace=5)

        for row, column in enumerate(X.columns):
            ax[row].set_title(column)
            X[column].plot(ax=ax[row], label=column, xlabel=None)
            ax[row].set_xlabel('Date')
            ax[row].set_ylabel('Feature value')
            if lim:
                ax[row].set_xlim(left=date, right=date + pd.DateOffset(days=7))
            ax[row].grid()
            ax[row].label_outer()

            start_time = None
            end_time = None
            for timestamp in y.index:
                if not start_time and y.loc[timestamp].values[0] == 1:
                    start_time = timestamp
                elif start_time and y.loc[timestamp].values[0] == 0:
                    end_time = timestamp
                if start_time and end_time:
                    ax[row].axvspan(start_time, end_time, facecolor='lightblue', alpha=0.7)
                    start_time=None
                    end_time=None

    plt.suptitle(dataset + ' Timelines')
    plt.tight_layout()

    pkl.dump((fig, ax), open(ANALYSIS_DIR + 'Timelines/' + dataset + '.pickle', 'wb'))
    plt.savefig(ANALYSIS_DIR + 'Timelines/' + dataset + '.png')
    plt.show()
    plt.clf()

def plot_correlation_matrix(dataset_name):
    dataset_functions = {
        'Graz': load_dataset_graz
    }
    dataset = dataset_functions[dataset_name]()

    corrMatrix = dataset.corr()

    plt.figure(figsize=(12, 8))

    ax = sn.heatmap(corrMatrix, annot=True)
    ax.figure.tight_layout()

    plt.savefig(ANALYSIS_DIR + 'Correlation_Matrices/Corr_' + dataset_name + '.png')
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

    X['hour'] = X.index.hour
    X = X.set_index([X.index, 'hour'])

    number_features = X.shape[-1]
    number_cols = 2 if number_features <= 6 else 3
    number_rows = int(np.ceil(number_features / number_cols))

    fig, axs = plt.subplots(number_cols, number_rows, figsize=(6*number_cols, 4*number_rows))
    fig.subplots_adjust(hspace=0.3)

    for it, feature in enumerate(X.columns):
        x = it % number_cols
        y = int(it / number_cols)

        X_hourly = X[feature].unstack(level=1)
        X_list = []
        for x_hour in X_hourly.columns:
            X_list.append(X_hourly[x_hour].dropna().to_numpy())

        bplot = axs[x,y].boxplot(x=X_list, widths=0.8, patch_artist=True)
        axs[x,y].set_title(feature)
        #axs[x,y].set_ylabel(labels[it])

        cmap = plt.cm.ScalarMappable(cmap='rainbow')
        test_mean = [x for x in range(len(X_list))]
        for patch, color in zip(bplot['boxes'], cmap.to_rgba(test_mean)):
            patch.set_facecolor(color)
    
    for ax in axs.flat:
        ax.set(xlabel='Hours')

    fig.suptitle('Hourly distributions of features for ' + dataset + ' dataset')

    plt.savefig(ANALYSIS_DIR + 'Hourly_distribution/' + dataset + '.png', dpi=144)
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