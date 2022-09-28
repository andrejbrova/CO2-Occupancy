import os
import glob
import pathlib
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

directory = pathlib.Path(__file__).parent
sys.path.append(str(directory.parent) + '/Repos/datascience/') # Path to datamodel location

from datamodels import datamodels as dm


def load_features():
    training = pd.read_csv(str(directory) + '/occupancy_data/datatraining.txt', parse_dates=['date'])
    test1 = pd.read_csv(str(directory) + '/occupancy_data/datatest.txt', parse_dates=['date'])
    test2 = pd.read_csv(str(directory) + '/occupancy_data/datatest2.txt', parse_dates=['date'])

    training = training.set_index('date')
    test1 = test1.set_index('date')
    test2 = test2.set_index('date')

    return training, test1, test2

def load_dataset(
    historical_co2=False,
    normalize=False,
    embedding=False
    ):
    training, test1, test2 = load_features()

    features = get_feature_list()
    if historical_co2:
        features.append('CO2+1h')
        training['CO2+1h'] = training.loc[:,'CO2'].shift(1)
        training = training.dropna()
        test1['CO2+1h'] = test1.loc[:,'CO2'].shift(1)
        test1 = test1.dropna()
        test2['CO2+1h'] = test2.loc[:,'CO2'].shift(1)
        test2 = test2.dropna()

    X_train = training.loc[:,features]
    y_train = pd.DataFrame(training.loc[:,'Occupancy'])

    X_test_1 = test1.loc[:,features]
    y_test_1 = pd.DataFrame(test1.loc[:,'Occupancy'])

    X_test_2 = test2.loc[:,features]
    y_test_2 = pd.DataFrame(test2.loc[:,'Occupancy'])

    X_test_combined = pd.concat([X_test_1, X_test_2])
    y_test_combined = pd.concat([y_test_1, y_test_2])

    if normalize:
        x_scaler = dm.processing.Normalizer().fit(X_train)

        X_train = x_scaler.transform(X_train)
        X_test_1 = x_scaler.transform(X_test_1)
        X_test_2 = x_scaler.transform(X_test_2)
        X_test_combined = x_scaler.transform(X_test_combined)

    if embedding:
        X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined, _ = get_embeddings()

    return X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined

def load_shaped_dataset(lookback_horizon, prediction_horizon, historical_co2=False, normalize=False):
    X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined = load_dataset(historical_co2, normalize)

    X_train, y_train = dm.processing.shape.get_windows(
        lookback_horizon, X_train.to_numpy(), prediction_horizon, y_train.to_numpy()
    )

    X_test_1, y_test_1 = dm.processing.shape.get_windows(
        lookback_horizon, X_test_1.to_numpy(), prediction_horizon, y_test_1.to_numpy(),
    )

    X_test_2, y_test_2 = dm.processing.shape.get_windows(
        lookback_horizon, X_test_2.to_numpy(), prediction_horizon, y_test_2.to_numpy(),
    )

    X_test_combined, y_test_combined = dm.processing.shape.get_windows(
        lookback_horizon, X_test_combined.to_numpy(), prediction_horizon, y_test_combined.to_numpy(),
    )

    return X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined

def get_embeddings():
    X, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined = load_dataset()

    cat_vars = [
        #'Week',
        'Weekday'
    ]

    #X['Week'] = X.index.isocalendar().week
    X['Weekday'] = X.index.weekday
    #X_test_1['Week'] = X_test_1.index.isocalendar().week
    X_test_1['Weekday'] = X_test_1.index.weekday
    #X_test_2['Week'] = X_test_2.index.isocalendar().week
    X_test_2['Weekday'] = X_test_2.index.weekday

    # Encoding
    X_combined = pd.concat([X, X_test_1, X_test_2])
    encoders = {}
    for v in cat_vars:
        le = LabelEncoder()
        le.fit(X_combined[v].values)
        encoders[v] = le
        X.loc[:, v] = le.transform(X[v].values)
        X_test_1.loc[:, v] = le.transform(X_test_1[v].values)
        X_test_2.loc[:, v] = le.transform(X_test_2[v].values)
        print('{0}: {1}'.format(v, le.classes_))

    # Normalizing
    for v in cat_vars:
        X[v] = X[v].astype('category').cat.as_ordered()
        X_test_1[v] = X_test_1[v].astype('category').cat.as_ordered()
        X_test_2[v] = X_test_2[v].astype('category').cat.as_ordered()

    # Reshape input
    X_array = []
    X_test_1_array = []
    X_test_2_array = []
    X_test_combined_array = []

    X_test_combined = pd.concat([X_test_1, X_test_2])

    for i, v in enumerate(cat_vars):
        X_array.append(X.loc[:, v])
        X_test_1_array.append(X_test_1.loc[:, v])
        X_test_2_array.append(X_test_2.loc[:, v])
        X_test_combined_array.append(X_test_combined.loc[:, v])

    X_array.append(X.iloc[:, ~X.columns.isin(cat_vars)])
    X_test_1_array.append(X_test_1.iloc[:, ~X_test_1.columns.isin(cat_vars)])
    X_test_2_array.append(X_test_2.iloc[:, ~X_test_2.columns.isin(cat_vars)])
    X_test_combined_array.append(X_test_combined.iloc[:, ~X_test_combined.columns.isin(cat_vars)])

    len(X_array), len(X_test_1_array), len(X_test_2_array)

    return X_array, X_test_1_array, X_test_2_array, X_test_combined_array, y_train, y_test_1, y_test_2, y_test_combined, encoders

def summarize_results(
    scores_train,
    scores_test_1,
    scores_test_2,
    scores_test_combined,
    batch_size='?',
    epochs='?',
    repeats='?',
    embedding='?',
    model_name='?'
    ):
    print(scores_train)
    print(scores_test_1)
    result = {
        'batch size': batch_size,
        'epochs': epochs,
        'repeats': repeats,
        'embedding': embedding,
        'accuracy_train_mean': np.mean(scores_train),
        'accuracy_train_std': np.std(scores_train),
        'accuracy_test_1_mean': np.mean(scores_test_1),
        'accuracy_test_1_std': np.std(scores_test_1),
        'accuracy_test_2_mean': np.mean(scores_test_2),
        'accuracy_test_2_std': np.std(scores_test_2),
        'accuracy_test_combined_mean': np.mean(scores_test_combined),
        'accuracy_test_combined_std': np.std(scores_test_combined)
    }
    return pd.DataFrame(result, index=[model_name])

def concat_tables():
    path = os.getcwd() + '/results/'
    csv_files = glob.glob(os.path.join(path, "*.csv"))

    tables = []
    for result_table in csv_files:
        if result_table != (path + 'results.csv'):
            tables.append(pd.read_csv(result_table))
    pd.concat(tables, axis=0).to_csv(path + 'results.csv')

def get_feature_list(set='full'):
    sets = {
        'full': [
            'Temperature',
            'Humidity',
            'Light',
            'CO2',
            'HumidityRatio'
        ],
        'Light+CO2': [
            'Light',
            'CO2'
        ]
    }
    return sets[set]

if __name__ == '__main__':
    concat_tables()