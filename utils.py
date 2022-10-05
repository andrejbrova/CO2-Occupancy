import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[0]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR.parent) + '/Repos/datascience/') # Path to datamodel location

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from datamodels import datamodels as dm


# Shape dimensions (used when 'shaped' is True)
LOOKBACK_HORIZON = 48
PREDICTION_HORIZON = 1

def load_dataset(
    dataset='uci', # 'uci', 'Australia', 'Denmark', 'Italy'
    feature_set='full', # 'full', 'Light+CO2', 'CO2'
    historical_co2=False,
    normalize=False,
    embedding=False,
    shaped=False
    ):

    if embedding and shaped:
        print('Cannot use embedding on shaped dataset')
        exit()

    features = get_feature_list(feature_set)
    
    if dataset == 'uci':
        training, test1, test2 = load_dataset_uci()

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

        print(f'Training on {X_train.shape[0]} samples.')
        print(f'Testing on {X_test_1.shape[0]} samples (Test1).')
        print(f'Testing on {X_test_2.shape[0]} samples (Test2).')
        print(f'input: {X_train.shape[-1]} features ({X_train.columns.tolist()}).')

        if embedding:
            X_train, X_test_1, X_test_2, X_test_combined, _ = get_embeddings(X_train, X_test_1, X_test_2, X_test_combined)

        if shaped:
            X_train, y_train = dm.processing.shape.get_windows(
                LOOKBACK_HORIZON, X_train.to_numpy(), PREDICTION_HORIZON, y_train.to_numpy()
            )
            X_test_1, y_test_1 = dm.processing.shape.get_windows(
                LOOKBACK_HORIZON, X_test_1.to_numpy(), PREDICTION_HORIZON, y_test_1.to_numpy(),
            )
            X_test_2, y_test_2 = dm.processing.shape.get_windows(
                LOOKBACK_HORIZON, X_test_2.to_numpy(), PREDICTION_HORIZON, y_test_2.to_numpy(),
            )
            X_test_combined, y_test_combined = dm.processing.shape.get_windows(
                LOOKBACK_HORIZON, X_test_combined.to_numpy(), PREDICTION_HORIZON, y_test_combined.to_numpy(),
            )
        
        return X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined

    else:
        data = load_dataset_brick(dataset)

        features.append('Room_ID')

        if historical_co2:
            features.append('CO2+1h')
            data['CO2+1h'] = data.loc[:,'CO2'].shift(1)

        X = data.loc[:,data.columns.intersection(features)]
        y = pd.DataFrame(data.loc[:,'Occupancy'])

        if normalize:
            X_scaleable = X.select_dtypes(exclude='category')
            x_scaler = dm.processing.Normalizer().fit(X_scaleable)
            X_scaleable = x_scaler.transform(X_scaleable)
            X[X_scaleable.columns] = X_scaleable

        if shaped:
            X, y = dm.processing.shape.get_windows(
                LOOKBACK_HORIZON, X.to_numpy(), PREDICTION_HORIZON, y.to_numpy()
            )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

        X_test_2 = X_test.iloc[0:100] # Not the best solution, but test2 and test_combined are substituted by dummies here
        X_test_combined = X_test.iloc[0:100]

        if embedding:
            X_train, X_test, X_test_2, X_test_combined, _ = get_embeddings(X_train, X_test, X_test_2, X_test_combined)

        return X_train, X_test, X_test_2, X_test_combined, y_train, y_test, y_test.iloc[0:100], y_test.iloc[0:100]

def load_dataset_uci():
    training = pd.read_csv(str(ROOT_DIR) + '/occupancy_data/datatraining.txt', parse_dates=['date'])
    test1 = pd.read_csv(str(ROOT_DIR) + '/occupancy_data/datatest.txt', parse_dates=['date'])
    test2 = pd.read_csv(str(ROOT_DIR) + '/occupancy_data/datatest2.txt', parse_dates=['date'])

    training = training.set_index('date')
    test1 = test1.set_index('date')
    test2 = test2.set_index('date')

    return training, test1, test2

def load_dataset_brick(country):
    filename_1 = {
        'Denmark': 'Denmark/Indoor_Measurement_Study4.csv',
        'Australia': 'Australia/Indoor_Measurement_Study7.csv',
        'Italy': 'Italy/Indoor_Measurement_Study10.csv',
    }
    filename_2 = {
        'Denmark': 'Denmark/Occupant_Number_Measurement_Study4.csv',
        'Australia': 'Australia/Occupancy_Measurement_Study7.csv',
        'Italy': 'Italy/Occupancy_Measurement_Study10.csv'
    }
    translate_columns = {
        'Indoor_Temp[C]': 'Temperature',
        'Indoor_RH[%]': 'Humidity',
        'Indoor_CO2[ppm]': 'CO2',
        'Occupancy_Measurement[0-Unoccupied;1-Occupied]': 'Occupancy',
        'Occupant_Number_Measurement': 'Occupancy'
    }

    dataset_1 = pd.read_csv(str(ROOT_DIR) + '/occupancy_data/' + filename_1[country], index_col='Date_Time', na_values=-999, parse_dates=True)
    dataset_2 = pd.read_csv(str(ROOT_DIR) + '/occupancy_data/' + filename_2[country], index_col='Date_Time', na_values=-999, parse_dates=True)

    dataset = pd.concat([dataset_1, dataset_2], axis=1)
    dataset = dataset.sort_index()

    dataset = dataset.loc[:,~dataset.columns.duplicated()] # Drop the second Room_ID column
    dataset['Room_ID'] = dataset['Room_ID'].astype('category')

    dataset = dataset.rename(columns=translate_columns)

    dataset = dataset[['Temperature', 'Humidity', 'CO2', 'Room_ID', 'Occupancy']]
    if country == 'Italy':
        dataset = dataset.loc[:, dataset.columns != 'Humidity']
    dataset = dataset.dropna()

    if country == 'Denmark':
        dataset['Occupancy'][dataset['Occupancy'] > 0] = 1

    return dataset

def get_embeddings(X, X_test_1, X_test_2, X_test_combined):
    cat_vars = [
        'Room_ID',
        'Weekday'
    ]

    X['Weekday'] = X.index.weekday
    X_test_1['Weekday'] = X_test_1.index.weekday
    X_test_2['Weekday'] = X_test_2.index.weekday
    X_test_combined['Weekday'] = X_test_combined.index.weekday

    cat_vars = X.columns.intersection(cat_vars)

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
        X_test_combined.loc[:, v] = le.transform(X_test_combined[v].values)
        print('{0}: {1}'.format(v, le.classes_))

    # Normalizing
    for v in cat_vars:
        X[v] = X[v].astype('category').cat.as_ordered()
        X_test_1[v] = X_test_1[v].astype('category').cat.as_ordered()
        X_test_2[v] = X_test_2[v].astype('category').cat.as_ordered()
        X_test_combined[v] = X_test_combined[v].astype('category').cat.as_ordered()

    # Reshape input
    X_array = []
    X_test_1_array = []
    X_test_2_array = []
    X_test_combined_array = []

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

    return X_array, X_test_1_array, X_test_2_array, X_test_combined_array, encoders

def summarize_results(
    scores_train,
    scores_test_1,
    scores_test_2,
    scores_test_combined,
    model_name='?',
    dataset='?',
    batch_size='?',
    epochs='?',
    repeats='?',
    embedding='?',
    feature_set='?',
    historical_co2='?'
    ):
    print(scores_train)
    print(scores_test_1)
    result = {
        'batch size': batch_size,
        'epochs': epochs,
        'repeats': repeats,
        'embedding': embedding,
        'feature set': feature_set,
        'historical co2': historical_co2,
        'accuracy_train_mean': np.mean(scores_train),
        'accuracy_train_std': np.std(scores_train),
        'accuracy_test_1_mean': np.mean(scores_test_1),
        'accuracy_test_1_std': np.std(scores_test_1),
    }

    suffix = ''
    if feature_set != '?' and feature_set != 'full':
        suffix += '_' + feature_set
    if embedding != '?' and embedding != False:
        suffix += '_embedding'
    if dataset == '?' or dataset == 'uci':
        result.update({ # Add evaluation for second and combined test set
            'accuracy_test_2_mean': np.mean(scores_test_2),
            'accuracy_test_2_std': np.std(scores_test_2),
            'accuracy_test_combined_mean': np.mean(scores_test_combined),
            'accuracy_test_combined_std': np.std(scores_test_combined)
        })
        folder = 'results/'
    else:
        folder = 'results_' + dataset + '/'

    pd.DataFrame(result, index=[model_name]).to_csv(str(ROOT_DIR) + '/models/' + folder + model_name + suffix + '.csv')

def concat_tables():
    path = str(ROOT_DIR) + '/models/results/'
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
        ],
        'CO2': [
            'CO2'
        ]
    }
    return sets[set]

if __name__ == '__main__':
    concat_tables()