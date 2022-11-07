import sys
from datetime import datetime
from pathlib import Path

from pandas import DatetimeIndex

ROOT_DIR = str(Path(__file__).parents[0])
sys.path.append(ROOT_DIR)
sys.path.append(str(Path(__file__).parents[1]) + '/Repos/datascience/')  # Path to datamodel location

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from datamodels import datamodels as dm

DATA_DIR = ROOT_DIR + '/occupancy_data/'

# Shape dimensions (used when 'shaped' is True)
LOOKBACK_HORIZON = 48
PREDICTION_HORIZON = 1


def load_dataset(
        dataset='uci',  # 'uci', 'Australia', 'Denmark', 'Italy', 'Graz'
        feature_set='full',  # 'full', 'Light+CO2', 'CO2'
        historical_co2=False,
        normalize=False,
        embedding=False,
        shaped=False,
        split_data=True # True, False, 'Seasonal'
):
    if embedding and shaped:
        print('Cannot use embedding on shaped dataset')
        exit()

    features = get_feature_list(feature_set)

    if dataset == 'uci':
        training, test1, test2 = load_dataset_uci()

        if historical_co2 != False:
            if isinstance(historical_co2, int):
                shift = historical_co2
            else:
                shift = 1
            features.append('CO2+shift')
            training['CO2+shift'] = training.loc[:, 'CO2'].shift(shift)
            training = training.dropna()
            test1['CO2+shift'] = test1.loc[:, 'CO2'].shift(shift)
            test1 = test1.dropna()
            test2['CO2+shift'] = test2.loc[:, 'CO2'].shift(shift)
            test2 = test2.dropna()

        X_train = training.loc[:, features]
        y_train = pd.DataFrame(training.loc[:, 'Occupancy'])

        X_test_1 = test1.loc[:, features]
        y_test_1 = pd.DataFrame(test1.loc[:, 'Occupancy'])

        X_test_2 = test2.loc[:, features]
        y_test_2 = pd.DataFrame(test2.loc[:, 'Occupancy'])

        X_test_combined = pd.concat([X_test_1, X_test_2])
        y_test_combined = pd.concat([y_test_1, y_test_2])

        X_test_list = [X_test_1, X_test_2, X_test_combined]
        y_test_list = [y_test_1, y_test_2, y_test_combined]

        if normalize:
            x_scaler = dm.processing.Normalizer().fit(X_train)
            X_train = x_scaler.transform(X_train)
            for it, X_test in enumerate(X_test_list):
                X_test_list[it] = x_scaler.transform(X_test)

        print(f'Training on {X_train.shape[0]} samples.')
        print(f'Testing on {X_test_1.shape[0]} samples (Test1).')
        print(f'Testing on {X_test_2.shape[0]} samples (Test2).')
        print(f'input: {X_train.shape[-1]} features ({X_train.columns.tolist()}).')

        if embedding:
            X_train, X_test_list, _ = get_embeddings(X_train, X_test_list)

        if shaped:
            X_train, y_train = dm.processing.shape.get_windows(
                LOOKBACK_HORIZON, X_train.to_numpy(), PREDICTION_HORIZON, y_train.to_numpy()
            )
            for X_test, y_test in zip(X_test_list, y_test_list):
                X_test, y_test = dm.processing.shape.get_windows(
                    LOOKBACK_HORIZON, X_test.to_numpy(), PREDICTION_HORIZON, y_test.to_numpy(),
                )

        return X_train, X_test_list, y_train, y_test_list

    elif dataset == 'Graz':
        data = load_dataset_graz()

        # TODO historical CO2

        # TODO feature sets

        X = data.loc[:, data.columns != 'Occupancy']
        y = pd.DataFrame(data.loc[:, 'Occupancy'])

        if normalize:
            X_scaleable = X.select_dtypes(exclude=['category', 'string', 'object'])
            x_scaler = dm.processing.Normalizer().fit(X_scaleable)

        if shaped:
            X, y = dm.processing.shape.get_windows(
                LOOKBACK_HORIZON, X.to_numpy(), PREDICTION_HORIZON, y.to_numpy()
            )

        if not split_data:
            return X, y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
        X_test_list = [X_test]
        y_test_list = [y_test]

        if normalize:
            columns = X.select_dtypes(exclude=['category', 'string', 'object']).columns
            x_scaler = dm.processing.Normalizer().fit(X_train)
            X_train.loc[:,columns] = x_scaler.transform(X_train.loc[:,columns])
            X_test_list[0].loc[:,columns] = x_scaler.transform(X_test_list[0].loc[:,columns])

        if embedding:
            X_train, X_test_list, _ = get_embeddings(X_train, X_test_list)

        return X_train, X_test_list, y_train, y_test_list

    else:
        data = load_dataset_brick(dataset)

        features.append('Room_ID')

        if historical_co2:
            if isinstance(historical_co2, int):
                shift = historical_co2
            else:
                shift = 1
            features.append('CO2+shift')
            data['CO2+shift'] = data.loc[:, 'CO2'].shift(shift)

        X = data.loc[:, data.columns.intersection(features)]
        y = pd.DataFrame(data.loc[:, 'Occupancy'])

        if shaped:
            X, y = dm.processing.shape.get_windows(
                LOOKBACK_HORIZON, X.to_numpy(), PREDICTION_HORIZON, y.to_numpy()
            )

        if not split_data:
            return X, y
        elif split_data == 'Seasonal':
            if dataset=='Denmark':
                X_test = X.loc['6/21/2018  12:00:00 AM':'9/14/2018  11:59:00 PM', :]
            elif dataset=='Italy':
                X_test = X.loc['6/21/2016  12:00:00 AM':'9/21/2016  11:59:00 PM', :]
            elif dataset=='Australia':
                X_test = X.loc['1/23/2020  12:00:00 AM':'3/23/2020  11:55:00 PM', :]
            else:
                print('The dataset you chose is not one of the new datasets!')
            X_train = X.drop(X_test.index)
            y_train = y.drop(X_test.index)
            y_test = y.loc[y_test.index,:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

        if normalize:
            columns = X.select_dtypes(exclude=['category', 'string', 'object']).columns
            x_scaler = dm.processing.Normalizer().fit(X_train)
            X_train.loc[:,columns] = x_scaler.transform(X_train.loc[:,columns])
            X_test.loc[:,columns] = x_scaler.transform(X_test.loc[:,columns])

        X_test_list = [X_test]
        y_test_list = [y_test]

        if embedding:
            X_train, X_test_list, _ = get_embeddings(X_train, X_test_list)

        return X_train, X_test_list, y_train, y_test_list

def load_dataset_uci():
    training = pd.read_csv(ROOT_DIR + '/occupancy_data/datatraining.txt', parse_dates=['date'])
    test1 = pd.read_csv(ROOT_DIR + '/occupancy_data/datatest.txt', parse_dates=['date'])
    test2 = pd.read_csv(ROOT_DIR + '/occupancy_data/datatest2.txt', parse_dates=['date'])

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

    dataset_1 = pd.read_csv(DATA_DIR + filename_1[country], index_col='Date_Time',
                            na_values=-999, parse_dates=True)
    dataset_2 = pd.read_csv(DATA_DIR + filename_2[country], index_col='Date_Time',
                            na_values=-999, parse_dates=True)

    dataset = pd.concat([dataset_1, dataset_2], axis=1)
    dataset = dataset.sort_index()

    dataset = dataset.loc[:, ~dataset.columns.duplicated()]  # Drop the second Room_ID column
    dataset['Room_ID'] = dataset['Room_ID'].astype('category')

    dataset = translate_columns(dataset)

    dataset = dataset[['Temperature', 'CO2', 'Room_ID', 'Occupancy']]
    if country == 'Italy':
        dataset = dataset.loc[:, dataset.columns != 'Humidity']
    dataset = dataset.dropna()

    if country == 'Denmark':
        dataset.loc[dataset.loc[:, 'Occupancy'] > 0, 'Occupancy'] = 1

    return dataset

def load_dataset_graz():
    dataset = pd.read_excel(
        DATA_DIR + 'Graz_occupants.xlsx',
        index_col='datetime',
        parse_dates=True
    )

    for column in dataset.iloc[:, 4:]:
        dataset.loc[:, column] = pd.to_numeric(dataset.loc[:, column].str.replace(r'[^0-9-.]+', ''))

    dataset = translate_columns(dataset)

    dataset.loc[dataset.loc[:, 'Occupancy'] > 0, 'Occupancy'] = 1
    
    return dataset

def translate_columns(dataset): # Uses a dictionary to translate columns to a unified naming system
    column_dict = {
        'Indoor_Temp[C]': 'Temperature',
        'Indoor_RH[%]': 'Humidity',
        'Indoor_CO2[ppm]': 'CO2',
        'Occupancy_Measurement[0-Unoccupied;1-Occupied]': 'Occupancy',
        'Occupant_Number_Measurement': 'Occupancy',
        'people': 'Occupancy'
    }

    dataset = dataset.rename(columns=column_dict)

    return dataset

def get_embeddings(X, X_test_list):
    cat_vars = [
        'Room_ID',
        'Weekday'
    ]

    X = X.assign(Weekday=X.index.weekday)
    for X_test in X_test_list:
        X_test.loc[:,'Weekday'] = X_test.index.weekday

    cat_vars = X.columns.intersection(cat_vars)

    # Encoding
    X_combined = pd.concat([X] + X_test_list)
    encoders = {}
    for v in cat_vars:
        le = LabelEncoder()
        le.fit(X_combined[v].values)
        encoders[v] = le
        X.loc[:, v] = le.transform(X[v].values)
        for X_test in X_test_list:
            X_test.loc[:, v] = le.transform(X_test[v].values)
        print('{0}: {1}'.format(v, le.classes_))

    # Normalizing
    for v in cat_vars:
        X[v] = X[v].astype('category').cat.as_ordered()
        for X_test in X_test_list:
            X_test.loc[:,v] = X_test[v].astype('category').cat.as_ordered()

    # Reshape input
    X_array = []
    X_test_array_list = []

    for i, v in enumerate(cat_vars):
        X_array.append(X.loc[:, v])
    for X_test in X_test_list:
        X_test_array = []
        for i, v in enumerate(cat_vars):
            X_test_array.append(X_test.loc[:, v])
        X_test_array_list.append(X_test_array)

    X_array.append(X.iloc[:, ~X.columns.isin(cat_vars)])
    for X_test, X_test_array in zip(X_test_list, X_test_array_list):
        X_test_array.append(X_test.iloc[:, ~X_test.columns.isin(cat_vars)])

    return X_array, X_test_array_list, encoders


def summarize_results(
        scores_train,
        scores_test_list,
        model_name='?',
        dataset='?',
        batch_size='?',
        epochs='?',
        repeats='?',
        embedding='?',
        feature_set='?',
        historical_co2='?',
        suffix=''
):
    result = {
        'batch size': batch_size,
        'epochs': epochs,
        'repeats': repeats,
        'embedding': embedding,
        'feature set': feature_set,
        'historical co2': historical_co2,
        'accuracy_train_mean': np.mean(scores_train),
        'accuracy_train_std': np.std(scores_train),
        'accuracy_test_1_mean': np.mean([sublist[0] for sublist in scores_test_list]),
        'accuracy_test_1_std': np.std([sublist[0] for sublist in scores_test_list]),
    }

    if feature_set != '?' and feature_set != 'full':
        suffix += '_' + feature_set
    if embedding != '?' and embedding != False:
        suffix += '_embedding'
    if dataset == '?' or dataset == 'uci':
        result.update({  # Add evaluation for second and combined test set
            'accuracy_test_2_mean': np.mean([sublist[1] for sublist in scores_test_list]),
            'accuracy_test_2_std': np.std([sublist[1] for sublist in scores_test_list]),
            'accuracy_test_combined_mean': np.mean([sublist[2] for sublist in scores_test_list]),
            'accuracy_test_combined_std': np.std([sublist[2] for sublist in scores_test_list])
        })
        folder = 'results/'
    else:
        folder = 'results_' + dataset + '/'

    pd.DataFrame(result, index=[model_name]).to_csv(ROOT_DIR + '/models/' + folder + model_name + suffix + '.csv')


def concat_tables():
    path = ROOT_DIR + '/models/results/'
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
