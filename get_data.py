import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).parents[0]) + '/'
sys.path.append(str(Path(__file__).parents[1]) + '/Repos/datascience/')  # Path to datamodel location

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from datamodels import datamodels as dm

DATA_DIR = ROOT_DIR + 'occupancy_data/'

# Shape dimensions (used when 'shaped' is True)
LOOKBACK_HORIZON = 48
PREDICTION_HORIZON = 1


def load_dataset(
        dataset='Denmark',  # 'uci', 'Australia', 'Denmark', 'Italy', 'Graz'
        feature_set='full',  # 'full', 'Light+CO2', 'CO2'
        historical_co2=False,
        normalize=False,
        embedding=False,
        shaped=False,
        split_data='Seasonal', # True, False, 'Seasonal'
        data_cleaning=True
):

    features = get_feature_list(feature_set)

    if dataset == 'uci':
        training, test1, test2 = load_dataset_uci(data_cleaning)

        if historical_co2:
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

        if not split_data:
            data = pd.concat([training, test1, test2], axis='rows').sort_index()
            X = data.loc[:, data.columns.intersection(features)]
            y = pd.DataFrame(data.loc[:, 'Occupancy'])

            return X, y

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

        print(f'Training on {X_train.shape[0]} samples.')
        print(f'Testing on {X_test_1.shape[0]} samples (Test1).')
        print(f'Testing on {X_test_2.shape[0]} samples (Test2).')
        print(f'input: {X_train.shape[-1]} features ({X_train.columns.tolist()}).')

    elif dataset == 'Graz':
        data = load_dataset_graz()

        # TODO historical CO2

        # TODO feature sets

        X = data.loc[:, data.columns != 'Occupancy']
        y = pd.DataFrame(data.loc[:, 'Occupancy'])

        if not split_data:
            return X, y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
        X_test_list = [X_test]
        y_test_list = [y_test]

    else:
        data = load_dataset_brick(dataset)

        features += ['Room_ID', 'Month']

        if historical_co2:
            if isinstance(historical_co2, int):
                shift = historical_co2
            else:
                shift = 1
            features.append('CO2+shift')
            data['CO2+shift'] = data.loc[:, 'CO2'].shift(shift)

        X = data.loc[:, data.columns.intersection(features)]
        y = pd.DataFrame(data.loc[:, 'Occupancy'])

        if not split_data:
            return X, y
        # THE COMMENTED CODE BELOW IS THE SEASONAL SPLIT, BASED ON DATES (SUMMER DATES FOR TEST SET, ALL OTHER FOR TRAIN SET)!
        # elif split_data == 'Seasonal':
        #     if dataset=='Denmark':
        #         X_test = X.loc['6/21/2018  12:00:00 AM':'9/14/2018  11:59:00 PM', :]
        #     elif dataset=='Italy':
        #         X_test = X.loc['6/21/2016  12:00:00 AM':'9/21/2016  11:59:00 PM', :]
        #     elif dataset=='Australia':
        #         X_test = X.loc['1/23/2020  12:00:00 AM':'3/23/2020  11:55:00 PM', :]
        #     else:
        #         print('The dataset you chose is not one of the new datasets!')
        #     X_train = X.drop(X_test.index)
        #     y_train = y.drop(X_test.index)
        #     #y_test = y.loc[y_test.index,:]

        # SAME AS ABOVE (SEASONAL SPLIT), BUT SPLITTING DEPENDING ON THE TEMPERATURE (HIGHER THAN 22 DEGREES GOES TO TEST SET, LOWER/EQUAL TO 22 DEGREES GOES TO TRAIN SET)!
        elif split_data == 'Seasonal':
            t = 22.0
            print("This is X: ")
            print(X)
            if dataset == 'Denmark':
                X_test = X.loc['11/1/2018  12:00:00 AM':'2/21/2019  11:59:00 PM', :]   # till end of October? Ask Mina
                print("This is X_test: ")
                print(X_test)
            elif dataset == 'Italy':
                X_test = X[X['Temperature'] > t]
                print("This is X_test: ")
                print(X_test)
            elif dataset == 'Australia':
                X_test = X[X['Temperature'] > t].copy()
                print("This is X_test: ")
                print(X_test)
                # X_train = X[X['Temperature'] <= t].copy()
                # print("This is X_train: ")
                # print(X_train)
                # train_data = data[data['Temperature'] <= t]
                # y_train = pd.DataFrame(train_data.loc[:, 'Occupancy'])
                # print("This is y_train: ")
                # print(y_train)
                # # y_test = y.drop(y_train.index)
                # test_data = data[data['Temperature'] > t]
                # y_test = pd.DataFrame(test_data.loc[:, 'Occupancy'])
                # print("This is y_test")
                # print(y_test)
            else:
                print('The dataset you chose is not one of the new datasets!')
            X_train = X.drop(X_test.index)
            print("This is X_train: ")
            print(X_train)
            y_train = y.drop(X_test.index)
            print("This is y_train: ")
            print(y_train)
            y_test = y.drop(y_train.index)
            print("This is y_test")
            print(y_test)

        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

        X_test_list = [X_test]
        y_test_list = [y_test]

    if normalize:
        columns = X_train.select_dtypes(exclude=['category', 'string', 'object']).columns
        x_scaler = dm.processing.Normalizer().fit(X_train.loc[:,columns])
        X_train.loc[:,columns] = x_scaler.transform(X_train.loc[:,columns])
        for X_test in X_test_list:
            X_test.loc[:,columns] = x_scaler.transform(X_test.loc[:,columns])

    for data in [X_train] + X_test_list:
    #    data.loc[:,'MinuteOfDay'] = data.index.hour * 60 + data.index.minute
        data.loc[:,'DayOfWeek'] = data.index.dayofweek
      #  data.loc[:,'WeekOfYear'] = data.index.isocalendar().week.astype(int)

    encoders = None
    if embedding:
        X_train, X_test_list, encoders = get_embeddings(X_train, X_test_list)

    if shaped:
        if embedding:
            y_test_list_temp = [None] * len(y_test_list)
            for emb in range(len(X_train)):
                X_train[emb], y_train_temp = dm.processing.shape.get_windows(
                    LOOKBACK_HORIZON, X_train[emb].to_numpy(dtype='float32'), PREDICTION_HORIZON, y_train.to_numpy(dtype='float32')
                )
                for it, (X_test, y_test) in enumerate(zip(X_test_list, y_test_list)):
                    X_test_list[it][emb], y_test_list_temp[it] = dm.processing.shape.get_windows(
                        LOOKBACK_HORIZON, X_test[emb].to_numpy(dtype='float32'), PREDICTION_HORIZON, y_test.to_numpy(dtype='float32'),
                    )
            y_train = y_train_temp
            y_test_list = y_test_list_temp
        else:
            X_train, y_train = dm.processing.shape.get_windows(
                LOOKBACK_HORIZON, X_train.to_numpy(dtype='float32'), PREDICTION_HORIZON, y_train.to_numpy(dtype='float32')
            )
            for it, (X_test, y_test) in enumerate(zip(X_test_list, y_test_list)):
                X_test_list[it], y_test_list[it] = dm.processing.shape.get_windows(
                    LOOKBACK_HORIZON, X_test.to_numpy(dtype='float32'), PREDICTION_HORIZON, y_test.to_numpy(dtype='float32'),
                )

    return X_train, X_test_list, y_train, y_test_list, encoders

def get_readings_data(location='Denmark', data_cleaning=True):
    if location == 'uci':
        return load_dataset_uci(data_cleaning)
    elif location == 'Graz':
        return load_dataset_graz(data_cleaning)
    else:
        return load_dataset_brick(location)

def load_dataset_uci(data_cleaning=True):
    """
    Loading and preparing the uci dataset.
    If data cleaning is true, faulty data points will be interpolated linearly.
    """

    training = pd.read_csv(ROOT_DIR + '/occupancy_data/datatraining.txt', parse_dates=['date'])
    test1 = pd.read_csv(ROOT_DIR + '/occupancy_data/datatest.txt', parse_dates=['date'])
    test2 = pd.read_csv(ROOT_DIR + '/occupancy_data/datatest2.txt', parse_dates=['date'])

    training = training.set_index('date')
    test1 = test1.set_index('date')
    test2 = test2.set_index('date')

    if data_cleaning:
        faulty = {
            'Light': [
                slice("2015-02-04 09:40:00", "2015-02-04 09:42:00"),
                slice("2015-02-07 09:40:59", "2015-02-07 09:43:59"),
                slice("2015-02-12 09:45:00", "2015-02-12 09:49:00"),
                slice("2015-02-13 09:49:00", "2015-02-13 09:49:00"),
            ],
            'CO2': [
                slice("2015-02-09 22:10:00", "2015-02-09 22:14:00"),
                slice("2015-02-11 18:51:00", "2015-02-11 18:53:00"),
                slice("2015-02-12 03:40:59", "2015-02-12 03:46:00"),
                slice("2015-02-16 01:34:00", "2015-02-16 01:40:59"),
            ]
        }

        # Data cleaning
        dataset_list = [training, test1, test2]
        for it in range(len(dataset_list)):
            for feature in faulty.keys():
                for time_window in faulty[feature]:
                    dataset_list[it].loc[time_window, feature] = np.nan
            dataset_list[it].update(dataset_list[it].interpolate(method='linear', axis='index'))

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

    dataset['Month'] = dataset.index.month.astype('category')

    return dataset

def load_dataset_graz(data_cleaning=True):
    dataset = pd.read_excel(
        DATA_DIR + 'Graz_occupants.xlsx',
        index_col='datetime',
        parse_dates=True
    )

    if data_cleaning:
        # Remove faulty data
        time_windows_to_exclude = [
            slice(pd.Timestamp('2022-06-22 15:45:00'), pd.Timestamp('2022-07-18 16:05:00'))
        ]

        for time_window in time_windows_to_exclude:
            dataset = dataset.drop(dataset.loc[time_window].index)

    # Convert colums to numeric values
    for column in dataset.iloc[:, 4:]:
        dataset.loc[:, column] = pd.to_numeric(dataset.loc[:, column].str.replace(r'[^0-9-.]+', ''))

    # Translate colum names
    dataset = translate_columns(dataset)

    # Instead of total number of people, 'Occupancy' should indicate if room is occupied or not
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
        'DayOfWeek',
        'Month'
    ]

    #X = X.assign(Weekday=X.index.weekday)
    #for X_test in X_test_list:
    #    X_test.loc[:,'Weekday'] = X_test.index.weekday

    cat_vars = X.columns.intersection(cat_vars)

    """
    Encoding:
    Categorical values are transformed into unique integers.
    """
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

    # Normalizing - Saves storage space / Not working?
    for v in cat_vars:
        X[v] = X[v].astype('category').cat.as_ordered()
        for X_test in X_test_list:
            X_test.loc[:,v] = X_test[v].astype('category').cat.as_ordered()

    # Reshape input: Arrange data in arrays so that it can be read by Keras
    X_array = []
    X_test_array_list = []

    for i, v in enumerate(cat_vars):
        X_array.append(X.loc[:, [v]])
    for X_test in X_test_list:
        X_test_array = []
        for i, v in enumerate(cat_vars):
            X_test_array.append(X_test.loc[:, [v]])
        X_test_array_list.append(X_test_array)

    X_array.append(X.iloc[:, ~X.columns.isin(cat_vars)])
    for X_test, X_test_array in zip(X_test_list, X_test_array_list):
        X_test_array.append(X_test.iloc[:, ~X_test.columns.isin(cat_vars)])

    return X_array, X_test_array_list, encoders

def get_embeddings_shaped(X, X_test_list, columns):
    cat_vars = [
        'Room_ID',
        'DayOfWeek',
        'Month'
    ]

    cat_var_indexes = []
    for cat_var in cat_vars:
        if cat_var in columns:
            cat_var_indexes.append(columns.index(cat_var))

    """
    Encoding:
    Categorical values are transformed into unique integers.
    """
    X_combined = np.concatenate([X] + X_test_list)
    encoders = {}
    for v_id, v in zip(cat_var_indexes, cat_vars):
        le = LabelEncoder()
        le.fit(X_combined[:,:,v_id])
        encoders[v] = le
        X[:,:,v_id] = le.transform(X[:,:,v_id].values)
        for it, X_test in enumerate(X_test_list):
            X_test[it][:,:,v_id] = le.transform(X_test[:,:,v_id])
        print('{0}: {1}'.format(v, le.classes_))

    # Reshape input: Arrange data in arrays so that it can be read by Keras
    X_array = []
    X_test_array_list = []

    for i, v in enumerate(cat_vars):
        X_array.append(X[:,:,v])
    for X_test in X_test_list:
        X_test_array = []
        for i, v in enumerate(cat_vars):
            X_test_array.append(X_test[:,:,v])
        X_test_array_list.append(X_test_array)

    for column_id in range(X.shape[-1]):
        if column_id not in cat_var_indexes:
            X_array.append(X[:,:,column_id])
            for X_test, X_test_array in zip(X_test_list, X_test_array_list):
                X_test_array.append(X_test[:,:,column_id])

    return X_array, X_test_array_list, encoders

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