import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

directory = pathlib.Path(__file__).parent

def load_dataset():
    features = [
        'Temperature',
        'Humidity',
        'Light',
        'CO2',
        'HumidityRatio'
    ]

    training = pd.read_csv(str(directory) + '/occupancy_data/datatraining.txt', parse_dates=['date'])
    test1 = pd.read_csv(str(directory) + '/occupancy_data/datatest.txt', parse_dates=['date'])
    test2 = pd.read_csv(str(directory) + '/occupancy_data/datatest2.txt', parse_dates=['date'])

    training = training.set_index('date')
    test1 = test1.set_index('date')
    test2 = test2.set_index('date')

    X_train = training.loc[:,features]
    y_train = pd.DataFrame(training.loc[:,'Occupancy'])

    X_test_1 = test1.loc[:,features]
    y_test_1 = pd.DataFrame(test1.loc[:,'Occupancy'])

    X_test_2 = test2.loc[:,features]
    y_test_2 = pd.DataFrame(test2.loc[:,'Occupancy'])

    X_test_combined = pd.concat([X_test_1, X_test_2])
    y_test_combined = pd.concat([y_test_1, y_test_2])

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

def summarize_results(scores_train, scores_test_1, scores_test_2, scores_test_combined, model_name):
    print(scores_train)
    print(scores_test_1)
    result = {
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