import pathlib
import numpy as np
import pandas as pd
directory = pathlib.Path(__file__).parent

def load_dataset():
    features = [
        'Temperature',
        'Humidity',
        'Light',
        'CO2',
        'HumidityRatio'
    ]

    training = pd.read_csv('/Users/Brova/Downloads/occupancy_data/datatraining.txt', parse_dates=['date'])
    test1 = pd.read_csv('/Users/Brova/Downloads/occupancy_data/datatest.txt')
    test2 = pd.read_csv('/Users/Brova/Downloads/occupancy_data/datatest2.txt')

    training = training.set_index('date')

    X_train = training.loc[:, features]
    y_train = pd.DataFrame(training.loc[:, 'Occupancy'])

    X_test_1 = test1.loc[:, features]
    y_test_1 = pd.DataFrame(test1.loc[:, 'Occupancy'])

    X_test_2 = test2.loc[:, features]
    y_test_2 = pd.DataFrame(test2.loc[:, 'Occupancy'])

    X_test_combined = pd.concat([X_test_1, X_test_2])
    y_test_combined = pd.concat([y_test_1, y_test_2])

    return X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined

def summarize_results(scores_train, scores_test_1, scores_test_2, scores_test_combined, model_name):
    print(scores_train)
    print(scores_test_1)
    print(scores_test_2)
    print(scores_test_combined)
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
    pd.DataFrame(result, index=[model_name]).to_csv(model_name + '.csv')
    print(result)