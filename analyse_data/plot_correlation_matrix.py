import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).parents[1])
ANALYSIS_DIR = ROOT_DIR + '/analyse_data/'
sys.path.append(ROOT_DIR)

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from get_data import load_dataset, get_readings_data


def plot_correlation_matrix(location='Denmark'):
    datasets = load_dataset(
        dataset=location,
        feature_set='full',
        historical_co2=False,
        normalize=False,
        embedding=False,
        shaped=False,
        split_data=False,
        data_cleaning=False
    )

    tests_combined = pd.concat([test1, test2])


    testsFrame = pd.DataFrame(datasets[1], columns=["Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"])
    corrMatrix = testsFrame.corr()
    print(corrMatrix)

    sn.heatmap(corrMatrix, annot=True)
    plt.show()
