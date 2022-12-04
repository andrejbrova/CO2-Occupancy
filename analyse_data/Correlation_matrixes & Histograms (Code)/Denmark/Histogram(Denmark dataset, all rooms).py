import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean

# Denmark dataset has 3 rooms

def main():
    dataset_DateTime = pd.read_csv('/Users/Brova/Downloads/occupancy_data/Denmark/Indoor_Measurement_Study4.csv',
                          index_col='Date_Time', na_values=-999, parse_dates=True)


    dataset1 = pd.read_csv('/Users/Brova/Downloads/occupancy_data/Denmark/Indoor_Measurement_Study4.csv')
    dataset2 = pd.read_csv('/Users/Brova/Downloads/occupancy_data/Denmark/Occupant_Number_Measurement_Study4.csv',
                           index_col='Date_Time', na_values=-999, parse_dates=True)

    dataset = pd.concat([dataset_DateTime, dataset2], axis=1)
    dataset = dataset.sort_index()

    dataset = dataset.loc[:, ~dataset.columns.duplicated()]  # Drop the second Room_ID column
    dataset['Room_ID'] = dataset['Room_ID'].astype('category')

    dataset = translate_columns(dataset)

    dataset = dataset[['Temperature', 'CO2', 'Room_ID', 'Occupancy']]
    dataset = dataset.loc[:, dataset.columns != 'Humidity']
    dataset = dataset.dropna()

    occupant_data = dataset[dataset['Occupancy'] == 1]
    nonoccupant_data = dataset[dataset['Occupancy'] == 0]
    CO2_occupant_data = occupant_data['CO2']
    mean_CO2_occupant_data = mean(CO2_occupant_data)
    CO2_nonoccupant_data = nonoccupant_data['CO2']
    mean_CO2_nonoccupant_data = mean(CO2_nonoccupant_data)

    occupied_data_number = len(occupant_data.index)
    nonoccupied_data_number = len(nonoccupant_data.index)
    occupied_data_percentage = occupied_data_number/(occupied_data_number+nonoccupied_data_number) * 100
    nonoccupied_data_percentage = nonoccupied_data_number/(occupied_data_number+nonoccupied_data_number) * 100

    CO2 = dataset1['Indoor_CO2[ppm]']
    testData = dataset.loc['6/21/2018  12:00:00 AM':'9/14/2018  11:59:00 PM', :]
    trainData = dataset.drop(testData.index)
    df_CO2_test = testData['CO2']   # Indoor_CO2[ppm]
    df_CO2_train = trainData['CO2']   # Indoor_CO2[ppm]

    CO2_mean = mean(CO2)
    CO2_mean_train = mean(df_CO2_train)
    CO2_mean_test = mean(df_CO2_test)


    # Different plots for the 3 different rooms
    for i in range(1,4):
        df_Room = dataset[dataset['Room_ID'] == i]
        df_CO2_Room = df_Room['CO2']
        print(df_CO2_Room)


    plt.hist([df_CO2_train, df_CO2_test], color=['Blue', 'Yellow'], label=['Other seasons', 'Summer season'])
    plt.title('CO2 histogram for Denmark dataset')
    plt.xlabel('CO2')
    plt.ylabel('Frequency')
    plt.annotate('Mean CO2: %.1f ppm' % CO2_mean, xy=(1000, 50000))
    plt.annotate('Mean CO2 for summer: %.1f ppm' % CO2_mean_test, xy=(1000, 48000))
    plt.annotate('Mean CO2 for other seasons: %.1f ppm' % CO2_mean_train, xy=(1000, 46000))
    plt.annotate('Mean CO2 for Occupied data points: %.1f ppm' % mean_CO2_occupant_data, xy=(1000, 44000))
    plt.annotate('Mean CO2 for Non-Occupied data points: %.1f ppm' % mean_CO2_nonoccupant_data, xy=(1000, 42000))
    plt.annotate('Occupied data percentage: %.1f percent' % occupied_data_percentage, xy=(1000, 40000))
    plt.annotate('Non-Occupied data percentage: %.1f percent' % nonoccupied_data_percentage, xy=(1000, 38000))
    plt.legend()
    plt.show()


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


if __name__ == '__main__':
    main()