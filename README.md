# CO2-Occupancy
Using different models (LSTM, GRU, CNN, RNN) to see which one gives best accuracy on predicting the occupancy.
Adding Embedded layers to best model to see if it predicts better with or without these layers.
Visualize plot_embedding for encoders.
Visualize Correlation matrix for the features.
Plot Temperature, CO2, Light distributions by hour.

# Parameters

|      Parameter | Description                                                                                                                                                                                                                                                                                                  |
|---------------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        dataset |  ['uci', 'Australia', 'Denmark', 'Italy']
Which dataset to load in the 'load_dataset()'-method                                                                                                                                                                                                               |
|    feature_set | ['full', 'Light+CO2', 'CO2']
Which features to load from the dataset. Setting this to 'full' will load the features 'Temperature', 'Humidity', 'Light', 'CO2' and 'HumidityRatio', if they exist in the selected dataset.                                                                                    |
|     batch_size |                                                                                                                                                                                                                                                                                                              |
|         epochs |                                                                                                                                                                                                                                                                                                              |
|        repeats | How often to execute the model. Should be set to 10. For the result, mean value and std of all runs are calculated.                                                                                                                                                                                          |
| historical_co2 | [True, False]
Specifies, if historical CO2-values should be added to the feature list. If True, a column  with CO2-values from one minute before each timestep will be added                                                                                                                                 |
|      embedding | Specifies, if embedding layers should be added for the categorical variables. Used categorical variables are the day of the week and, if the dataset is 'Australia', 'Denmark' or 'Italy', the Room ID. This parameter can be used from autoencoder.py and embedding.py. The last one also generates a plot. |
|         shaped | [True, False] Add a lookback and a prediction horizon to each timestep. By default, lookback_horizon is set to 48 and prediction_horizon to 1, which can be changed in utils.py. If true, the new shape of the data is (n_timesteps, 48, 1, n_features).                                                     |

# Models
- CO2+histCO2
Used with models CNN and SRNN

| dataset | feature_set | batch_size | epochs | repeats | historical_co2 | embedding | shaped |
|---------|-------------|------------|--------|---------|----------------|-----------|--------|
| uci     | CO2         | 32         | 50     | 10      | True           | False     | True   |
- Autoencoder embedding with Light+CO2

| dataset | feature_set | batch_size | epochs | repeats | historical_co2 | embedding | shaped |
|:-------:|-------------|------------|--------|---------|----------------|-----------|--------|
| uci     | CO2         | 16         | 50     | 10      | True           | False     | True   |
