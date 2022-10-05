# CO2-Occupancy
Using different models (LSTM, GRU, CNN, RNN) to see which one gives best accuracy on predicting the occupancy.
Adding Embedded layers to best model to see if it predicts better with or without these layers.
Visualize plot_embedding for encoders.
Visualize Correlation matrix for the features.
Plot Temperature, CO2, Light distributions by hour.

# Parameters


|      Parameter | Description                                                                                                                                                                                                                                                                                           |
|---------------:|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        dataset |  ['uci', 'Australia', 'Denmark', 'Italy'] Which dataset to load in the 'load_dataset()'-method                                                                                                                                                                                                        |
|    feature_set | ['full', 'Light+CO2', 'CO2'] Which features are to be loaded from the dataset. With the setting 'full', the features 'Temperature', 'Humidity', 'Light', 'CO2' and 'HumidityRatio' are loaded, provided they are available in the selected dataset.                                                   |
|     batch_size |                                                                                                                                                                                                                                                                                                       |
|         epochs |                                                                                                                                                                                                                                                                                                       |
|        repeats | How often the model should be run. Should be set to 10. For the result, mean value and standard deviation of all runs are calculated.                                                                                                                                                                 |
| historical_co2 | [True, False] Specifies whether historical CO2-values should be added to the feature list. If True, a column  with CO2-values from one minute before each timestep will be added                                                                                                                      |
|      embedding | Specifies whether to add embedding layers for the categorical variables. Used categorical variables are the day of the week and, if the dataset is 'Australia', 'Denmark' or 'Italy', the Room ID. This parameter can be used from autoencoder.py and embedding.py. The latter also generates a plot. |
|         shaped | [True, False] Adding a lookback and a prediction horizon to each timestep. By default, lookback_horizon is set to 48 and prediction_horizon to 1, which can be changed in utils.py. If true, the new shape of the data is (n_timesteps, 48, 1, n_features).                                           |

# Models
- CO2+histCO2
Used with models CNN, LSTM, SRNN and GRU

| dataset | feature_set | batch_size | epochs | repeats | historical_co2 | embedding | shaped |
|---------|-------------|------------|--------|---------|----------------|-----------|--------|
| uci     | CO2         | 32         | 50     | 10      | True           | False     | True   |

- Autoencoder
These models no longer use PCA or TSNE for the representation. Instead, the size of the encoded layer is 2, so it can be used for the representation itself.

| dataset | feature_set | batch_size | epochs | repeats | historical_co2 | embedding | shaped |
|:-------:|-------------|------------|--------|---------|----------------|-----------|--------|
| uci     | Light+CO2   | 16         | 30     | 10      | False          | True      | False  |
| uci     | full        | 16         | 30     | 10      | False          | False     | False  |

- New datasets (Australia, Denmark, Italy)
Used with models CNN, LSTM and Autoencoder (TODO)
Italy doesn't include the Humidity-feature because of the large amount of nan-values.
0.33 test size with shuffle=False

| dataset   | feature_set | batch_size | epochs | repeats | historical_co2 | embedding | shaped |
|:---------:|-------------|------------|--------|---------|----------------|-----------|--------|
| Australia | full        | 32         | 50     | 10      | False          | True      | False  |
| Denmark   | full        | 32         | 50     | 10      | False          | True      | False  |
| Italy     | full        | 32         | 50     | 10      | False          | True      | False  |
