import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('/Users/Brova/Downloads/occupancy_data/datatraining.txt')
train_CO2 = train['CO2']

fig = plt.figure()
ax = fig.gca()
train_CO2.hist(ax=ax)
plt.title('CO2 histogram for UCI training dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()
