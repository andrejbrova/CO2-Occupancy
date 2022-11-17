import pandas as pd
import matplotlib.pyplot as plt

test1 = pd.read_csv('/Users/Brova/Downloads/occupancy_data/datatest.txt')
test1_CO2 = test1['CO2']

fig = plt.figure()
ax = fig.gca()
test1_CO2.hist(ax=ax)
plt.title('CO2 histogram for UCI test1 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()
