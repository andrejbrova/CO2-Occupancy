import pandas as pd
import matplotlib.pyplot as plt

test1 = pd.read_csv('/Users/Brova/Downloads/occupancy_data/datatest.txt')
test2 = pd.read_csv('/Users/Brova/Downloads/occupancy_data/datatest2.txt')
tests_combined = pd.concat([test1, test2])
tests_CO2 = tests_combined['CO2']

fig = plt.figure()
ax = fig.gca()
tests_CO2.hist(ax=ax)
plt.title('CO2 histogram for UCI combined tests dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()
