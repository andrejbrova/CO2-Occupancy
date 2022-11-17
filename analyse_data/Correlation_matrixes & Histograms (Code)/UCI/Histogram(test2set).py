import pandas as pd
import matplotlib.pyplot as plt

test2 = pd.read_csv('/Users/Brova/Downloads/occupancy_data/datatest2.txt')
test2_CO2 = test2['CO2']

fig = plt.figure()
ax = fig.gca()
test2_CO2.hist(ax=ax)
plt.title('CO2 histogram for UCI test2 dataset')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.show()
