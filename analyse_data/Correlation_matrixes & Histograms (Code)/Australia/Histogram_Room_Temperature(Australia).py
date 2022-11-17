import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/Brova/Downloads/occupancy_data/Australia/Indoor_Measurement_Study7.csv')
df_Temperature = dataset['Indoor_Temp[C]']

fig = plt.figure()
ax = fig.gca()
df_Temperature.hist(ax=ax)
plt.title('Room Temperature Histogram for Australia dataset')
plt.xlabel('Indoor Temperature')
plt.ylabel('Frequency')
plt.show()
