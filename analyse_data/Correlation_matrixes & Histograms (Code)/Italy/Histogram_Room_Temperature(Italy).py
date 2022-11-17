import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/Brova/Downloads/occupancy_data/Italy/Indoor_Measurement_Study10.csv')
df_Temperature = dataset['Indoor_Temp[C]']
df_Temperature = df_Temperature[(df_Temperature >= 0)]

fig = plt.figure()
ax = fig.gca()
df_Temperature.hist(ax=ax)
plt.title('Room Temperature Histogram for Italy dataset')
plt.xlabel('Indoor Temperature')
plt.ylabel('Frequency')
plt.show()