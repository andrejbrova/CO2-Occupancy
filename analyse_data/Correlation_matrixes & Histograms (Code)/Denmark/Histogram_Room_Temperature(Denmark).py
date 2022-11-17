import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/Brova/Downloads/occupancy_data/Denmark/Indoor_Measurement_Study4.csv')
df_Temperature = dataset['Indoor_Temp[C]']

fig = plt.figure()
ax = fig.gca()
df_Temperature.hist(ax=ax)
plt.title('Room Temperature Histogram for Denmark dataset')
plt.xlabel('Indoor Temperature')
plt.ylabel('Frequency')
plt.show()
