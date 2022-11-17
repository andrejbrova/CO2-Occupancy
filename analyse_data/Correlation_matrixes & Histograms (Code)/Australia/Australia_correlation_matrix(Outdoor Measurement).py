import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/Brova/Downloads/occupancy_data/Australia/Outdoor_Measurement_Study7.csv')
dataset = dataset.set_index('Date_Time')

DataFrame = pd.DataFrame(dataset, columns=["Outdoor_Temp[C]", "Outdoor_RH[%]", "Wind_Speed[m/s]", "Wind_Direction[deg]", "Solar_Radiation[w/m2]", "Precipitation"])
corrMatrix = DataFrame.corr()
print(corrMatrix)

sn.heatmap(corrMatrix, annot=True)
plt.show()
