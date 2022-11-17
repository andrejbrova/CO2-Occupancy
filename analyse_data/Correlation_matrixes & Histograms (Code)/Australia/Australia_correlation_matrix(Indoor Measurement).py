import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/Brova/Downloads/occupancy_data/Australia/Indoor_Measurement_Study7.csv')
dataset = dataset.set_index('Date_Time')

DataFrame = pd.DataFrame(dataset, columns=["Indoor_Temp[C]", "Indoor_RH[%]", "Indoor_CO2[ppm]", "Room_ID"])
corrMatrix = DataFrame.corr()
print(corrMatrix)

sn.heatmap(corrMatrix, annot=True)
plt.show()
