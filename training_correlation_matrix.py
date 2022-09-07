# THIS IS NOT EMBEDDED LAYERS, BUT RATHER THE CORRELATION MATRIX
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


training = pd.read_csv('/Users/Brova/Downloads/occupancy_data/datatraining.txt', parse_dates=['date'])
training = training.set_index('date')


trainFrame = pd.DataFrame(training, columns=["Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"])
corrMatrix = trainFrame.corr()
print(corrMatrix)

sn.heatmap(corrMatrix, annot=True)
plt.show()
