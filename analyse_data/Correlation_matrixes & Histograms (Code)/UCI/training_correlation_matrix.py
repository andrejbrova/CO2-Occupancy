import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

train = pd.read_csv('/Users/Brova/Downloads/occupancy_data/datatraining.txt')
train = train.set_index('date')

trainFrame = pd.DataFrame(train, columns=["Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy"])
corrMatrix = trainFrame.corr()
print(corrMatrix)

sn.heatmap(corrMatrix, annot=True)
plt.show()
