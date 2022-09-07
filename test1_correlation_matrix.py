# THIS IS NOT EMBEDDED LAYERS, BUT RATHER THE CORRELATION MATRIX
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

test1 = pd.read_csv('/Users/Brova/Downloads/occupancy_data/datatest.txt')
test1 = test1.set_index('date')


testFrame1 = pd.DataFrame(test1, columns=["Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"])
corrMatrix = testFrame1.corr()
print(corrMatrix)

sn.heatmap(corrMatrix, annot=True)
plt.show()
