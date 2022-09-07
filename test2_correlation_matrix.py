# THIS IS NOT EMBEDDED LAYERS, BUT RATHER THE CORRELATION MATRIX
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

test2 = pd.read_csv('/Users/Brova/Downloads/occupancy_data/datatest2.txt')
test2 = test2.set_index('date')


testFrame2 = pd.DataFrame(test2, columns=["Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"])
corrMatrix = testFrame2.corr()
print(corrMatrix)

sn.heatmap(corrMatrix, annot=True)
plt.show()
