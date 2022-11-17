import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

test1 = pd.read_csv('/Users/Brova/Downloads/occupancy_data/datatest.txt')
test1 = test1.set_index('date')
test2 = pd.read_csv('/Users/Brova/Downloads/occupancy_data/datatest2.txt')
test2 = test2.set_index('date')

tests_combined = pd.concat([test1, test2])


testsFrame = pd.DataFrame(tests_combined, columns=["Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"])
corrMatrix = testsFrame.corr()
print(corrMatrix)

sn.heatmap(corrMatrix, annot=True)
plt.show()
