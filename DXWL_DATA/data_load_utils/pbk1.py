import numpy as np
import pandas as pd
from scipy.io import arff

data, meta = arff.loadarff('../InsectWingbeat/InsectWingbeat_TEST.arff')
df = pd.DataFrame(data)
print(df.head())
df.to_csv('InsectWingbeat.csv')
# print(df.head().to_numpy())