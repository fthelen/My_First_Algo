import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('VFSTX', index_col='Date')
print(df['Adj Close'].head())

plt.plot(df['Adj Close'])
plt.show()