import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style




df = pd.read_csv('VFSTX', parse_dates=True, index_col=0)
print(df['Adj Close'].head())
print(df.describe())
print(df.shape)

style.use('ggplot')
df['Adj Close'].plot()
plt.show()
