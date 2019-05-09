import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

df = pd.read_csv(r'tests\SP500.csv', parse_dates=True, index_col=0)
print(df['Adj Close'].head())
print(df['Adj Close'].tail())

print(df.shape)

df['log_ret'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
print(df['log_ret'].head())

# style.use('ggplot')
# df['Adj Close'].plot()
# plt.show()
