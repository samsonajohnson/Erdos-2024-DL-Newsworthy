import pandas as pd
import pmdarima
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import time

from seaborn import set_style
set_style("whitegrid")

def get_baseline(file):
    stock_df = pd.read_csv(f"Stock_data/{file}.csv")
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True)

    # splitting the data into the train and test sets
    stock_train, stock_test = _train_test_split(stock_df)

    # plotting the open price
    plt.plot(stock_train.Date,
             stock_train.Open)
    plt.title(file +" Open Price")
    plt.show()

    # plotting the first differences
    stock_train['First_diff'] = stock_train['Open'].diff()
    sns.lineplot(data=stock_train.iloc[1:],
                x='Date',
                y='First_diff')
    plt.title(file+" Open Price First Difference")
    plt.show()

    # plotting the autocorrelation and partial autocorrelation
    plot_acf(stock_train.Open)
    plot_acf(stock_train.First_diff.iloc[1:])
    plot_pacf(stock_train.First_diff.iloc[1:])
    plt.show()

    return pmdarima.auto_arima(stock_train.Open)



def _train_test_split(df):
    train = df.loc[df.Date < '2023-03-01']
    test = df.drop(train.index)
    return train, test
