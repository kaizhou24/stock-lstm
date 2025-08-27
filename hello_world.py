from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import requests, json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

API_KEY = 'KOSRKMA8X6FK2B86'
TICKER = 'AAL'
URL = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={TICKER}&interval=5min&apikey={API_KEY}'

r = requests.get(URL)
data = r.json()
df = pd.DataFrame(data)

file_to_save = f'{TICKER}_data.csv'
df.to_csv(file_to_save, index=False)