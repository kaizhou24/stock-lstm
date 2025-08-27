from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

API_KEY = 'KOSRKMA8X6FK2B86'
TICKER = 'AAL'
URL_STRING = None
FILE_TO_SAVE = None