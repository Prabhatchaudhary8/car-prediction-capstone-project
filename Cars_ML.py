# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# reading Dataset
df=pd.read_csv("CAR DETAILS.csv")
df.head(5)
# Data Cleaning
## Understanding the data by looking upto the columns 
df.columns
df.info()
# Checking null values
df.isnull().sum()
# checking weather duplicates are available or not and droping them
df.duplicated().sum()
df.drop_duplicates(keep=False,inplace=True)
df.duplicated().sum()   # checking again weather duplicates are still available or not.
df.dtypes # showing different columns catagorical dtta type which is difficult to handle so we will try to encode it.
# Detecting outliers and handling them

df['fuel'].value_counts()
df2=df.copy()
from sklearn.preprocessing import LabelEncoder # importing label encoder to encode fuel and owner type column