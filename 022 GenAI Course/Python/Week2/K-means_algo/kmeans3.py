# Dependencies and libraries

import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
file_path = "celebs2.csv"

# We just use Height and Weight for clustering.
# X is our feature matrix, can be seen here
train = pd.read_csv(file_path)
type(train)
X = train.values
print(type(X))
print(X)
X = X[:,0:2]
print("Print X after slicing to first two columns:")
print(X)
