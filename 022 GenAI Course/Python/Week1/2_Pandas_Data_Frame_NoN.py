# Importaing Pandas and Numpy
import pandas as pd
import numpy as np

data= {"Product": ["A", "B", "C", "D", "E", "B", "C"],
        "Price": [10, np.nan, 30, np.nan, 50, 30, 20],
        "Quantity": [100, 200, 300, 400, 500, 300, 200]}
df = pd.DataFrame(data)
print("\nOriginal DataFrame:\n", df)

#Handling missing values Filling NaN values with a specific value
df_filled = df.fillna(-1)
print("\nDataFrame after filling NaN values with -1:\n", df_filled)

#Handling missing values - Filling NaN values with the mean of the column
df_mean_filled = df.fillna(df.mean(numeric_only=True))
print("\nDataFrame after filling NaN values in 'Price' with the mean:\n", df_mean_filled)

#Handling missing values - Dropping rows with NaN values
df_dropped = df.dropna()
print("\nDataFrame after dropping rows with NaN values:\n", df_dropped)

# Handling duplicates
print("\nOriginal DataFrame with duplicates:\n", df)
df_no_duplicates_product = df.drop_duplicates(subset=['Product'])
print("\nDataFrame after removing duplicates based on 'Product':\n", df_no_duplicates_product)