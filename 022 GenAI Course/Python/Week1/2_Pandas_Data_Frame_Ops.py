# Importaing Pandas and Nump
import pandas as pd
import numpy as np


#Dataframe example
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Ethan'],
    'Age': [25, 30, 35, 28, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Score': [88, 92, 85, 90, np.nan]
}
df = pd.DataFrame(data)
print("\nOriginal DataFrame:\n", df)

# Filtering data
print("\nFiltering DataFrame where Age > 30:\n", df[df['Age'] > 30])

# Filtering with multiple conditions
print("\nFiltering DataFrame where Age > 30 and City is 'Chicago':\n", df[(df['Age'] > 30) & (df['City'] == 'Chicago')])

# Sorting data
print("\nSorting DataFrame by Age:\n", df.sort_values(by='Age'))
# Sorting data by Name
print("\nSorting DataFrame by Name:\n", df.sort_values(by='Name'))

# Sorting data by multiple columns
print("\nSorting DataFrame by Age and then by Name:\n", df.sort_values(by=['Age', 'Name']))

# Adding a new column
# =======================================================
df['Grade'] = ['B', 'A', 'B', 'A+', 'B-']
print("\nDataFrame after adding a new column 'Grade':\n", df)   

# Adding a new column based on a condition
# =======================================================   
df['Senior'] = df['Age'] > 30
print("\nDataFrame after adding a new column 'Senior':\n", df)

# Removing a column
# =======================================================
df.drop(columns=['Senior'], inplace=True)
print("\nDataFrame after removing the 'Senior' column:\n", df)