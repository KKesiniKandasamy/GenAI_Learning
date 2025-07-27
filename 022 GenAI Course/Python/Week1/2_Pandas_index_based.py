# Importaing Pandas and Matplotlib
import pandas
import pandas as pd


# Series example
s=pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])

print("\n Integer-based indexing (Series):")
# print(s[1])  # Accessing by label  - old method throws error

print(s.iloc[1])  # Accessing by integer position
#Dataframe example
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)

# Integer-based indexing in DataFrame - Accessing a row using iloc
# =======================================================
print("Integer-based indexing (DataFrame - row): Index 1")
print(df.iloc[1])  # Accessing the row with index label 1


# Boolean-based indexing in DataFrame - Accessing rows based on a condition
# =======================================================   
print("\nBoolean-based indexing (DataFrame): if Age > 28")
print(df[df['Age'] > 28])  # Accessing rows where Age is greater

# Column-based indexing in DataFrame - Accessing a column using loc
# =======================================================   
print("\nColumn-based indexing (DataFrame): Column 'City'")
print(df.loc[:, 'City'])  # Accessing the 'Name' column