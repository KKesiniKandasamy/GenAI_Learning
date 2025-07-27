# Importaing Pandas and Matplotlib
import pandas
import pandas as pd
import matplotlib
#import matplotlib.pyplot as plt

# Displaying the version of Pandas and Matplotlib
#print("Pandas version:", pd.__version__)
#print("Matplotlib version:", plt.__version__)
print("Pandas Version:", pandas.__version__)
print("MatPlotlib Version:", matplotlib.__version__) 

# Label-based indexing in Pandas
s=pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print("Label-based indexing (Series):")
print(s['b'])  # Accessing by label

#Dataframe example
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)

# Label-based indexing in DataFrame - Accessing a row using loc
print("\nLabel-based indexing (DataFrame - row):")
print(df.loc[1])  # Accessing the row with index label 1
# Label-based indexing in DataFrame - Accessing a column
print("\nLabel-based indexing (DataFrame - column):")
print(df['Name'])  # Accessing the 'Age' column