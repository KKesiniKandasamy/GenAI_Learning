# Import Pandas and Nump
import pandas as pd
import matplotlib.pyplot as plt

# Create a sample dataframe
data = {'Year': [2020, 2021, 2022, 2023],
        'Sales': [150, 200, 180, 220],
        'Profit': [50, 80, 100, 120]}
df = pd.DataFrame(data)
# Display the dataframe
print("\nSample DataFrame:\n", df)

# Line plot
plt.plot(df['Year'], df['Sales'])
plt.title('Sales Over Years')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.xticks(df['Year'])  # Set x-ticks to the years
plt.show()

#Bar plot
plt.bar(df['Year'], df['Profit'], color='orange')
plt.title('Profit Over Years')
plt.xlabel('Year')  
plt.ylabel('Profit')
plt.xticks(df['Year'])  # Set x-ticks to the years
plt.show()

# Scatter plot
plt.scatter(df['Sales'], df['Profit'], color='green')
plt.title('Sales vs Profit')
plt.xlabel('Sales')
plt.ylabel('Profit')
plt.grid(True)  # Add grid for better readability
plt.show()