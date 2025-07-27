import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('student_data.csv')

# Create new columns based on scores Excellent: 90-100 - Good: 80-89 - Average: 70-79 - Below Average: below 70 
df['Performance'] = pd.cut(df['score'],
                            bins=[0, 69, 79, 89, 100],
                            labels=['Below Average', 'Average', 'Good', 'Excellent'])  
print("\nData with Performance Categories:\n", df)

# Filter data to show only student with scores above 80
high_performers = df[df['score'] > 80]
print("\nHigh Performers:\n", high_performers)

# Check for missing value and display the count of missing values
missing_values = df.isnull().sum()
print("\nMissing Values in Each Column:\n", missing_values)

