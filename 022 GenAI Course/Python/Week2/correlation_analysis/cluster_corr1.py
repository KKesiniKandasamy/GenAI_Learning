
import pandas as pd
import numpy as np


# Load the dataset
file_path = "celebs2_corr.xlsx"
df=pd.read_excel(file_path, sheet_name="celebs2")
print("\nDataFrame loaded from Excel:\n", df)

# Display the first rows of the DataFrame
print("\nFirst few rows of the DataFrame:\n", df.head())

# check for numerical columns
numeric_df= df.select_dtypes(include=[np.number])
print("\nNumerical columns in the DataFrame:\n", numeric_df.columns)

# Compute the correlation matrix for numerical variables
corr_matrix = numeric_df.corr()
print("\nCorrelation matrix:\n", corr_matrix)

# Save the correlation matrix to a new Excel file
output_file_path = "celebs2_corr_output.xlsx"
corr_matrix.to_excel(output_file_path, index=True)
print(f"\nCorrelation matrix saved to {output_file_path}")

