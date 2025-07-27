import files
import io
import pandas as pd
import matplotlib.pyplot as plt

uploaded_file = files.upload()
file_name = list(uploaded_file.keys())[0]

data_dict = pd.read_csv(io.BytesIO(uploaded_file[file_name]))
print("\nSample DataFrame:\n", data_dict)
