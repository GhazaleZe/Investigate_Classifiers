import pandas as pd
import numpy as np

# Define the number of rows and columns
num_rows = 10
num_cols = 5

# Create a DataFrame with random values
data = np.random.rand(num_rows, num_cols)

# Define column names
column_names = ["Column1", "Column2", "Column3", "Column4", "Column5"]

# Create the DataFrame
df = pd.DataFrame(data, columns=column_names)

selected_features = [1,2]

# Display the DataFrame
print(df.iloc[:, selected_features])
