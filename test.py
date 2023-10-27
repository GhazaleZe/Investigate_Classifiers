import pandas as pd

# Sample data (replace this with your dataset)
data = [
    (1, 0, 1),
    (1, 1, 0),
    (2, 0, 0),
    (2, 1, 1),
    (3, 0, 1),
]

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=["instance", "feature", "value"])

# Pivot the DataFrame to transform it into the desired format
df_pivot = df.pivot(index="instance", columns="feature", values="value").fillna(0)

# Reset the index to make "instance" a regular column
df_pivot.reset_index(inplace=True)

# Rename the columns for clarity (0 and 1 instead of 0.0 and 1.0)
df_pivot.columns = [f"feature_{col}" if col != "instance" else col for col in df_pivot.columns]

# Optional: Convert "instance" column to an integer (if it's not already)
df_pivot["instance"] = df_pivot["instance"].astype(int)

# Optional: Set the "instance" column as the DataFrame's index
# df_pivot.set_index("instance", inplace=True)

print(df_pivot)
