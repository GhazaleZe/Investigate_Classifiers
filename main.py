import pandas as pd
import numpy as np


def loading_data():
    # x train
    custom_column_names = ["instance", "feature", "value"]
    df_train_x = pd.read_csv("train_X.csv", names=custom_column_names)
    df_pivot = df_train_x.pivot(index="instance", columns="feature", values="value").fillna(0)

    # Reset the index to make "instance" a regular column
    df_pivot.reset_index(inplace=True)

    # Rename the columns for clarity (0 and 1 instead of 0.0 and 1.0)
    df_pivot.columns = [f"feature_{col}" if col != "instance" else col for col in df_pivot.columns]

    # Optional: Convert "instance" column to an integer (if it's not already)
    df_pivot["instance"] = df_pivot["instance"]
    # y train
    custom_Y_names = ["poisonous"]
    df_train_Y = pd.read_csv("train_Y.csv", names=custom_Y_names)

    # x test
    df_test_x = pd.read_csv("test_X.csv", names=custom_column_names)
    df_pivot_test = df_test_x.pivot(index="instance", columns="feature", values="value").fillna(0)

    # Reset the index to make "instance" a regular column
    df_pivot_test.reset_index(inplace=True)

    # Rename the columns for clarity (0 and 1 instead of 0.0 and 1.0)
    df_pivot_test.columns = [f"feature_{col}" if col != "instance" else col for col in df_pivot_test.columns]

    # Optional: Convert "instance" column to an integer (if it's not already)
    df_pivot_test["instance"] = df_pivot_test["instance"]

    # y test

    df_test_Y = pd.read_csv("test_Y.csv", names=custom_Y_names)

    return df_pivot, df_train_Y, df_pivot_test, df_test_Y


def main():
    # Your main program logic goes here
    x_train, y_train, x_test, y_test = loading_data()
    #print(x_train)


if __name__ == "__main__":
    main()
