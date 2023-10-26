import pandas as pd
import numpy as np



def loading_data():
    df = pd.read_csv("test_X.csv")
    return df



def main():
    # Your main program logic goes here
    df = loading_data()
    print(df.head(5))

if __name__ == "__main__":
    main()
