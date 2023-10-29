import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


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


def SVM_l2_regularization_Kfold(x_train, y_train):
    class_weights = {-1: 1, 1: 5}

    k = 5  # Number of folds for cross-validation
    C_values = [0.001, 0.1, 1, 10]
    mean_scores = []

    for c in C_values:
        svm_classifier = SVC(kernel='linear', C=c, class_weight=class_weights)
        kf = KFold(n_splits=k)
        scores = cross_val_score(svm_classifier, x_train, y_train, cv=kf)
        mean_accuracy = np.mean(scores)
        mean_scores.append(mean_accuracy)
        print(f'C={c}: Mean Accuracy = {mean_accuracy:.2f}')
    plt.figure(figsize=(8, 6))
    plt.plot(C_values, mean_scores, marker='o', linestyle='-')
    plt.title("Mean Accuracy vs. C Values")
    plt.xlabel("C Values")
    plt.ylabel("Mean Accuracy")
    plt.xscale('log')  # Use a log scale for the x-axis if C values vary widely
    plt.grid(True)
    plt.show()
    return mean_scores


def main():
    # Your main program logic goes here
    x_train, y_train, x_test, y_test = loading_data()
    y_train = y_train.values.ravel()
    mean_scores = SVM_l2_regularization_Kfold(x_train, y_train)
    extra_columns_in_test = [col for col in x_train.columns if col not in x_test.columns]

    # Drop these extra columns from the training dataset
    x_train.drop(extra_columns_in_test, axis=1, inplace=True)

    # svm_classifier.fit(x_train, y_train)
    #
    # # Make predictions on the test data
    # y_pred = svm_classifier.predict(x_test)
    #
    # # Calculate and print the accuracy of the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {accuracy:.2f}')

    # *****************************************


if __name__ == "__main__":
    main()
